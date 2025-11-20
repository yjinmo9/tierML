"""2단계: 피처 생성."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .data_utils import load_train_test
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)

ENABLE_SIZE_RISK = os.getenv("ENABLE_SIZE_RISK_SCORE", "0").lower() in {"1", "true", "yes"}
ENABLE_MASS_SIZE_INTERACTION = os.getenv("ENABLE_MASS_SIZE_INTERACTION", "0").lower() in {"1", "true", "yes"}
ENABLE_PLANT_RISK = os.getenv("ENABLE_PLANT_RISK_FEATURES", "0").lower() in {"1", "true", "yes"}
ENABLE_PILOT_PARAM_ZSCORE = os.getenv("ENABLE_PILOT_PARAM_ZSCORE", "0").lower() in {"1", "true", "yes"}
PILOT_ZSCORE_TOPK = int(os.getenv("PILOT_ZSCORE_TOPK", "5"))
ENABLE_SEQ_GRAD = os.getenv("ENABLE_SEQ_GRAD_FEATURES", "0").lower() in {"1", "true", "yes"}


def _compute_outlier_flags(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    flag_arrays = []
    for col in cols:
        series = df[col]
        if series.dropna().empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if np.isclose(iqr, 0):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_arrays.append(((series < lower) | (series > upper)).astype(int))
    if not flag_arrays:
        return pd.DataFrame(index=df.index)
    return pd.concat(flag_arrays, axis=1).fillna(0).astype(int)


def _compute_plant_outlier_rate(
    df: pd.DataFrame,
    plant_col: str,
    numeric_cols: Iterable[str],
) -> dict | None:
    if plant_col not in df.columns or not numeric_cols:
        return None
    flag_df = _compute_outlier_flags(df, numeric_cols)
    if flag_df.empty:
        return None
    row_flag = flag_df.sum(axis=1) > 0
    rates = row_flag.groupby(df[plant_col]).mean()
    overall = float(row_flag.mean())
    return {"rates": rates, "overall": overall}


def _normalize_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    min_val = series.min()
    max_val = series.max()
    denom = max_val - min_val
    if np.isclose(denom, 0):
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / denom


def _compute_pilot_param_stats(
    df: pd.DataFrame,
    pilot_col: str,
    param_cols: list[str],
    topk: int,
) -> dict | None:
    if pilot_col not in df.columns or not param_cols:
        return None
    std_series = df[param_cols].std().sort_values(ascending=False)
    selected_cols = std_series.head(topk).index.tolist()
    if not selected_cols:
        return None
    pilot_groups = df.groupby(pilot_col)
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for col in selected_cols:
        group_mean = pilot_groups[col].mean()
        group_std = pilot_groups[col].std().replace(0, np.nan)
        stats[col] = {
            "mean": group_mean.to_dict(),
            "std": group_std.to_dict(),
            "global_mean": float(df[col].mean()),
            "global_std": float(df[col].std() or 1.0),
        }
    return stats

def _prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()],
        key=lambda col: int(col[len(prefix) :]),
    )


def _stat_features(matrix: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": matrix.mean(axis=1),
        f"{prefix}_std": matrix.std(axis=1),
        f"{prefix}_min": matrix.min(axis=1),
        f"{prefix}_max": matrix.max(axis=1),
        f"{prefix}_q25": np.quantile(matrix, 0.25, axis=1),
        f"{prefix}_q75": np.quantile(matrix, 0.75, axis=1),
    }


def _pca_features(matrix: np.ndarray, prefix: str, n_components: int = 8) -> Dict[str, np.ndarray]:
    n_components = min(n_components, matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(matrix)
    return {f"{prefix}_pca_{i}": transformed[:, i] for i in range(transformed.shape[1])}


def _tensor_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = _prefixed_columns(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    matrix = df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    features = {**_stat_features(matrix, prefix), **_pca_features(matrix, prefix)}
    if prefix == "p":
        segments = np.array_split(matrix, 4, axis=1)
        if len(segments) >= 3:
            features["p_diff_13"] = segments[0].mean(axis=1) - segments[2].mean(axis=1)
        features["p_peak"] = matrix.max(axis=1)
    return pd.DataFrame(features)


def _compute_bin_edges(series: pd.Series, n_bins: int = 5) -> np.ndarray:
    if series.dropna().empty:
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = series.quantile(quantiles).to_numpy(dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def _assign_bins(series: pd.Series, edges: np.ndarray) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(-1, index=series.index)
    return pd.cut(series, bins=edges, labels=False, include_lowest=True).astype("Int64").fillna(-1)


def _build_feature_context(train_df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Dict]:
    context: Dict[str, Dict] = {}
    target_binary = None
    if config.target_column in train_df.columns:
        target_binary = (train_df[config.target_column] != "Good").astype(float)

    size_risk_info: Dict[str, Dict[str, object]] = {}

    for col in ("Width", "Aspect", "Inch"):
        if col in train_df.columns:
            context[f"{col.lower()}_edges"] = _compute_bin_edges(train_df[col], n_bins=5)
            if ENABLE_SIZE_RISK and target_binary is not None:
                bins = _assign_bins(train_df[col], context[f"{col.lower()}_edges"])
                rates = target_binary.groupby(bins).mean()
                overall = target_binary.mean()
                risk_bins = [
                    int(bin_value)
                    for bin_value, rate in rates.items()
                    if pd.notna(bin_value) and rate >= overall
                ]
                size_risk_info[col] = {
                    "risk_bins": risk_bins,
                    "rates": rates.to_dict(),
                    "threshold": float(overall),
                }
    if ENABLE_SIZE_RISK and size_risk_info:
        context["size_risk_info"] = size_risk_info
    proc_cols = [
        col
        for col in train_df.columns
        if col.startswith("Proc_Param") and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    context["proc_cols"] = proc_cols
    if "Plant" in train_df.columns:
        plant_groups = train_df.groupby("Plant")
        context["plant_proc_mean"] = plant_groups[proc_cols].mean() if proc_cols else pd.DataFrame()
        context["plant_proc_std_mean"] = (
            plant_groups[proc_cols].std().mean(axis=1) if proc_cols else pd.Series(dtype=float)
        )
        if config.target_column in train_df.columns:
            target_binary = (train_df[config.target_column] != "Good").astype(float)
            context["plant_ng_rate"] = target_binary.groupby(train_df["Plant"]).mean()
            context["overall_ng_rate"] = target_binary.mean()
        if ENABLE_PLANT_RISK:
            outlier_info = _compute_plant_outlier_rate(train_df, "Plant", proc_cols)
            if outlier_info is not None:
                context["plant_outlier_rate_map"] = outlier_info

    if ENABLE_PLANT_RISK and "Plant" in train_df.columns:
        risk_components = []
        ng_series = context.get("plant_ng_rate")
        if ng_series is not None and not ng_series.empty:
            risk_components.append(_normalize_series(ng_series))
        std_series = context.get("plant_proc_std_mean")
        if std_series is not None and not std_series.empty:
            risk_components.append(_normalize_series(std_series))
        outlier_map = context.get("plant_outlier_rate_map")
        if outlier_map is not None:
            rates = outlier_map["rates"]
            if rates is not None and not rates.empty:
                risk_components.append(_normalize_series(rates))
        if risk_components:
            risk_df = pd.concat(risk_components, axis=1).fillna(0)
            risk_score = risk_df.mean(axis=1)
            thresholds = risk_score.quantile([0.33, 0.66]).tolist()
            context["plant_risk"] = {
                "score": risk_score.to_dict(),
                "default_score": float(risk_score.mean()),
                "quality_thresholds": thresholds,
            }
    if ENABLE_PILOT_PARAM_ZSCORE and "Mass_Pilot" in train_df.columns:
        pilot_stats = _compute_pilot_param_stats(
            train_df,
            "Mass_Pilot",
            proc_cols,
            PILOT_ZSCORE_TOPK,
        )
        if pilot_stats is not None:
            context["pilot_param_stats"] = pilot_stats

    width_edges = context.get("width_edges")
    if width_edges is not None and "Plant" in train_df.columns:
        width_bins = _assign_bins(train_df["Width"], width_edges)
        cross = train_df.assign(_width_bin=width_bins)
        context["plant_width_bin_rate"] = (
            cross.groupby(["Plant", "_width_bin"])[config.target_column]
            .apply(lambda s: (s != "Good").mean())
            if config.target_column in train_df.columns
            else pd.Series(dtype=float)
        )
    return context


def _process_dataframe(
    df: pd.DataFrame,
    config: PipelineConfig,
    context: Dict[str, Dict],
    is_train: bool,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if config.id_column in df.columns:
        out[config.id_column] = df[config.id_column]
    if is_train:
        out[config.target_column] = df[config.target_column]

    base_columns = ["Plant", "Mass_Pilot", "Width", "Aspect", "Inch"]
    for col in base_columns:
        if col in df.columns:
            out[col] = df[col]

    # Size bins & interactions
    size_risk_score = pd.Series(0, index=out.index, dtype="float32") if ENABLE_SIZE_RISK else None

    for col in ("Width", "Aspect", "Inch"):
        edges = context.get(f"{col.lower()}_edges")
        if edges is not None and col in df.columns:
            out[f"{col}_bin"] = _assign_bins(df[col], edges)

        if ENABLE_SIZE_RISK:
            risk_info = context.get("size_risk_info", {}).get(col)
            bin_col = f"{col}_bin"
            if risk_info and bin_col in out.columns:
                risk_bins = set(risk_info.get("risk_bins", []))
                flag = (
                    out[bin_col]
                    .fillna(-1)
                    .astype(int)
                    .isin(risk_bins)
                    .astype("int8")
                )
                out[f"{col}_risk_flag"] = flag
                size_risk_score += flag

    if {"Plant", "Width"}.issubset(df.columns) and "Width_bin" in out.columns:
        combo = df["Plant"].astype(str) + "_" + out["Width_bin"].astype(str)
        out["plant_width_bin"] = combo.map(hash)
    if {"Mass_Pilot", "Width"}.issubset(df.columns) and "Width_bin" in out.columns:
        combo = df["Mass_Pilot"].astype(str) + "_" + out["Width_bin"].astype(str)
        out["mass_width_bin"] = combo.map(hash)
        if ENABLE_MASS_SIZE_INTERACTION:
            out["width_bin_x_pilot"] = pd.Categorical(combo).codes.astype("int32")
    if ENABLE_MASS_SIZE_INTERACTION and {"Mass_Pilot", "Aspect"}.issubset(df.columns) and "Aspect_bin" in out.columns:
        combo = df["Mass_Pilot"].astype(str) + "_" + out["Aspect_bin"].astype(str)
        out["aspect_bin_x_pilot"] = pd.Categorical(combo).codes.astype("int32")

    proc_cols = context.get("proc_cols") or [
        col
        for col in df.columns
        if col.startswith("Proc_Param") and pd.api.types.is_numeric_dtype(df[col])
    ]
    if proc_cols:
        proc_values = df[proc_cols].to_numpy(dtype=np.float32)
        out["proc_mean"] = proc_values.mean(axis=1)
        out["proc_std"] = proc_values.std(axis=1)
        out["proc_range"] = proc_values.max(axis=1) - proc_values.min(axis=1)

    for prefix in ("x", "y", "p"):
        feats = _tensor_features(df, prefix)
        for col in feats.columns:
            out[col] = feats[col].values
        if ENABLE_SEQ_GRAD:
            cols = _prefixed_columns(df, prefix)
            if cols:
                matrix = (
                    df[cols]
                    .ffill(axis=1)
                    .bfill(axis=1)
                    .to_numpy(dtype=np.float32)
                )
                gradients = np.diff(matrix, axis=1)
                out[f"{prefix}_grad_std"] = gradients.std(axis=1)
                out[f"{prefix}_grad_max"] = gradients.max(axis=1)
                out[f"{prefix}_grad_min"] = gradients.min(axis=1)

    if {"x_range", "y_range"}.issubset(out.columns):
        out["xy_range_ratio"] = out["x_range"] / (out["y_range"] + 1e-6)

    if {"Mass_Pilot", "Plant"}.issubset(df.columns):
        combo = df["Mass_Pilot"].astype(str) + "_" + df["Plant"].astype(str)
        out["mass_plant_hash"] = combo.map(hash)

    summary_cols = [col for col in df.columns if col.startswith("G")]
    for col in summary_cols:
        out[col] = df[col]
    if {"G1", "G2", "G3", "G4"}.issubset(df.columns):
        out["G_total"] = df[["G1", "G2", "G3", "G4"]].sum(axis=1)
        out["G_balance"] = df["G1"] - df["G4"]
        out["G_ratio_12"] = df["G1"] / (df["G2"] + 1e-6)

    plant_ng_rate = context.get("plant_ng_rate")
    if plant_ng_rate is not None and "Plant" in df.columns:
        overall = context.get("overall_ng_rate", plant_ng_rate.mean())
        out["plant_ng_rate"] = df["Plant"].map(plant_ng_rate).fillna(overall)

    plant_proc_std_mean = context.get("plant_proc_std_mean")
    if plant_proc_std_mean is not None and "Plant" in df.columns:
        out["plant_proc_std_mean"] = df["Plant"].map(plant_proc_std_mean).fillna(plant_proc_std_mean.mean())

    if ENABLE_PLANT_RISK and "Plant" in df.columns:
        outlier_map = context.get("plant_outlier_rate_map")
        if outlier_map is not None:
            rates = outlier_map.get("rates")
            overall = outlier_map.get("overall", 0.0)
            if rates is not None:
                out["plant_outlier_rate"] = df["Plant"].map(rates).fillna(overall)
        risk_info = context.get("plant_risk")
        if risk_info is not None:
            score_map = risk_info.get("score", {})
            default_score = risk_info.get("default_score", 0.0)
            thresholds = risk_info.get("quality_thresholds", [0.33, 0.66])
            risk_score = df["Plant"].map(score_map).fillna(default_score)
            out["plant_risk_score"] = risk_score
            bins = [-np.inf] + thresholds + [np.inf]
            labels = [0, 1, 2]
            quality = pd.cut(risk_score, bins=bins, labels=labels)
            out["plant_quality_group"] = quality.cat.codes.astype("int64")

    plant_proc_mean = context.get("plant_proc_mean")
    if isinstance(plant_proc_mean, pd.DataFrame) and not plant_proc_mean.empty and "Plant" in df.columns:
        for col in proc_cols:
            if col in df.columns:
                mean_map = plant_proc_mean[col]
                out[f"{col}_dev_from_plant"] = df[col] - df["Plant"].map(mean_map).fillna(mean_map.mean())

    if ENABLE_PILOT_PARAM_ZSCORE and "Mass_Pilot" in df.columns:
        pilot_stats = context.get("pilot_param_stats", {})
        pilot_series = df["Mass_Pilot"].astype(str)
        for col, stats in pilot_stats.items():
            if col not in df.columns:
                continue
            means = stats.get("mean", {})
            stds = stats.get("std", {})
            global_mean = stats.get("global_mean", 0.0)
            global_std = stats.get("global_std", 1.0)
            mean_series = pilot_series.map(means).fillna(global_mean)
            std_series = pilot_series.map(stds).fillna(global_std)
            std_series = std_series.replace(0, global_std)
            z = (df[col] - mean_series) / (std_series + 1e-6)
            out[f"{col}_pilot_zscore"] = z.astype(np.float32)

    latent_frames = context.get("latent")
    if latent_frames is not None:
        latent_df = latent_frames["train" if is_train else "test"]
        for col in latent_df.columns:
            out[col] = latent_df.loc[df.index, col].values

    if ENABLE_SIZE_RISK and size_risk_score is not None and size_risk_score.any():
        out["Size_Risk_Score"] = size_risk_score.astype(int)

    return out


def run_feature_generation(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    with log_time(logger, "데이터 로딩"):
        train_df, test_df = load_train_test(config)

    context = _build_feature_context(train_df, config)

    latent_dir = config.output_dir / "features" / "latent"
    latent_train_path = latent_dir / "train_latent.npy"
    latent_test_path = latent_dir / "test_latent.npy"
    if latent_train_path.exists() and latent_test_path.exists():
        train_latent = np.load(latent_train_path)
        test_latent = np.load(latent_test_path)
        if len(train_latent) == len(train_df) and len(test_latent) == len(test_df):
            latent_cols = [f"latent_{i}" for i in range(train_latent.shape[1])]
            context["latent"] = {
                "train": pd.DataFrame(train_latent, columns=latent_cols, index=train_df.index),
                "test": pd.DataFrame(test_latent, columns=latent_cols, index=test_df.index),
            }
            logger.info("Latent feature 사용: %s", latent_dir)
        else:
            logger.warning("Latent feature 크기가 데이터와 맞지 않아 무시합니다.")

    with log_time(logger, "학습 피처 생성"):
        train_features = _process_dataframe(train_df, config, context, is_train=True)
        train_path = config.output_dir / "features" / "train_features.pkl"
        train_features.to_pickle(train_path)
        logger.info("학습 피처 저장: %s (%d cols)", train_path, len(train_features.columns))

    with log_time(logger, "테스트 피처 생성"):
        test_features = _process_dataframe(test_df, config, context, is_train=False)
        test_path = config.output_dir / "features" / "test_features.pkl"
        test_features.to_pickle(test_path)
        logger.info("테스트 피처 저장: %s (%d cols)", test_path, len(test_features.columns))


if __name__ == "__main__":
    run_feature_generation()

