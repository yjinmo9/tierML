"""2단계: 피처 생성."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .data_utils import load_train_test
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


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
    for col in ("Width", "Aspect", "Inch"):
        if col in train_df.columns:
            context[f"{col.lower()}_edges"] = _compute_bin_edges(train_df[col], n_bins=5)
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

    # Size bins & interactions
    for col in ("Width", "Aspect", "Inch"):
        edges = context.get(f"{col.lower()}_edges")
        if edges is not None and col in df.columns:
            out[f"{col}_bin"] = _assign_bins(df[col], edges)
    if {"Plant", "Width"}.issubset(df.columns) and "Width_bin" in out.columns:
        combo = df["Plant"].astype(str) + "_" + out["Width_bin"].astype(str)
        out["plant_width_bin"] = combo.map(hash)
    if {"Mass_Pilot", "Width"}.issubset(df.columns) and "Width_bin" in out.columns:
        combo = df["Mass_Pilot"].astype(str) + "_" + out["Width_bin"].astype(str)
        out["mass_width_bin"] = combo.map(hash)

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

    plant_proc_mean = context.get("plant_proc_mean")
    if isinstance(plant_proc_mean, pd.DataFrame) and not plant_proc_mean.empty and "Plant" in df.columns:
        for col in proc_cols:
            if col in df.columns:
                mean_map = plant_proc_mean[col]
                out[f"{col}_dev_from_plant"] = df[col] - df["Plant"].map(mean_map).fillna(mean_map.mean())

    latent_frames = context.get("latent")
    if latent_frames is not None:
        latent_df = latent_frames["train" if is_train else "test"]
        for col in latent_df.columns:
            out[col] = latent_df.loc[df.index, col].values

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

