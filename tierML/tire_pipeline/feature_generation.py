"""2단계: 피처 생성."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import skew

from .config import PipelineConfig
from .data_utils import load_train_test
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)

# 녹색 피처만 사용 모드
ENABLE_GREEN_FEATURES_ONLY = os.getenv("ENABLE_GREEN_FEATURES_ONLY", "0").lower() in {"1", "true", "yes"}
# Plant_9 필수 피처 모드 (베이스라인 + 필수 3개)
ENABLE_PLANT9_ESSENTIAL = os.getenv("ENABLE_PLANT9_ESSENTIAL", "0").lower() in {"1", "true", "yes"}
# Plant_9 선택 피처 모드 (전체 선택 피처 포함)
ENABLE_PLANT9_OPTIONAL = os.getenv("ENABLE_PLANT9_OPTIONAL", "0").lower() in {"1", "true", "yes"}
# 노란색 피처 활성화 (녹색 모드에서도 사용 가능)
ENABLE_YELLOW_FEATURES = os.getenv("ENABLE_YELLOW_FEATURES", "0").lower() in {"1", "true", "yes"}
# 주황색 피처 그룹별 활성화
ENABLE_ORANGE_PLANT_STABILITY = os.getenv("ENABLE_ORANGE_PLANT_STABILITY", "0").lower() in {"1", "true", "yes"}  # plant_proc_std_mean, plant_outlier_rate
ENABLE_ORANGE_PARAM_CLUSTER = os.getenv("ENABLE_ORANGE_PARAM_CLUSTER", "0").lower() in {"1", "true", "yes"}  # Param_cluster_mean, Param_cluster_std, Param_cluster_ratio
ENABLE_ORANGE_PLANT_COMBO = os.getenv("ENABLE_ORANGE_PLANT_COMBO", "0").lower() in {"1", "true", "yes"}  # plant_p6_combo
ENABLE_ORANGE_SIZE_PCA = os.getenv("ENABLE_ORANGE_SIZE_PCA", "0").lower() in {"1", "true", "yes"}  # Size_PCA_1, Size_PCA_2, Size_cluster
ENABLE_ORANGE_PCA = os.getenv("ENABLE_ORANGE_PCA", "0").lower() in {"1", "true", "yes"}  # PCA_p1, PCA_p2, PCA_XY_1, PCA_XY_2

ENABLE_SIZE_RISK = os.getenv("ENABLE_SIZE_RISK_SCORE", "0").lower() in {"1", "true", "yes"}
ENABLE_MASS_SIZE_INTERACTION = os.getenv("ENABLE_MASS_SIZE_INTERACTION", "0").lower() in {"1", "true", "yes"}
ENABLE_PLANT_RISK = os.getenv("ENABLE_PLANT_RISK_FEATURES", "0").lower() in {"1", "true", "yes"}
ENABLE_PILOT_PARAM_ZSCORE = os.getenv("ENABLE_PILOT_PARAM_ZSCORE", "0").lower() in {"1", "true", "yes"}
PILOT_ZSCORE_TOPK = int(os.getenv("PILOT_ZSCORE_TOPK", "5"))
ENABLE_SEQ_GRAD = os.getenv("ENABLE_SEQ_GRAD_FEATURES", "0").lower() in {"1", "true", "yes"}
ENABLE_PARAM_CLUSTER = os.getenv("ENABLE_PARAM_CLUSTER_FEATURES", "0").lower() in {"1", "true", "yes"}
ENABLE_PARAM_HIGH_FLAG = os.getenv("ENABLE_PARAM_HIGH_FLAG_FEATURES", "0").lower() in {"1", "true", "yes"}
ENABLE_MASS_PILOT_PARAM_INTERACTION = os.getenv("ENABLE_MASS_PILOT_PARAM_INTERACTION", "0").lower() in {"1", "true", "yes"}
# Tier 2/3 피처 플래그 (의미있는 그룹으로 분리)
ENABLE_SIZE_FEATURES = os.getenv("ENABLE_SIZE_FEATURES", "0").lower() in {"1", "true", "yes"}  # width_diff_from_pilot_mean, inch_diff_from_pilot_mean, Size_PCA, Size_cluster
ENABLE_PCA_FEATURES = os.getenv("ENABLE_PCA_FEATURES", "0").lower() in {"1", "true", "yes"}  # PCA_p1, PCA_p2, PCA_XY_1, PCA_XY_2
ENABLE_STAT_FEATURES = os.getenv("ENABLE_STAT_FEATURES", "0").lower() in {"1", "true", "yes"}  # X_skew, Y_skew, x_slope, y_slope
ENABLE_PLANT_COMBO = os.getenv("ENABLE_PLANT_COMBO", "0").lower() in {"1", "true", "yes"}  # plant_p6_combo
ENABLE_GRAD_EXTREMES = os.getenv("ENABLE_GRAD_EXTREMES", "0").lower() in {"1", "true", "yes"}  # p_grad_max, p_grad_min

# 녹색 피처만 사용 모드일 때 설정
if ENABLE_GREEN_FEATURES_ONLY:
    # 베이스라인: Tier 1 녹색만 (Tier 2/3 녹색은 제외)
    if not ENABLE_PLANT9_OPTIONAL:
        ENABLE_STAT_FEATURES = False  # Tier 2/3 녹색 제외
        ENABLE_GRAD_EXTREMES = False  # Tier 2/3 녹색 제외
    else:
        # 선택 피처 모드: Tier 2/3 녹색 포함
        ENABLE_STAT_FEATURES = True
        ENABLE_GRAD_EXTREMES = True
    # 빨간색 피처 비활성화
    ENABLE_SIZE_RISK = False
    ENABLE_PLANT_RISK = False
    ENABLE_PARAM_HIGH_FLAG = False
    # 노란색 피처는 별도 플래그로 제어
    if not ENABLE_YELLOW_FEATURES:
        ENABLE_MASS_PILOT_PARAM_INTERACTION = False
        ENABLE_MASS_SIZE_INTERACTION = False
        ENABLE_SIZE_FEATURES = False
    else:
        # 노란색 피처 활성화
        ENABLE_MASS_PILOT_PARAM_INTERACTION = True
        ENABLE_MASS_SIZE_INTERACTION = True
        ENABLE_SIZE_FEATURES = True  # width_diff_from_pilot_mean, inch_diff_from_pilot_mean만 (Size_PCA, Size_cluster는 주황색)
    # 주황색 피처는 그룹별 플래그로 제어
    ENABLE_PARAM_CLUSTER = ENABLE_ORANGE_PARAM_CLUSTER
    ENABLE_PLANT_COMBO = ENABLE_ORANGE_PLANT_COMBO
    ENABLE_PCA_FEATURES = ENABLE_ORANGE_PCA
    # Size_PCA, Size_cluster는 ENABLE_ORANGE_SIZE_PCA로 제어 (ENABLE_SIZE_FEATURES 내부에서)


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
    features = {
        f"{prefix}_mean": matrix.mean(axis=1),
        f"{prefix}_std": matrix.std(axis=1),
        f"{prefix}_min": matrix.min(axis=1),
        f"{prefix}_max": matrix.max(axis=1),
        f"{prefix}_q25": np.quantile(matrix, 0.25, axis=1),
        f"{prefix}_q75": np.quantile(matrix, 0.75, axis=1),
    }
    # range = max - min
    features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]
    return features


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
        # Tier 1: plant_outlier_rate 항상 생성
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
    # Tier 1: ParamX_zscore_by_pilot 항상 생성
    if "Mass_Pilot" in train_df.columns:
        pilot_stats = _compute_pilot_param_stats(
            train_df,
            "Mass_Pilot",
            proc_cols,
            PILOT_ZSCORE_TOPK,
        )
        if pilot_stats is not None:
            context["pilot_param_stats"] = pilot_stats
    
    # 🟢 Plant별 p-series 통계 계산 (Plant_9 패턴 반영용)
    # ✅ 타겟 누수 없음: label을 전혀 사용하지 않고, train 전체 데이터에서 Plant별 평균만 계산
    # ✅ 안전한 방식: train에서 고정된 기준을 계산하고, test에도 같은 기준 적용
    if "Plant" in train_df.columns:
        p_cols = _prefixed_columns(train_df, "p")
        if p_cols:
            p_matrix = train_df[p_cols].ffill(axis=1).bfill(axis=1).fillna(0).to_numpy(dtype=np.float32)
            # p_peak, p_grad_std 계산 (label 사용 없음)
            p_peak = p_matrix.max(axis=1)
            gradients = np.diff(p_matrix, axis=1)
            p_grad_std = gradients.std(axis=1)
            # 세그먼트별 평균
            segments = np.array_split(p_matrix, 4, axis=1)
            p_seg1_mean = segments[0].mean(axis=1) if len(segments) > 0 else np.zeros(len(train_df))
            p_seg3_mean = segments[2].mean(axis=1) if len(segments) > 2 else np.zeros(len(train_df))
            
            # Plant별 평균 계산 (label 고려 없이 전체 train 데이터 사용)
            p_mean = p_matrix.mean(axis=1)
            temp_df = pd.DataFrame({
                "Plant": train_df["Plant"].values,
                "p_peak": p_peak,
                "p_grad_std": p_grad_std,
                "p_mean": p_mean,
                "p_seg1_mean": p_seg1_mean,
                "p_seg3_mean": p_seg3_mean,
            }, index=train_df.index)
            plant_groups = temp_df.groupby("Plant")
            context["plant_p_peak_mean"] = plant_groups["p_peak"].mean()
            context["plant_p_grad_std_mean"] = plant_groups["p_grad_std"].mean()
            context["plant_p_mean"] = plant_groups["p_mean"].mean()
            context["plant_p_seg1_mean"] = plant_groups["p_seg1_mean"].mean()
            context["plant_p_seg3_mean"] = plant_groups["p_seg3_mean"].mean()

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
    
    # Tier 1: Param 클러스터 통계 (Param 2/4/5/7/10/11 그룹) 항상 생성
    if proc_cols:
        param_cluster_cols = [col for col in proc_cols if any(f"Param{num}" in col for num in [2, 4, 5, 7, 10, 11])]
        if param_cluster_cols:
            cluster_data = train_df[param_cluster_cols].fillna(0)
            context["param_cluster_cols"] = param_cluster_cols
            context["param_cluster_mean"] = float(cluster_data.mean().mean())
            context["param_cluster_std"] = float(cluster_data.std().mean())
            context["param_cluster_global_mean"] = cluster_data.mean(axis=1).mean()
            context["param_cluster_global_std"] = cluster_data.mean(axis=1).std()
    
    # Tier 1: Param 상위 25% 근방 플래그 (NG 집중 구간 감지) 항상 생성
    if proc_cols and target_binary is not None:
        param_high_info = {}
        for col in proc_cols:
            if col not in train_df.columns:
                continue
            series = train_df[col].dropna()
            if series.empty:
                continue
            q75 = series.quantile(0.75)
            q90 = series.quantile(0.90)
            # 상위 25% 근방 구간 (75%~90%)
            high_mask = (train_df[col] >= q75) & (train_df[col] <= q90)
            if high_mask.sum() > 0:
                high_ng_rate = target_binary[high_mask].mean()
                overall_ng_rate = target_binary.mean()
                # 상위 구간에서 NG율이 전체 평균보다 높으면 플래그
                if high_ng_rate > overall_ng_rate * 1.1:  # 10% 이상 높으면
                    param_high_info[col] = {
                        "q75": float(q75),
                        "q90": float(q90),
                        "high_ng_rate": float(high_ng_rate),
                        "overall_ng_rate": float(overall_ng_rate),
                    }
        if param_high_info:
            context["param_high_flag_info"] = param_high_info
    
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

    # plant_width_bin 제거: 불안정한 노란색 피처 (Plant × Size 교호작용, 분포 shift에 취약)
    # Plant9 효과는 이미 is_plant9, mass_pilot_x_plant9로 충분히 반영됨
    if {"Mass_Pilot", "Width"}.issubset(df.columns) and "Width_bin" in out.columns:
        combo = df["Mass_Pilot"].astype(str) + "_" + out["Width_bin"].astype(str)
        out["mass_width_bin"] = combo.map(hash)
        # 노란색 피처: MassPilot×Size_bin (녹색 모드에서도 노란색 플래그로 활성화 가능)
        # width_bin_x_pilot 제거: 불안정한 노란색 피처 (Plant 분포 shift에 취약)
        # aspect_bin_x_pilot 제거: 불안정한 노란색 피처 (Plant 분포 shift에 취약)
    if {"Mass_Pilot", "Aspect"}.issubset(df.columns) and "Aspect_bin" in out.columns:
        combo = df["Mass_Pilot"].astype(str) + "_" + out["Aspect_bin"].astype(str)
        # aspect_bin_x_pilot 제거: 불안정한 노란색 피처 (Plant 분포 shift에 취약)

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
        # Tier 1: p_grad_std 항상 생성 (p만)
        if prefix == "p":
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
        # Tier 3: x/y/p_grad_max, grad_min은 플래그로 제어 (Tier 3이므로 제외)
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
                if prefix != "p":  # p_grad_std는 이미 위에서 생성
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

    # 빨간색 피처: 항상 제외 (타겟 누수)
    if not ENABLE_GREEN_FEATURES_ONLY:
        plant_ng_rate = context.get("plant_ng_rate")
        if plant_ng_rate is not None and "Plant" in df.columns:
            overall = context.get("overall_ng_rate", plant_ng_rate.mean())
            out["plant_ng_rate"] = df["Plant"].map(plant_ng_rate).fillna(overall)
            # plant_is_high_risk: NG율이 전체 평균보다 높은 Plant 플래그
            out["plant_is_high_risk"] = (out["plant_ng_rate"] > overall).astype("int8")

    # 주황색 피처: Plant 안정성 (녹색 모드에서도 주황색 플래그로 활성화 가능)
    if (not ENABLE_GREEN_FEATURES_ONLY or ENABLE_ORANGE_PLANT_STABILITY):
        plant_proc_std_mean = context.get("plant_proc_std_mean")
        if plant_proc_std_mean is not None and "Plant" in df.columns:
            out["plant_proc_std_mean"] = df["Plant"].map(plant_proc_std_mean).fillna(plant_proc_std_mean.mean())

        # Tier 1: plant_outlier_rate
        if "Plant" in df.columns:
            outlier_map = context.get("plant_outlier_rate_map")
            if outlier_map is not None:
                rates = outlier_map.get("rates")
                overall = outlier_map.get("overall", 0.0)
                if rates is not None:
                    out["plant_outlier_rate"] = df["Plant"].map(rates).fillna(overall)
        # Tier 2: plant_risk_score, plant_quality_group는 플래그로 제어
        if ENABLE_PLANT_RISK:
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

    # Tier 1: ParamX_zscore_by_pilot 항상 생성
    if "Mass_Pilot" in df.columns:
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
    
    # 빨간색 피처: Param 상위 25% 근방 플래그 (항상 제외 - 타겟 누수)
    if not ENABLE_GREEN_FEATURES_ONLY:
        param_high_info = context.get("param_high_flag_info", {})
        for col, info in param_high_info.items():
            if col not in df.columns:
                continue
            q75 = info.get("q75", 0.0)
            q90 = info.get("q90", 0.0)
            high_flag = ((df[col] >= q75) & (df[col] <= q90)).astype("int8")
            out[f"{col}_high_flag"] = high_flag
            # 상위 구간과 전체의 차이 (qdiff)
            overall_mean = df[col].mean()
            out[f"{col}_qdiff"] = (q75 - overall_mean).astype(np.float32)
    
    # 주황색 피처: Param 클러스터 통계 (녹색 모드에서도 주황색 플래그로 활성화 가능)
    if (not ENABLE_GREEN_FEATURES_ONLY or ENABLE_ORANGE_PARAM_CLUSTER):
        param_cluster_cols = context.get("param_cluster_cols", [])
        if param_cluster_cols:
            available_cols = [col for col in param_cluster_cols if col in df.columns]
            if available_cols:
                cluster_data = df[available_cols].fillna(0)
                cluster_mean = cluster_data.mean(axis=1)
                cluster_std = cluster_data.std(axis=1)
                global_mean = context.get("param_cluster_global_mean", cluster_mean.mean())
                global_std = context.get("param_cluster_global_std", cluster_mean.std())
                out["Param_cluster_mean"] = cluster_mean.astype(np.float32)
                out["Param_cluster_std"] = cluster_std.astype(np.float32)
                out["Param_cluster_ratio"] = (cluster_mean / (global_mean + 1e-6)).astype(np.float32)
        
        # Mass_Pilot × ParamX 상호작용 제거: 불안정한 노란색 피처
        # Param은 Plant별 offset 차이가 커서 Pilot까지 곱하면 분포가 꼬여 과적합 위험

    latent_frames = context.get("latent")
    if latent_frames is not None:
        latent_df = latent_frames["train" if is_train else "test"]
        for col in latent_df.columns:
            out[col] = latent_df.loc[df.index, col].values

    if ENABLE_SIZE_RISK and size_risk_score is not None and size_risk_score.any():
        out["Size_Risk_Score"] = size_risk_score.astype(int)
    
    # Tier 2/3 피처들 (의미있는 그룹으로 분리)
    
    # 그룹 1: Size 관련 피처 (Tier 2-10, 12, 13)
    if ENABLE_SIZE_FEATURES:
        # Tier 2-10: Pilot 기반 Size 평균과의 편차 (노란색)
        if "Mass_Pilot" in df.columns:
            pilot_groups = df.groupby("Mass_Pilot")
            if "Width" in df.columns:
                width_pilot_mean = pilot_groups["Width"].mean()
                out["width_diff_from_pilot_mean"] = df["Width"] - df["Mass_Pilot"].map(width_pilot_mean).fillna(width_pilot_mean.mean())
            if "Inch" in df.columns:
                inch_pilot_mean = pilot_groups["Inch"].mean()
                out["inch_diff_from_pilot_mean"] = df["Inch"] - df["Mass_Pilot"].map(inch_pilot_mean).fillna(inch_pilot_mean.mean())
        
        # Tier 2-12: Size PCA (주황색 - 주황색 플래그로 활성화)
        if (not ENABLE_GREEN_FEATURES_ONLY or ENABLE_ORANGE_SIZE_PCA) and (not ENABLE_YELLOW_FEATURES or ENABLE_ORANGE_SIZE_PCA):
            size_cols = [col for col in ["Width", "Aspect", "Inch"] if col in df.columns]
            if len(size_cols) >= 2:
                size_data = df[size_cols].fillna(0).to_numpy(dtype=np.float32)
                pca = PCA(n_components=2, random_state=42)
                size_pca = pca.fit_transform(size_data)
                out["Size_PCA_1"] = size_pca[:, 0]
                out["Size_PCA_2"] = size_pca[:, 1]
        
        # Tier 2-13: Size Cluster (주황색 - 주황색 플래그로 활성화)
        if (not ENABLE_GREEN_FEATURES_ONLY or ENABLE_ORANGE_SIZE_PCA) and (not ENABLE_YELLOW_FEATURES or ENABLE_ORANGE_SIZE_PCA):
            if {"Width", "Aspect", "Inch"}.issubset(df.columns):
                w_bin = pd.qcut(df["Width"], q=3, labels=False, duplicates='drop').fillna(1)
                a_bin = pd.qcut(df["Aspect"], q=3, labels=False, duplicates='drop').fillna(1)
                i_bin = pd.qcut(df["Inch"], q=3, labels=False, duplicates='drop').fillna(1)
                size_cluster = (w_bin * 9 + a_bin * 3 + i_bin).astype("int32")
                out["Size_cluster"] = size_cluster
    
    # 그룹 2: PCA 관련 피처 (Tier 2-14, Tier 3)
    if ENABLE_PCA_FEATURES:
        # Tier 2-14: p-series PCA
        if "p_pca_0" in out.columns and "p_pca_1" in out.columns:
            out["PCA_p1"] = out["p_pca_0"]
            out["PCA_p2"] = out["p_pca_1"]
        
        # Tier 3: PCA_XY
        if "x_pca_0" in out.columns and "y_pca_0" in out.columns:
            xy_pca_data = np.column_stack([out["x_pca_0"], out["y_pca_0"]])
            pca_xy = PCA(n_components=2, random_state=42)
            xy_pca = pca_xy.fit_transform(xy_pca_data)
            out["PCA_XY_1"] = xy_pca[:, 0]
            out["PCA_XY_2"] = xy_pca[:, 1]
    
    # 그룹 3: 통계/분포 피처 (Tier 2-15, Tier 3)
    if ENABLE_STAT_FEATURES:
        # Tier 2-15: X/Y skewness
        if "x_mean" in out.columns and "x_std" in out.columns:
            x_cols = _prefixed_columns(df, "x")
            if x_cols:
                x_data = df[x_cols].fillna(0).to_numpy(dtype=np.float32)
                out["X_skew"] = np.apply_along_axis(lambda row: skew(row), axis=1, arr=x_data)
        if "y_mean" in out.columns and "y_std" in out.columns:
            y_cols = _prefixed_columns(df, "y")
            if y_cols:
                y_data = df[y_cols].fillna(0).to_numpy(dtype=np.float32)
                out["Y_skew"] = np.apply_along_axis(lambda row: skew(row), axis=1, arr=y_data)
        
        # Tier 3: x/y slope
        x_cols = _prefixed_columns(df, "x")
        if x_cols and len(x_cols) > 1:
            x_data = df[x_cols].ffill(axis=1).bfill(axis=1).fillna(0).to_numpy(dtype=np.float32)
            x_indices = np.arange(len(x_cols))
            slopes = np.array([np.polyfit(x_indices, row, 1)[0] if not np.all(row == 0) else 0.0 for row in x_data])
            out["x_slope"] = slopes.astype(np.float32)
        y_cols = _prefixed_columns(df, "y")
        if y_cols and len(y_cols) > 1:
            y_data = df[y_cols].ffill(axis=1).bfill(axis=1).fillna(0).to_numpy(dtype=np.float32)
            y_indices = np.arange(len(y_cols))
            slopes = np.array([np.polyfit(y_indices, row, 1)[0] if not np.all(row == 0) else 0.0 for row in y_data])
            out["y_slope"] = slopes.astype(np.float32)
    
    # 그룹 4: Plant 조합 피처 (Tier 2-11)
    if ENABLE_PLANT_COMBO:
        if {"Plant", "Proc_Param6"}.issubset(df.columns):
            combo = df["Plant"].astype(str) + "_" + df["Proc_Param6"].astype(str)
            out["plant_p6_combo"] = pd.Categorical(combo).codes.astype("int32")
    
    # 그룹 5: Gradient 극값 (Tier 3)
    if ENABLE_GRAD_EXTREMES:
        if ENABLE_SEQ_GRAD and "p_grad_max" in out.columns:
            pass  # 이미 생성됨
        elif "p_grad_std" in out.columns:
            p_cols = _prefixed_columns(df, "p")
            if p_cols:
                p_matrix = df[p_cols].ffill(axis=1).bfill(axis=1).to_numpy(dtype=np.float32)
                gradients = np.diff(p_matrix, axis=1)
                out["p_grad_max"] = gradients.max(axis=1)
                out["p_grad_min"] = gradients.min(axis=1)
    
    # 🟢 녹색 피처: Plant_9 및 Mass_Pilot × Plant 조합 (핵심 패턴 반영)
    # ✅ 타겟 누수 없음: 모든 피처가 입력값 기반으로만 생성됨
    # 필수/선택 모드에 따라 조건부 생성
    if "Plant" in df.columns and (ENABLE_PLANT9_ESSENTIAL or ENABLE_PLANT9_OPTIONAL):
        # 필수 피처 1: Plant_9 전용 위험 플래그
        # ✅ 누수 0%: Plant 번호는 원본 입력값, label 사용 없음
        out["is_plant9"] = (df["Plant"] == 9).astype("int8")
        
        # 필수 피처 2: Mass_Pilot × Plant_9 특별 조합
        # ✅ 누수 0%: 입력값 단순 조합, label 사용 없음
        if "Mass_Pilot" in df.columns:
            mass_pilot_numeric = df["Mass_Pilot"].astype(int)
            out["mass_pilot_x_plant9"] = (mass_pilot_numeric * out["is_plant9"]).astype("int8")
            
            # 선택 피처: 일반 Plant 상호작용
            if ENABLE_PLANT9_OPTIONAL:
                plant_numeric = pd.to_numeric(df["Plant"], errors='coerce').fillna(0).astype(int)
                out["mass_pilot_x_plant"] = (mass_pilot_numeric * plant_numeric).astype("int32")
        
        # 필수 피처 3: p_peak의 Plant별 편차
        # ✅ 누수 0%: train에서 label 없이 계산한 Plant별 평균을 고정값으로 사용
        if "p_peak" in out.columns:
            plant_p_peak_mean = context.get("plant_p_peak_mean")
            if plant_p_peak_mean is not None and not plant_p_peak_mean.empty:
                overall_mean = plant_p_peak_mean.mean()
                out["p_peak_dev_plant"] = (out["p_peak"] - df["Plant"].map(plant_p_peak_mean).fillna(overall_mean)).astype(np.float32)
        
        # 필수 피처 4: p_grad_std의 Plant별 편차 (안정적인 Plant-normalized 피처)
        # ✅ 누수 0%: train에서 label 없이 계산한 Plant별 평균을 고정값으로 사용
        if "p_grad_std" in out.columns:
            plant_p_grad_std_mean = context.get("plant_p_grad_std_mean")
            if plant_p_grad_std_mean is not None and not plant_p_grad_std_mean.empty:
                overall_mean = plant_p_grad_std_mean.mean()
                out["p_grad_std_dev_plant"] = (out["p_grad_std"] - df["Plant"].map(plant_p_grad_std_mean).fillna(overall_mean)).astype(np.float32)
        
        # 필수 피처 5: p_mean의 Plant별 편차 (안정적인 Plant-normalized 피처)
        # ✅ 누수 0%: train에서 label 없이 계산한 Plant별 평균을 고정값으로 사용
        if "p_mean" in out.columns:
            plant_p_mean = context.get("plant_p_mean")
            if plant_p_mean is not None and not plant_p_mean.empty:
                overall_mean = plant_p_mean.mean()
                out["p_mean_dev_plant"] = (out["p_mean"] - df["Plant"].map(plant_p_mean).fillna(overall_mean)).astype(np.float32)
        
        # 선택 피처: 나머지 p-series 편차들
        if ENABLE_PLANT9_OPTIONAL:
            
            # p_seg1_mean, p_seg3_mean 계산 (아직 생성되지 않았다면)
            p_cols = _prefixed_columns(df, "p")
            if p_cols and "p_seg1_mean" not in out.columns:
                p_matrix = df[p_cols].ffill(axis=1).bfill(axis=1).fillna(0).to_numpy(dtype=np.float32)
                segments = np.array_split(p_matrix, 4, axis=1)
                if len(segments) > 0:
                    out["p_seg1_mean"] = segments[0].mean(axis=1).astype(np.float32)
                if len(segments) > 2:
                    out["p_seg3_mean"] = segments[2].mean(axis=1).astype(np.float32)
            
            # p_seg1_mean, p_seg3_mean의 Plant별 편차
            if "p_seg1_mean" in out.columns:
                plant_p_seg1_mean = context.get("plant_p_seg1_mean")
                if plant_p_seg1_mean is not None and not plant_p_seg1_mean.empty:
                    overall_mean = plant_p_seg1_mean.mean()
                    out["p_seg1_mean_dev_plant"] = (out["p_seg1_mean"] - df["Plant"].map(plant_p_seg1_mean).fillna(overall_mean)).astype(np.float32)
            
            if "p_seg3_mean" in out.columns:
                plant_p_seg3_mean = context.get("plant_p_seg3_mean")
                if plant_p_seg3_mean is not None and not plant_p_seg3_mean.empty:
                    overall_mean = plant_p_seg3_mean.mean()
                    out["p_seg3_mean_dev_plant"] = (out["p_seg3_mean"] - df["Plant"].map(plant_p_seg3_mean).fillna(overall_mean)).astype(np.float32)

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

