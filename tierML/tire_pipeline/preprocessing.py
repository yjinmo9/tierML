"""3단계: 데이터 전처리.

요약:
1) CSV 원본을 직접 읽어 타입별 전처리를 수행한다.
2) 결측/이상치/시계열요약/그룹분할 로직을 한 곳에 모아 재현성을 확보한다.
3) 최종적으로 모델 입력용 NumPy(.npy)와 메타데이터를 저장한다.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import PipelineConfig
from .data_utils import load_train_test, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "Plant",
    "Mass_Pilot",
    "Size_bins",
    "Size_Cluster",
    "plant_quality_group",
    "plant_p6_combo",
]

OUTLIER_REFERENCE_COLS = [
    "Width",
    "Aspect",
    "Inch",
    *[f"Proc_Param{i}" for i in range(1, 11)],
    "G1",
    "G2",
    "G3",
    "G4",
]

VECTOR_PREFIXES = ("p", "x", "y")
TREE_MODELS = {"tree", "catboost", "lgbm", "xgb"}
NUMPY_MODELS = {"linear", "logistic", "mlp", "svm", "autoencoder"}
SCALER_FACTORY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()],
        key=lambda col: int(col[len(prefix) :]),
    )


def _fill_numeric_missing(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    medians = train[cols].median()
    train[cols] = train[cols].fillna(medians)
    test[cols] = test[cols].fillna(medians)
    return medians.to_dict()


def _fill_categorical_missing(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        train_col = train[col] if col in train.columns else pd.Series("Unknown", index=train.index)
        test_col = test[col] if col in test.columns else pd.Series("Unknown", index=test.index)
        train[col] = train_col.fillna("Unknown").astype(str)
        test[col] = test_col.fillna("Unknown").astype(str)


def _fill_sequence_missing(df: pd.DataFrame, cols: List[str]) -> None:
    if not cols:
        return

    def _row_fill(row: pd.Series) -> pd.Series:
        filled = row.interpolate(method="linear", limit_direction="both")
        return filled.fillna(filled.mean())

    df[cols] = df[cols].apply(_row_fill, axis=1)


def _add_outlier_flags(train: pd.DataFrame, test: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    flag_cols: List[str] = []
    for col in cols:
        if col not in train.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_col = f"{col}_is_outlier"
        train[flag_col] = ((train[col] < lower) | (train[col] > upper)).astype(int)
        test_series = test[col] if col in test.columns else pd.Series(index=test.index, dtype=float)
        test[flag_col] = ((test_series < lower) | (test_series > upper)).astype(int)
        flag_cols.append(flag_col)
    return flag_cols


def _plant_outlier_rate(train: pd.DataFrame, test: pd.DataFrame, plant_col: str, flag_cols: List[str]) -> None:
    if not flag_cols or plant_col not in train.columns:
        return
    row_flag = train[flag_cols].sum(axis=1) > 0
    rates = row_flag.groupby(train[plant_col]).mean()
    overall = row_flag.mean()
    train["plant_outlier_rate"] = train[plant_col].map(rates).fillna(overall)
    test["plant_outlier_rate"] = test.get(plant_col, pd.Series(index=test.index)).map(rates).fillna(overall)


def _label_encode_for_tree(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, Dict[str, int]]:
    encoders: Dict[str, Dict[str, int]] = {}
    for col in categorical_cols:
        train_series = train[col] if col in train.columns else pd.Series("Unknown", index=train.index)
        test_series = test[col] if col in test.columns else pd.Series("Unknown", index=test.index)
        combined = pd.concat([train_series, test_series], axis=0).fillna("Unknown").astype(str)
        categories = {label: idx for idx, label in enumerate(sorted(combined.unique()))}
        train[col] = train_series.fillna("Unknown").astype(str).map(categories).fillna(-1).astype(int)
        test[col] = test_series.fillna("Unknown").astype(str).map(categories).fillna(-1).astype(int)
        encoders[col] = categories
    return encoders


def _serialize_scaler(scaler, scaler_name: str) -> Dict[str, List[float] | str | None]:
    state: Dict[str, List[float] | str | None] = {"type": scaler_name}
    for attr in ("mean_", "scale_", "min_", "data_min_", "data_max_"):
        if hasattr(scaler, attr):
            value = getattr(scaler, attr)
            if isinstance(value, np.ndarray):
                state[attr] = value.tolist()
            else:
                state[attr] = value
    return state


def _one_hot_and_scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: List[str],
    scaler_name: str,
    id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[float] | str | None]]:
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    if categorical_cols:
        encoded = pd.get_dummies(combined, columns=categorical_cols, dummy_na=True)
    else:
        encoded = combined.copy()
    if id_col and id_col in encoded.columns:
        encoded = encoded.drop(columns=[id_col])
    train_encoded = encoded.iloc[: len(train)].reset_index(drop=True)
    test_encoded = encoded.iloc[len(train) :].reset_index(drop=True)
    feature_cols = train_encoded.columns.tolist()
    scaler_cls = SCALER_FACTORY.get(scaler_name, StandardScaler)
    scaler = scaler_cls()
    if feature_cols:
        scaler.fit(train_encoded[feature_cols])
        train_encoded[feature_cols] = scaler.transform(train_encoded[feature_cols])
        test_encoded[feature_cols] = scaler.transform(test_encoded[feature_cols])
        scaler_state = _serialize_scaler(scaler, scaler_name)
    else:
        scaler_state = {"type": scaler_name}
    return train_encoded, test_encoded, feature_cols, scaler_state


def _segment_stats(values: np.ndarray, prefix: str, n_segments: int = 4) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    segments = np.array_split(values, n_segments, axis=1)
    for idx, seg in enumerate(segments, start=1):
        features[f"{prefix}_seg{idx}_mean"] = seg.mean(axis=1)
        features[f"{prefix}_seg{idx}_slope"] = seg[:, -1] - seg[:, 0]
    return features


def _summarize_sequence(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = _prefixed_columns(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    matrix = df[cols].to_numpy(dtype=np.float32)
    gradients = np.gradient(matrix, axis=1)
    stats = {
        f"{prefix}_mean": matrix.mean(axis=1),
        f"{prefix}_std": matrix.std(axis=1),
        f"{prefix}_min": matrix.min(axis=1),
        f"{prefix}_max": matrix.max(axis=1),
        f"{prefix}_q25": np.quantile(matrix, 0.25, axis=1),
        f"{prefix}_q75": np.quantile(matrix, 0.75, axis=1),
        f"{prefix}_range": matrix.ptp(axis=1),
        f"{prefix}_slope": matrix[:, -1] - matrix[:, 0],
        f"{prefix}_grad_std": gradients.std(axis=1),
    }
    stats.update(_segment_stats(matrix, prefix))
    return pd.DataFrame(stats, index=df.index)


def _build_group_folds(
    target: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> Dict[int, Dict[str, List[int]]]:
    gkf = GroupKFold(n_splits=n_splits)
    folds: Dict[int, Dict[str, List[int]]] = {}
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(np.zeros_like(target), target, groups=groups)):
        folds[fold] = {
            "train_idx": train_idx.tolist(),
            "valid_idx": valid_idx.tolist(),
        }
    return folds


def _build_group_keys(df: pd.DataFrame) -> pd.Series:
    """GroupKFold용 그룹 키 생성: Plant + Mass_Pilot 조합.
    
    - Class는 타겟 누수 방지를 위해 제외
    - Width/Aspect/Inch는 continuous 변수라 그룹에 포함하지 않음
    - Plant + Mass_Pilot만 사용하여 안정적인 fold 분할
    """
    plant = df.get("Plant", pd.Series("Unknown", index=df.index)).astype(str)
    mass = df.get("Mass_Pilot", pd.Series("Unknown", index=df.index)).astype(str)
    
    return plant + "__" + mass


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def run_preprocessing(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    processed_dir = config.output_dir / "processed"

    with log_time(logger, "전처리 전체 수행"):
        # feature_generation에서 만든 피처 사용 (latent 포함)
        features_dir = config.output_dir / "features"
        train_features_path = features_dir / "train_features.pkl"
        test_features_path = features_dir / "test_features.pkl"
        
        if train_features_path.exists() and test_features_path.exists():
            logger.info("생성된 피처 파일 사용: %s", features_dir)
            train_df = pd.read_pickle(train_features_path)
            test_df = pd.read_pickle(test_features_path)
        else:
            logger.info("피처 파일이 없어 원본 데이터 사용")
            train_df, test_df = load_train_test(config)
            train_df = train_df.copy()
            test_df = test_df.copy()

        target_series = train_df[config.target_column]
        if target_series.dtype.kind in "fc":
            y_train = target_series.to_numpy(dtype=np.float32)
            target_mapping = None
        else:
            classes = sorted(target_series.dropna().unique().tolist())
            mapping = {label: idx for idx, label in enumerate(classes)}
            y_train = target_series.map(mapping).to_numpy(dtype=np.float32)
            target_mapping = mapping
        train_df = train_df.drop(columns=[config.target_column])
        id_col = config.id_column

        # STEP 1. 변수 타입 구분
        sequence_cols = {prefix: _prefixed_columns(train_df, prefix) for prefix in VECTOR_PREFIXES}
        numeric_cols = [
            col
            for col in train_df.columns
            if col not in {id_col}
            and not any(col in cols for cols in sequence_cols.values())
            and train_df[col].dtype.kind in "fc"
        ]
        auto_cats = {
            col
            for col in train_df.select_dtypes(include=["object", "category"]).columns
            if col not in {id_col}
        }
        categorical_cols = sorted(set(col for col in CATEGORICAL_COLS if col in train_df.columns) | auto_cats)

        # STEP 2. 결측치 처리
        numeric_medians = _fill_numeric_missing(train_df, test_df, numeric_cols)
        _fill_categorical_missing(train_df, test_df, categorical_cols)
        for prefix, cols in sequence_cols.items():
            _fill_sequence_missing(train_df, cols)
            _fill_sequence_missing(test_df, cols)

        # STEP 3. 이상치 플래그
        flag_cols = _add_outlier_flags(train_df, test_df, OUTLIER_REFERENCE_COLS)
        _plant_outlier_rate(train_df, test_df, "Plant", flag_cols)

        # STEP 4. 그룹 기반 분할 키
        group_keys = _build_group_keys(train_df).to_numpy()

        # STEP 5 & 7. 시계열 요약 + 구조적 통계
        for prefix in VECTOR_PREFIXES:
            train_seq_cols = _prefixed_columns(train_df, prefix)
            test_seq_cols = _prefixed_columns(test_df, prefix)
            feats = _summarize_sequence(train_df, prefix)
            if not feats.empty:
                test_feats = _summarize_sequence(test_df, prefix)
                train_df = pd.concat([train_df.drop(columns=train_seq_cols), feats], axis=1)
                test_df = pd.concat([test_df.drop(columns=test_seq_cols), test_feats], axis=1)

        vector_feature_cols = [
            col
            for col in train_df.columns
            if any(col.startswith(f"{prefix}_") for prefix in VECTOR_PREFIXES)
        ]

        # 트리 모델용 원본(카테고리 유지) 저장
        tree_ready_train = train_df.copy()
        tree_ready_test = test_df.copy()
        tree_label_encoders: Dict[str, Dict[str, int]] | None = None
        tree_train_path = processed_dir / "train_tree_ready.pkl"
        tree_test_path = processed_dir / "test_tree_ready.pkl"
        if categorical_cols:
            tree_label_encoders = _label_encode_for_tree(tree_ready_train, tree_ready_test, categorical_cols)
        tree_ready_train.to_pickle(tree_train_path)
        tree_ready_test.to_pickle(tree_test_path)

        is_tree_model = config.model_type in TREE_MODELS
        if config.model_type not in TREE_MODELS and config.model_type not in NUMPY_MODELS:
            logger.warning(
                "알 수 없는 model_type '%s'이 지정되어 기본적으로 non-tree 경로를 사용합니다.",
                config.model_type,
            )

        scaler_state: Dict[str, List[float] | str | None] | None = None
        feature_columns: List[str]
        train_matrix: np.ndarray | None = None
        test_matrix: np.ndarray | None = None

        if is_tree_model:
            train_features_df = tree_ready_train
            test_features_df = tree_ready_test
            feature_columns = [col for col in train_features_df.columns if col != id_col]
        else:
            train_features_df, test_features_df, feature_columns, scaler_state = _one_hot_and_scale(
                train_df, test_df, categorical_cols, config.scaler_type, id_col
            )
            train_matrix = train_features_df[feature_columns].to_numpy(dtype=np.float32)
            test_matrix = test_features_df[feature_columns].to_numpy(dtype=np.float32)

        folds = _build_group_folds(y_train, group_keys, config.n_splits)

        if train_matrix is not None and test_matrix is not None:
            save_numpy(processed_dir / "train_X.npy", train_matrix)
            save_numpy(processed_dir / "test_X.npy", test_matrix)
        else:
            logger.info("선택한 model_type='%s'에서는 NumPy 행렬을 생성하지 않습니다.", config.model_type)

        save_numpy(processed_dir / "train_y.npy", y_train)

        metadata = {
            "feature_columns": feature_columns,
            "numeric_medians": numeric_medians,
            "categorical_features": categorical_cols,
            "vector_feature_cols": vector_feature_cols,
            "target_mapping": target_mapping,
            "folds": folds,
            "model_type": config.model_type,
            "scaler": scaler_state,
            "tree_ready_paths": {
                "train": str(tree_train_path),
                "test": str(tree_test_path),
            },
            "tree_label_encoders": tree_label_encoders if is_tree_model else None,
            "numpy_output": train_matrix is not None,
        }
        (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.info(
            "전처리 완료: %s | 출력 형식: %s",
            processed_dir,
            "NumPy" if train_matrix is not None else "Parquet(트리 전용)",
        )


if __name__ == "__main__":
    run_preprocessing()

