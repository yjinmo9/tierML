"""4단계: 트리 모델 학습 (LightGBM/XGBoost/CatBoost 비교 및 튜닝)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)

# 트리 모델 import
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


def _load_tree_data(config: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """트리 모델용 전처리된 데이터 로딩."""
    processed_dir = config.output_dir / "processed"
    train_df = pd.read_pickle(processed_dir / "train_tree_ready.pkl")
    y = load_numpy(processed_dir / "train_y.npy")
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    return train_df, y, metadata


def _get_feature_columns(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    """ID와 타겟을 제외한 피처 컬럼 리스트."""
    exclude = {config.id_column, config.target_column}
    return [col for col in df.columns if col not in exclude]


def _train_lgbm_fold(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """LightGBM fold 학습."""
    if lgb is None:
        raise ImportError("lightgbm이 설치되지 않았습니다. pip install lightgbm")
    
    feature_cols = _get_feature_columns(X, config)
    X_train, X_valid = X.loc[train_idx, feature_cols], X.loc[valid_idx, feature_cols]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    # 불균형 대응: scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=pos_weight,
        random_state=config.random_seed + fold,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lgbm_fold{fold}.joblib"
    joblib.dump(model, model_path)
    logger.info("모델 저장: %s", model_path)
    
    return model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]


def _train_xgb_fold(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """XGBoost fold 학습."""
    if xgb is None:
        raise ImportError("xgboost가 설치되지 않았습니다. pip install xgboost")
    
    feature_cols = _get_feature_columns(X, config)
    X_train, X_valid = X.loc[train_idx, feature_cols], X.loc[valid_idx, feature_cols]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=config.random_seed + fold,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"xgb_fold{fold}.joblib"
    joblib.dump(model, model_path)
    logger.info("모델 저장: %s", model_path)
    
    return model.predict_proba(X_valid)[:, 1]


def _train_catboost_fold(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """CatBoost fold 학습."""
    if cb is None:
        raise ImportError("catboost가 설치되지 않았습니다. pip install catboost")
    
    feature_cols = _get_feature_columns(X, config)
    X_train, X_valid = X.loc[train_idx, feature_cols], X.loc[valid_idx, feature_cols]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    # 카테고리 컬럼 자동 감지
    cat_features = [
        i for i, col in enumerate(feature_cols)
        if X[col].dtype in ["object", "category", "int64"] and X[col].nunique() < 100
    ]
    
    model = cb.CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bylevel=0.8,
        scale_pos_weight=pos_weight,
        random_seed=config.random_seed + fold,
        cat_features=cat_features if cat_features else None,
        verbose=False,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
    )
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"catboost_fold{fold}.joblib"
    joblib.dump(model, model_path)
    logger.info("모델 저장: %s", model_path)
    
    return model.predict_proba(X_valid)[:, 1]


TREE_MODEL_FACTORY = {
    "lgbm": _train_lgbm_fold,
    "xgb": _train_xgb_fold,
    "catboost": _train_catboost_fold,
}


def run_tree_model_training(
    config: PipelineConfig | None = None,
    model_name: str = "lgbm",
) -> dict[str, float]:
    """트리 모델 학습 실행."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    if model_name not in TREE_MODEL_FACTORY:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 선택: {list(TREE_MODEL_FACTORY.keys())}")
    
    train_func = TREE_MODEL_FACTORY[model_name]
    train_df, y, metadata = _load_tree_data(config)
    folds = metadata["folds"]
    oof = np.zeros_like(y, dtype=np.float32)
    
    with log_time(logger, f"{model_name.upper()} 교차검증 학습"):
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            fold_probs = train_func(fold, train_idx, valid_idx, train_df, y, config)
            oof[valid_idx] = fold_probs.astype(np.float32)
    
    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    logger.info("OOF ROC-AUC: %.4f | AP: %.4f", roc_auc, ap)
    
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / f"oof_probs_{model_name}.npy", oof)
    
    metrics = {"roc_auc": roc_auc, "average_precision": ap, "model": model_name}
    metrics_path = config.output_dir / "logs"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / f"train_metrics_{model_name}.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False)
    )
    
    return metrics


def compare_tree_models(config: PipelineConfig | None = None) -> dict[str, dict[str, float]]:
    """모든 트리 모델 비교."""
    config = config or PipelineConfig()
    results = {}
    
    for model_name in TREE_MODEL_FACTORY.keys():
        try:
            logger.info("=" * 50)
            logger.info("모델 비교: %s", model_name.upper())
            logger.info("=" * 50)
            metrics = run_tree_model_training(config, model_name=model_name)
            results[model_name] = metrics
        except Exception as e:
            logger.error("%s 학습 실패: %s", model_name, e)
            results[model_name] = {"error": str(e)}
    
    logger.info("=" * 50)
    logger.info("모델 비교 결과 요약")
    logger.info("=" * 50)
    for name, metrics in results.items():
        if "error" not in metrics:
            logger.info("%s: ROC-AUC=%.4f, AP=%.4f", name, metrics["roc_auc"], metrics["average_precision"])
        else:
            logger.info("%s: 실패 (%s)", name, metrics["error"])
    
    return results


def run_tree_inference(
    config: PipelineConfig | None = None,
    model_name: str = "catboost",
    use_tuned: bool = True,
) -> None:
    """트리 모델 추론."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    processed_dir = config.output_dir / "processed"
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # 테스트 데이터 로딩
    test_df = pd.read_pickle(processed_dir / "test_tree_ready.pkl")
    feature_cols = _get_feature_columns(test_df, config)
    
    # 모델 로딩
    model_dir = config.output_dir / "models"
    if use_tuned:
        model_pattern = f"{model_name}_tuned_fold*.joblib"
    else:
        model_pattern = f"{model_name}_fold*.joblib"
    
    model_paths = sorted(model_dir.glob(model_pattern))
    if not model_paths:
        raise FileNotFoundError(f"모델이 존재하지 않습니다: {model_pattern}")
    
    logger.info("%s 모델 %d개 로딩", model_name.upper(), len(model_paths))
    
    all_probs = []
    with log_time(logger, f"{model_name.upper()} 추론"):
        for path in model_paths:
            model = joblib.load(path)
            probs = model.predict_proba(test_df[feature_cols])[:, 1]
            all_probs.append(probs)
            logger.info("추론 완료: %s", path.name)
    
    ensemble_probs = np.mean(np.vstack(all_probs), axis=0).astype(np.float32)
    save_numpy(predictions_dir / "raw_probs.npy", ensemble_probs)
    logger.info("평균 확률 저장: %s (shape: %s)", predictions_dir / "raw_probs.npy", ensemble_probs.shape)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "catboost", "all"], default="all")
    args = parser.parse_args()
    
    if args.model == "all":
        compare_tree_models()
    else:
        run_tree_model_training(model_name=args.model)

