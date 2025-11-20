"""2단계: AutoEncoder latent feature 검증 실험.

4가지 피처 조합 비교:
A: 도메인·공정 피처만 (latent 제외)
B: AE latent만
C: 도메인·공정 + AE latent
D: 공정/메타만 (도메인 피처 제외)
"""

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
from .tree_hyperparameter_tuning import _get_feature_columns, _load_tree_data

logger = setup_logger(__name__)

try:
    import catboost as cb
except ImportError:
    cb = None


def _filter_features_by_type(
    df: pd.DataFrame,
    config: PipelineConfig,
    feature_type: str,  # 'A', 'B', 'C', 'D'
) -> list[str]:
    """피처 타입에 따라 컬럼 필터링."""
    all_cols = _get_feature_columns(df, config)
    
    if feature_type == "A":  # 도메인·공정 피처만 (latent 제외)
        # latent 제외, 나머지 모두 포함
        return [col for col in all_cols if not col.startswith("latent_")]
    
    elif feature_type == "B":  # AE latent만
        return [col for col in all_cols if col.startswith("latent_")]
    
    elif feature_type == "C":  # 도메인·공정 + AE latent
        return all_cols  # 모든 피처 포함
    
    elif feature_type == "D":  # 공정/메타만 (도메인 피처 제외)
        # (x, y, p) 관련 피처 제외, latent 제외
        excluded_prefixes = ["x_", "y_", "p_", "latent_"]
        return [
            col for col in all_cols
            if not any(col.startswith(prefix) for prefix in excluded_prefixes)
        ]
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def _train_catboost_with_features(
    train_df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    folds: dict,
    config: PipelineConfig,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """주어진 피처로 CatBoost 학습 및 OOF 예측."""
    if cb is None:
        raise ImportError("catboost가 설치되지 않았습니다.")
    
    oof = np.zeros_like(y, dtype=np.float32)
    
    for fold_str, idx_dict in folds.items():
        fold = int(fold_str)
        train_idx = np.array(idx_dict["train_idx"])
        valid_idx = np.array(idx_dict["valid_idx"])
        
        X_train = train_df.loc[train_idx, feature_cols]
        X_valid = train_df.loc[valid_idx, feature_cols]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        
        # 카테고리 컬럼 자동 감지
        cat_features = [
            i for i, col in enumerate(feature_cols)
            if train_df[col].dtype in ["object", "category", "int64"] and train_df[col].nunique() < 100
        ]
        
        fold_params = params.copy()
        if cat_features:
            fold_params["cat_features"] = cat_features
        
        model = cb.CatBoostClassifier(**fold_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
        )
        
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
    
    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    
    return oof, {"roc_auc": roc_auc, "average_precision": ap}


def run_feature_ablation_study(config: PipelineConfig | None = None) -> dict:
    """피처 조합별 실험 실행."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    if cb is None:
        raise ImportError("catboost가 설치되지 않았습니다.")
    
    # 튜닝된 최적 파라미터 로드
    tuned_params_path = config.output_dir / "logs" / "train_metrics_catboost_tuned.jsonl"
    if not tuned_params_path.exists():
        raise FileNotFoundError("튜닝된 파라미터를 찾을 수 없습니다. 먼저 튜닝을 실행하세요.")
    
    tuned_metrics = json.loads(tuned_params_path.read_text())
    best_params = tuned_metrics["params"]
    
    # scale_pos_weight 계산을 위해 데이터 로드
    train_df, y, metadata = _load_tree_data(config)
    pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
    
    # 파라미터 준비
    params = {
        "iterations": best_params.get("iterations", 225),
        "depth": best_params.get("depth", 10),
        "learning_rate": best_params.get("learning_rate", 0.074),
        "min_data_in_leaf": best_params.get("min_data_in_leaf", 32),
        "subsample": best_params.get("subsample", 0.63),
        "colsample_bylevel": best_params.get("colsample_bylevel", 0.93),
        "scale_pos_weight": pos_weight,
        "random_seed": config.random_seed,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    
    folds = metadata["folds"]
    results = {}
    
    logger.info("=" * 60)
    logger.info("피처 조합별 실험 시작")
    logger.info("=" * 60)
    
    for feature_type, description in [
        ("A", "도메인·공정 피처만 (latent 제외)"),
        ("B", "AE latent만"),
        ("C", "도메인·공정 + AE latent"),
        ("D", "공정/메타만 (도메인 피처 제외)"),
    ]:
        logger.info("")
        logger.info("-" * 60)
        logger.info("실험 %s: %s", feature_type, description)
        logger.info("-" * 60)
        
        feature_cols = _filter_features_by_type(train_df, config, feature_type)
        logger.info("사용 피처 수: %d", len(feature_cols))
        
        if len(feature_cols) == 0:
            logger.warning("피처가 없어 실험을 건너뜁니다.")
            results[feature_type] = {"error": "No features"}
            continue
        
        with log_time(logger, f"실험 {feature_type} 학습"):
            oof, metrics = _train_catboost_with_features(
                train_df, y, feature_cols, folds, config, params
            )
        
        results[feature_type] = {
            **metrics,
            "n_features": len(feature_cols),
            "description": description,
        }
        
        logger.info("결과: ROC-AUC=%.4f, AP=%.4f", metrics["roc_auc"], metrics["average_precision"])
        
        # OOF 저장
        predictions_dir = config.output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        save_numpy(predictions_dir / f"oof_probs_ablation_{feature_type}.npy", oof)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("피처 조합별 실험 결과 요약")
    logger.info("=" * 60)
    for feature_type, result in results.items():
        if "error" not in result:
            logger.info(
                "%s (%s): ROC-AUC=%.4f, AP=%.4f (피처 수: %d)",
                feature_type,
                result["description"],
                result["roc_auc"],
                result["average_precision"],
                result["n_features"],
            )
        else:
            logger.info("%s: 실패 (%s)", feature_type, result["error"])
    
    # 결과 저장
    results_path = config.output_dir / "logs" / "feature_ablation_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("결과 저장: %s", results_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    run_feature_ablation_study()

