"""4단계: 모델 학습."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


def _load_processed(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    processed_dir = config.output_dir / "processed"
    X = load_numpy(processed_dir / "train_X.npy")
    y = load_numpy(processed_dir / "train_y.npy")
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    return X, y, metadata


def _train_fold(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X[train_idx], y[train_idx])
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"logreg_fold{fold}.joblib"
    joblib.dump(model, model_path)
    logger.info("모델 저장: %s", model_path)
    fold_probs = model.predict_proba(X[valid_idx])[:, 1]
    return fold_probs


NUMPY_MODEL_TYPES = {"linear", "logistic", "mlp", "svm", "autoencoder"}


def run_model_training(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    if config.model_type not in NUMPY_MODEL_TYPES:
        raise ValueError(
            f"현재 model_training.py는 NumPy 기반 모델만 지원합니다. "
            f"model_type='{config.model_type}'에 맞는 학습 스크립트를 사용해주세요."
        )
    config.ensure_output_dirs()
    X, y, metadata = _load_processed(config)
    folds = metadata["folds"]
    oof = np.zeros_like(y, dtype=np.float32)

    with log_time(logger, "교차검증 학습"):
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            fold_probs = _train_fold(fold, train_idx, valid_idx, X, y, config)
            oof[valid_idx] = fold_probs.astype(np.float32)

    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    logger.info("OOF ROC-AUC: %.4f | AP: %.4f", roc_auc, ap)

    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs.npy", oof)
    metrics_path = config.output_dir / "logs"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / "train_metrics.jsonl").write_text(
        json.dumps({"roc_auc": roc_auc, "average_precision": ap})
    )


if __name__ == "__main__":
    run_model_training()

