"""5단계: 모델 추론."""

from __future__ import annotations

import joblib
import numpy as np

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


NUMPY_MODEL_TYPES = {"linear", "logistic", "mlp", "svm", "autoencoder"}


def run_inference(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    if config.model_type not in NUMPY_MODEL_TYPES:
        raise ValueError(
            f"현재 model_inference.py는 NumPy 기반 모델만 지원합니다. "
            f"model_type='{config.model_type}'에 맞는 추론 스크립트를 사용해주세요."
        )
    processed_dir = config.output_dir / "processed"
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    test_X = load_numpy(processed_dir / "test_X.npy")

    model_dir = config.output_dir / "models"
    model_paths = sorted(model_dir.glob("logreg_fold*.joblib"))
    if not model_paths:
        raise FileNotFoundError("모델이 존재하지 않습니다. 먼저 학습을 실행하세요.")

    all_probs = []
    with log_time(logger, "모델 추론"):
        for path in model_paths:
            model = joblib.load(path)
            probs = model.predict_proba(test_X)[:, 1]
            all_probs.append(probs)
            logger.info("추론 완료: %s", path.name)

    ensemble_probs = np.mean(np.vstack(all_probs), axis=0).astype(np.float32)
    save_numpy(predictions_dir / "raw_probs.npy", ensemble_probs)
    logger.info("평균 확률 저장: %s", predictions_dir / "raw_probs.npy")


if __name__ == "__main__":
    run_inference()

