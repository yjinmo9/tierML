"""6단계: 확률 보정."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


def run_calibration(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    predictions_dir = config.output_dir / "predictions"
    oof_probs = load_numpy(predictions_dir / "oof_probs.npy")
    train_y = load_numpy(config.output_dir / "processed" / "train_y.npy")

    with log_time(logger, "Isotonic 보정 학습"):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_probs, train_y)
    calibrator_path = config.output_dir / "models" / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    logger.info("캘리브레이터 저장: %s", calibrator_path)

    raw_probs = load_numpy(predictions_dir / "raw_probs.npy")
    calibrated = calibrator.predict(raw_probs).astype(np.float32)
    save_numpy(predictions_dir / "calibrated_probs.npy", calibrated)
    logger.info("보정 확률 저장: %s", predictions_dir / "calibrated_probs.npy")


if __name__ == "__main__":
    run_calibration()

