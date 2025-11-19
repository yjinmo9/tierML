"""7단계: 앙상블 및 제출 파일 생성."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_utils import load_numpy
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def _load_prediction_files(paths: List[Path]) -> np.ndarray:
    arrays = [load_numpy(path) for path in paths]
    stacked = np.vstack(arrays)
    return stacked.mean(axis=0)


def _determine_prob_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "prob" in col.lower():
            return col
    df["probability"] = 0.0
    return "probability"


def _decision(prob: np.ndarray, threshold: float) -> np.ndarray:
    return prob >= threshold


def _base_id(identifier: str) -> str:
    parts = identifier.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in {"L", "P"}:
        return parts[0]
    return identifier


def _build_prob_map(config: PipelineConfig, probs: np.ndarray) -> Dict[str, float]:
    test_df = pd.read_csv(config.test_path)
    if config.id_column not in test_df.columns:
        raise KeyError(f"테스트 CSV에서 ID 컬럼({config.id_column})을 찾을 수 없습니다.")
    test_ids = test_df[config.id_column].astype(str).tolist()
    if len(test_ids) != len(probs):
        raise ValueError(
            f"예측 길이({len(probs)})와 테스트 ID 수({len(test_ids)})가 다릅니다. "
            "전처리/추론 과정을 확인하세요."
        )
    return dict(zip(test_ids, probs))


def _map_sample_probabilities(sample: pd.DataFrame, id_col: str, prob_map: Dict[str, float]) -> np.ndarray:
    mapped = []
    for identifier in sample[id_col].astype(str):
        base = _base_id(identifier)
        if base not in prob_map:
            raise KeyError(f"샘플 ID '{identifier}'에 대응하는 테스트 ID '{base}'를 찾을 수 없습니다.")
        mapped.append(prob_map[base])
    return np.array(mapped, dtype=np.float32)


def run_ensemble_and_submission(
    config: PipelineConfig | None = None, extra_prediction_paths: List[str] | None = None
) -> Path:
    config = config or PipelineConfig()
    predictions_dir = config.output_dir / "predictions"
    default_path = predictions_dir / "calibrated_probs.npy"
    if not default_path.exists():
        logger.warning("보정 확률이 없어 raw_probs를 사용합니다.")
        default_path = predictions_dir / "raw_probs.npy"

    paths = [default_path]
    if extra_prediction_paths:
        paths.extend(Path(p) for p in extra_prediction_paths)

    probs = _load_prediction_files(paths)
    prob_map = _build_prob_map(config, probs)

    sample = pd.read_csv(config.sample_submission_path)
    id_col = next((col for col in sample.columns if col.lower().startswith("id")), None)
    if id_col is None:
        raise KeyError("샘플 제출 파일에서 ID 컬럼을 찾을 수 없습니다.")

    prob_col = _determine_prob_column(sample)
    prob_values = _map_sample_probabilities(sample, id_col, prob_map)
    sample[prob_col] = prob_values

    decision_col = "decision" if "decision" in sample.columns else "Decision"
    sample[decision_col] = _decision(prob_values, config.decision_threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.output_dir / "submissions" / f"submission_{timestamp}.csv"
    sample.to_csv(submission_path, index=False)
    logger.info("제출 파일 저장: %s", submission_path)
    return submission_path


if __name__ == "__main__":
    run_ensemble_and_submission()

