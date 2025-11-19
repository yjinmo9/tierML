"""데이터 로딩/저장 유틸."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def load_train_test(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("train/test 로딩: %s, %s", config.train_path, config.test_path)
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    return train_df, test_df


def save_markdown_report(path: Path, sections: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for title, body in sections.items():
            fp.write(f"## {title}\n\n{body.strip()}\n\n")
    logger.info("리포트 저장: %s", path)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("JSON 저장: %s", path)


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    logger.info("NumPy 저장: %s", path)


def load_numpy(path: Path) -> np.ndarray:
    logger.info("NumPy 로딩: %s", path)
    return np.load(path, allow_pickle=False)

