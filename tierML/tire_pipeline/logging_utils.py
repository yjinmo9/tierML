"""공통 로깅/타이머 유틸."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path


def setup_logger(name: str = "tire_pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def log_time(logger: logging.Logger, task_name: str):
    start = time.time()
    logger.info("%s 시작", task_name)
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info("%s 완료 (%.2fs)", task_name, duration)

