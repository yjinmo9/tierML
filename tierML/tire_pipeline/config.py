"""파이프라인 공통 설정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """데이터·산출물 경로와 공통 옵션을 관리."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path("/Users/yangjinmo/tierML/5-ai-and-datascience-competition")
    output_dir: Path = Path("/Users/yangjinmo/tierML/artifacts")
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"
    target_column: str = "Class"
    id_column: str = "ID"
    random_seed: int = 2025
    n_splits: int = 5
    decision_threshold: float = 0.5
    model_type: str = "tree"  # tree | catboost | lgbm | xgb | linear | logistic | mlp | svm | autoencoder
    scaler_type: str = "standard"  # standard | minmax

    def path(self, *parts: str | Path) -> Path:
        return self.project_root.joinpath(*parts)

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / self.sample_submission_file

    def ensure_output_dirs(self) -> None:
        for sub in [
            "reports",
            "features",
            "processed",
            "models",
            "predictions",
            "submissions",
        ]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def set_data_dir(self, new_dir: str | Path) -> "PipelineConfig":
        self.data_dir = Path(new_dir)
        return self

    def override(
        self,
        *,
        output_dir: Optional[str | Path] = None,
        n_splits: Optional[int] = None,
        decision_threshold: Optional[float] = None,
        model_type: Optional[str] = None,
        scaler_type: Optional[str] = None,
    ) -> "PipelineConfig":
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        if n_splits is not None:
            self.n_splits = n_splits
        if decision_threshold is not None:
            self.decision_threshold = decision_threshold
        if model_type is not None:
            self.model_type = model_type
        if scaler_type is not None:
            self.scaler_type = scaler_type
        return self

    def set_model_type(self, model_type: str, scaler_type: Optional[str] = None) -> "PipelineConfig":
        self.model_type = model_type
        if scaler_type is not None:
            self.scaler_type = scaler_type
        return self

