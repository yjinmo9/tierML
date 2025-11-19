"""고차원 시퀀스를 Dense AutoEncoder로 요약."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - 환경별 의존성 안내
    raise ImportError(
        "AutoEncoder 기능을 사용하려면 PyTorch가 필요합니다. "
        "pip install torch==2.* 또는 conda install pytorch 등으로 설치해주세요."
    ) from exc

from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .data_utils import load_train_test
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


def _sequence_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return sorted(
        [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()],
        key=lambda col: int(col[len(prefix) :]),
    )


def _stack_sequences(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> Tuple[np.ndarray, List[str]]:
    matrices = []
    all_columns: List[str] = []
    for prefix in prefixes:
        cols = _sequence_columns(df, prefix)
        if not cols:
            continue
        all_columns.extend(cols)
        matrices.append(df[cols].fillna(0.0).to_numpy(dtype=np.float32))
    if not matrices:
        raise ValueError("시퀀스 컬럼(x/y/p)이 존재하지 않습니다.")
    stacked = np.hstack(matrices)
    return stacked, all_columns


class SequenceDataset(Dataset):
    def __init__(self, array: np.ndarray):
        self.tensor = torch.from_numpy(array).float()

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


class DenseAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass
class AutoEncoderConfig:
    latent_dim: int = 16
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


def _train_model(model: DenseAutoEncoder, loader: DataLoader, epochs: int, lr: float, weight_decay: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        avg_loss = running_loss / len(loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("AE Epoch %d/%d | loss=%.6f", epoch + 1, epochs, avg_loss)


def _encode_latent(model: DenseAutoEncoder, array: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = SequenceDataset(array)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            latent = model.encode_latent(batch)
            latents.append(latent.cpu().numpy())
    return np.vstack(latents)


def train_autoencoder(config: PipelineConfig | None = None, ae_config: AutoEncoderConfig | None = None) -> None:
    config = config or PipelineConfig()
    ae_config = ae_config or AutoEncoderConfig()
    config.ensure_output_dirs()

    with log_time(logger, "AutoEncoder 학습 전체"):
        train_df, test_df = load_train_test(config)
        train_sequences, used_columns = _stack_sequences(train_df, ("x", "y", "p"))
        test_sequences, _ = _stack_sequences(test_df, ("x", "y", "p"))

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_sequences)
        test_scaled = scaler.transform(test_sequences)

        dataset = SequenceDataset(train_scaled)
        loader = DataLoader(dataset, batch_size=ae_config.batch_size, shuffle=True, drop_last=False)

        model = DenseAutoEncoder(input_dim=train_scaled.shape[1], latent_dim=ae_config.latent_dim)
        _train_model(model, loader, ae_config.epochs, ae_config.learning_rate, ae_config.weight_decay)

        train_latent = _encode_latent(model, train_scaled)
        test_latent = _encode_latent(model, test_scaled)

        latent_dir = config.output_dir / "features" / "latent"
        latent_dir.mkdir(parents=True, exist_ok=True)
        np.save(latent_dir / "train_latent.npy", train_latent.astype(np.float32))
        np.save(latent_dir / "test_latent.npy", test_latent.astype(np.float32))
        torch.save(model.state_dict(), latent_dir / "autoencoder.pt")

        scaler_meta = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "used_columns": used_columns,
            "latent_dim": ae_config.latent_dim,
            "batch_size": ae_config.batch_size,
            "epochs": ae_config.epochs,
        }
        (latent_dir / "metadata.json").write_text(json.dumps(scaler_meta, indent=2, ensure_ascii=False))
        logger.info("AE latent 저장: %s", latent_dir)


if __name__ == "__main__":
    train_autoencoder()

