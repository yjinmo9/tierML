"""4단계: Transformer 시퀀스 모델 (심화).

시퀀스 데이터 (256, 3)를 Transformer Encoder로 처리하여 임베딩 생성.
이 임베딩을 탭 피처와 결합하여 최종 분류.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

# PyTorch 환경 변수 설정 (segmentation fault 방지)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    
    # PyTorch 스레드 설정 (이미 설정되었을 수 있음)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError:
    raise ImportError("PyTorch가 필요합니다. pip install torch")

from .config import PipelineConfig
from .data_utils import load_numpy, load_train_test, save_numpy
from .logging_utils import log_time, setup_logger
from .tree_hyperparameter_tuning import _get_feature_columns, _load_tree_data
from .cnn_sequence_model import _extract_sequences

logger = setup_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder: (256, 3) → embedding_dim 차원 임베딩."""
    
    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=256)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) = (batch, 256, 3)
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model: (batch, 256, 3) → (batch, 256, d_model)
        x = self.input_proj(x)
        
        # Transpose for Transformer: (batch, 256, d_model) → (256, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder: (256, batch, d_model) → (256, batch, d_model)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling: (256, batch, d_model) → (batch, d_model)
        x = x.transpose(0, 1)  # (batch, 256, d_model)
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_proj(x)
        
        return x


def _train_transformer_encoder(
    train_sequences: np.ndarray,
    y_true: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    embedding_dim: int = 64,
    weight_decay: float = 0.0,
) -> TransformerEncoder:
    """Transformer 인코더 학습 (지도 학습: 타겟 레이블 사용)."""
    device = torch.device("cpu")
    
    # PyTorch 스레드 설정 (한 번만 호출 가능)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    
    # Transformer + 분류 헤드
    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TransformerEncoder(
                input_dim=3,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                embedding_dim=embedding_dim,
            )
            self.classifier = nn.Linear(embedding_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            embedding = self.encoder(x)
            logit = self.classifier(embedding)
            return self.sigmoid(logit), embedding
    
    model = TransformerClassifier()
    model.to(device)
    model.train()
    
    n_samples = len(train_sequences)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # NumPy 배열을 미리 준비
    y_array = y_true.astype(np.float32)
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        
        # 랜덤 셔플
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_indices = indices[i:end_idx]
            
            # NumPy 배열 준비
            batch_seq = train_sequences[batch_indices].copy()
            batch_y_np = y_array[batch_indices].copy()
            
            # Tensor 변환
            batch_tensor = torch.from_numpy(batch_seq).float()
            batch_y_tensor = torch.from_numpy(batch_y_np).float()
            
            try:
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(True):
                    probs, _ = model(batch_tensor)
                    loss = criterion(probs.squeeze(-1), batch_y_tensor)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                loss_val = float(loss.item())
                running_loss += loss_val
                batch_count += 1
                
            except RuntimeError as e:
                logger.warning("배치 %d-%d 런타임 오류: %s", i, end_idx, str(e))
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            except Exception as e:
                logger.warning("배치 %d-%d 오류: %s", i, end_idx, str(e))
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            finally:
                # 명시적 메모리 정리
                del batch_tensor, batch_y_tensor, batch_seq, batch_y_np
                if 'probs' in locals():
                    del probs
                if 'loss' in locals():
                    del loss
                import gc
                gc.collect()
        
        if batch_count > 0 and (epoch + 1) % 5 == 0:
            logger.info("Transformer Epoch %d/%d | loss=%.6f", epoch + 1, epochs, running_loss / batch_count)
    
    # 인코더만 반환
    return model.encoder


def _extract_transformer_embeddings(
    model: TransformerEncoder,
    sequences: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    """시퀀스에서 Transformer 임베딩 추출."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    n_samples = len(sequences)
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            try:
                end_idx = min(i + batch_size, n_samples)
                batch_seq = sequences[i:end_idx]
                batch_tensor = torch.from_numpy(batch_seq).float().to(device)
                
                embedding = model(batch_tensor)
                embeddings.append(embedding.cpu().numpy())
            except Exception as e:
                logger.warning("배치 %d-%d 처리 중 오류: %s", i, end_idx, e)
                # 오류 발생 시 더미 임베딩 추가
                dummy_emb = np.zeros((end_idx - i, 64), dtype=np.float32)
                embeddings.append(dummy_emb)
    
    return np.vstack(embeddings) if embeddings else np.zeros((n_samples, 64), dtype=np.float32)


def run_transformer_sequence_experiment(config: PipelineConfig | None = None) -> dict:
    """Transformer 시퀀스 실험 실행."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    logger.info("=" * 60)
    logger.info("Transformer 시퀀스 실험 시작")
    logger.info("=" * 60)
    
    # 1. 데이터 로딩
    with log_time(logger, "데이터 로딩"):
        train_df, test_df = load_train_test(config)
        y_true = load_numpy(config.output_dir / "processed" / "train_y.npy")
    
    # 2. 시퀀스 데이터 추출
    with log_time(logger, "시퀀스 데이터 추출"):
        train_sequences = _extract_sequences(train_df)
        test_sequences = _extract_sequences(test_df)
        logger.info("시퀀스 shape: train=%s, test=%s", train_sequences.shape, test_sequences.shape)
    
    # 3. Transformer 인코더 학습 (지도 학습)
    with log_time(logger, "Transformer 인코더 학습"):
        transformer_model = _train_transformer_encoder(
            train_sequences,
            y_true,
            epochs=20,
            batch_size=8,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            embedding_dim=64,
        )
    
    # 4. Transformer 임베딩 추출
    with log_time(logger, "Transformer 임베딩 추출"):
        train_embeddings = _extract_transformer_embeddings(transformer_model, train_sequences)
        test_embeddings = _extract_transformer_embeddings(transformer_model, test_sequences)
        logger.info("임베딩 shape: train=%s, test=%s", train_embeddings.shape, test_embeddings.shape)
    
    # 5. 탭 피처 로드 (도메인·공정 피처만, latent 제외)
    with log_time(logger, "탭 피처 로드"):
        train_tabular, y, metadata = _load_tree_data(config)
        feature_cols = _get_feature_columns(train_tabular, config)
        
        # latent 제외 (A 타입)
        tabular_cols = [col for col in feature_cols if not col.startswith("latent_")]
        logger.info("탭 피처 수: %d (latent 제외)", len(tabular_cols))
    
    # 6. Transformer 임베딩 + 탭 피처 결합
    with log_time(logger, "피처 결합"):
        train_tabular_features = train_tabular[tabular_cols].values.astype(np.float32)
        train_combined = np.hstack([train_tabular_features, train_embeddings])
        
        # 테스트 데이터도 준비
        test_tabular = pd.read_pickle(config.output_dir / "processed" / "test_tree_ready.pkl")
        test_tabular_features = test_tabular[tabular_cols].values.astype(np.float32)
        test_combined = np.hstack([test_tabular_features, test_embeddings])
        
        logger.info("결합 피처 shape: train=%s, test=%s", train_combined.shape, test_combined.shape)
    
    # 7. CatBoost로 학습 (Transformer 임베딩 포함)
    try:
        import catboost as cb
    except ImportError:
        raise ImportError("catboost가 필요합니다.")
    
    folds = metadata["folds"]
    oof = np.zeros_like(y_true, dtype=np.float32)
    
    # NumPy 배열로 변환된 경우 CatBoost는 cat_features를 인식하지 못함
    cat_features = None
    
    with log_time(logger, "CatBoost 학습 (Transformer 임베딩 포함)"):
        pos_weight = (y_true == 0).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 1.0
        
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_combined[train_idx]
            X_valid = train_combined[valid_idx]
            y_train = y_true[train_idx]
            y_valid = y_true[valid_idx]
            
            model = cb.CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.05,
                min_data_in_leaf=20,
                subsample=0.8,
                colsample_bylevel=0.8,
                scale_pos_weight=pos_weight,
                random_seed=config.random_seed + fold,
                verbose=False,
                early_stopping_rounds=50,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
            )
            
            oof[valid_idx] = model.predict_proba(X_valid, ntree_end=model.best_iteration_)[:, 1]
            
            # 모델 저장
            model_dir = config.output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"catboost_transformer_fold{fold}.joblib"
            joblib.dump(model, model_path)
    
    # 8. 성능 평가
    roc_auc = roc_auc_score(y_true, oof)
    ap = average_precision_score(y_true, oof)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Transformer + CatBoost 결과")
    logger.info("=" * 60)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("AP: %.4f", ap)
    
    # OOF 저장
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_transformer_catboost.npy", oof)
    
    # 결과 저장
    results = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "n_tabular_features": len(tabular_cols),
        "n_transformer_embeddings": train_embeddings.shape[1],
        "n_combined_features": train_combined.shape[1],
    }
    
    results_path = config.output_dir / "logs" / "transformer_sequence_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    
    # Transformer 모델 저장
    transformer_model_dir = config.output_dir / "models"
    transformer_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer_model.state_dict(), transformer_model_dir / "transformer_encoder.pt")
    logger.info("Transformer 모델 저장: %s", transformer_model_dir / "transformer_encoder.pt")
    
    return results


if __name__ == "__main__":
    from .config import PipelineConfig
    
    config = PipelineConfig()
    run_transformer_sequence_experiment(config)

