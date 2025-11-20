"""4단계: 1D CNN 시퀀스 모델.

시퀀스 데이터 (256, 3)를 1D CNN으로 처리하여 64차원 임베딩 생성.
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
    from torch.utils.data import DataLoader, Dataset
    
    # PyTorch 스레드 설정
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except ImportError:
    raise ImportError("PyTorch가 필요합니다. pip install torch")

from .config import PipelineConfig
from .data_utils import load_numpy, load_train_test, save_numpy
from .logging_utils import log_time, setup_logger
from .tree_hyperparameter_tuning import _get_feature_columns, _load_tree_data

logger = setup_logger(__name__)


class SequenceDataset(Dataset):
    """시퀀스 데이터셋."""
    
    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class CNN1DEncoder(nn.Module):
    """1D CNN 인코더: (256, 3) → embedding_dim 차원 임베딩."""
    
    def __init__(
        self,
        input_channels: int = 3,
        embedding_dim: int = 64,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 입력: (batch, 256, 3) → transpose → (batch, 3, 256)
        self.conv1 = nn.Conv1d(input_channels, conv1_channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(conv2_channels, embedding_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 256, 3) → (batch, 3, 256)
        if x.dim() == 3 and x.size(-1) == 3:
            x = x.transpose(1, 2)  # (batch, 3, 256)
        # Conv1D: (batch, 3, 256) → (batch, conv1_channels, 256) → (batch, conv2_channels, 256)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Global Average Pooling: (batch, conv2_channels, 256) → (batch, conv2_channels, 1)
        x = self.pool(x)
        # Flatten: (batch, conv2_channels, 1) → (batch, conv2_channels)
        x = x.squeeze(-1)
        # Dropout
        x = self.dropout(x)
        # Dense: (batch, conv2_channels) → (batch, embedding_dim)
        x = self.fc(x)
        return x


def _extract_sequences(df: pd.DataFrame) -> np.ndarray:
    """시퀀스 데이터 추출: (n_samples, 256, 3)."""
    x_cols = sorted([col for col in df.columns if col.startswith("x") and col[1:].isdigit()], 
                    key=lambda c: int(c[1:]))
    y_cols = sorted([col for col in df.columns if col.startswith("y") and col[1:].isdigit()], 
                    key=lambda c: int(c[1:]))
    p_cols = sorted([col for col in df.columns if col.startswith("p") and col[1:].isdigit()], 
                    key=lambda c: int(c[1:]))
    
    if not (x_cols and y_cols and p_cols):
        raise ValueError("시퀀스 컬럼(x, y, p)을 찾을 수 없습니다.")
    
    n_samples = len(df)
    seq_length = len(x_cols)
    sequences = np.zeros((n_samples, seq_length, 3), dtype=np.float32)
    
    sequences[:, :, 0] = df[x_cols].fillna(0.0).values
    sequences[:, :, 1] = df[y_cols].fillna(0.0).values
    sequences[:, :, 2] = df[p_cols].fillna(0.0).values
    
    return sequences


def _train_cnn_encoder(
    train_sequences: np.ndarray,
    y_true: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64,
    conv1_channels: int = 32,
    conv2_channels: int = 64,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
) -> CNN1DEncoder:
    """CNN 인코더 학습 (지도 학습: 타겟 레이블 사용)."""
    device = torch.device("cpu")
    
    # PyTorch 스레드 설정 (한 번만 호출 가능)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass  # 이미 설정됨
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # 이미 설정됨
    
    # CNN + 분류 헤드
    class CNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = CNN1DEncoder(
                input_channels=3,
                embedding_dim=embedding_dim,
                conv1_channels=conv1_channels,
                conv2_channels=conv2_channels,
                dropout=dropout,
            )
            self.classifier = nn.Linear(embedding_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            embedding = self.encoder(x)
            logit = self.classifier(embedding)
            return self.sigmoid(logit), embedding
    
    model = CNNClassifier()
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
            
            # Tensor 변환 (메모리 공유 최소화)
            batch_tensor = torch.from_numpy(batch_seq).float()
            batch_y_tensor = torch.from_numpy(batch_y_np).float()
            
            try:
                # Gradient 초기화
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
                # 메모리 정리
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
                # Python GC 강제 실행
                import gc
                gc.collect()
        
        if batch_count > 0 and (epoch + 1) % 5 == 0:
            logger.info("CNN Epoch %d/%d | loss=%.6f", epoch + 1, epochs, running_loss / batch_count)
    
    # 인코더만 반환
    return model.encoder


def _extract_cnn_embeddings(
    model: CNN1DEncoder,
    sequences: np.ndarray,
    batch_size: int = 16,
    embedding_dim: int | None = None,
) -> np.ndarray:
    """시퀀스에서 CNN 임베딩 추출."""
    device = torch.device("cpu")  # CPU만 사용
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
                dim = embedding_dim if embedding_dim else 64
                dummy_emb = np.zeros((end_idx - i, dim), dtype=np.float32)
                embeddings.append(dummy_emb)
    
    if embeddings:
        return np.vstack(embeddings)
    else:
        dim = embedding_dim if embedding_dim else 64
        return np.zeros((n_samples, dim), dtype=np.float32)


def run_cnn_sequence_experiment(config: PipelineConfig | None = None) -> dict:
    """1D CNN 시퀀스 실험 실행."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    logger.info("=" * 60)
    logger.info("1D CNN 시퀀스 실험 시작")
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
    
    # 3. CNN 인코더 학습 (지도 학습)
    with log_time(logger, "CNN 인코더 학습"):
        cnn_model = _train_cnn_encoder(train_sequences, y_true, epochs=15, batch_size=8)
    
    # 4. CNN 임베딩 추출
    with log_time(logger, "CNN 임베딩 추출"):
        train_embeddings = _extract_cnn_embeddings(cnn_model, train_sequences)
        test_embeddings = _extract_cnn_embeddings(cnn_model, test_sequences)
        logger.info("임베딩 shape: train=%s, test=%s", train_embeddings.shape, test_embeddings.shape)
    
    # 5. 탭 피처 로드 (도메인·공정 피처만, latent 제외)
    with log_time(logger, "탭 피처 로드"):
        train_tabular, y, metadata = _load_tree_data(config)
        feature_cols = _get_feature_columns(train_tabular, config)
        
        # latent 제외 (A 타입)
        tabular_cols = [col for col in feature_cols if not col.startswith("latent_")]
        logger.info("탭 피처 수: %d (latent 제외)", len(tabular_cols))
    
    # 6. CNN 임베딩 + 탭 피처 결합
    with log_time(logger, "피처 결합"):
        train_tabular_features = train_tabular[tabular_cols].values.astype(np.float32)
        train_combined = np.hstack([train_tabular_features, train_embeddings])
        
        # 테스트 데이터도 준비
        test_tabular = pd.read_pickle(config.output_dir / "processed" / "test_tree_ready.pkl")
        test_tabular_features = test_tabular[tabular_cols].values.astype(np.float32)
        test_combined = np.hstack([test_tabular_features, test_embeddings])
        
        logger.info("결합 피처 shape: train=%s, test=%s", train_combined.shape, test_combined.shape)
    
    # 7. CatBoost로 학습 (CNN 임베딩 포함)
    try:
        import catboost as cb
    except ImportError:
        raise ImportError("catboost가 필요합니다.")
    
    folds = metadata["folds"]
    oof = np.zeros_like(y_true, dtype=np.float32)
    
    # NumPy 배열로 변환된 경우 CatBoost는 cat_features를 인식하지 못함
    # 따라서 cat_features를 사용하지 않고 모두 연속형으로 처리
    cat_features = None
    
    with log_time(logger, "CatBoost 학습 (CNN 임베딩 포함)"):
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
            if cat_features:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    cat_features=cat_features,
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                )
            
            oof[valid_idx] = model.predict_proba(X_valid, ntree_end=model.best_iteration_)[:, 1]
            
            # 모델 저장
            model_dir = config.output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"catboost_cnn_fold{fold}.joblib"
            joblib.dump(model, model_path)
    
    # 8. 성능 평가
    roc_auc = roc_auc_score(y_true, oof)
    ap = average_precision_score(y_true, oof)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("1D CNN + CatBoost 결과")
    logger.info("=" * 60)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("AP: %.4f", ap)
    
    # OOF 저장
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_cnn_catboost.npy", oof)
    
    # 결과 저장
    results = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "n_tabular_features": len(tabular_cols),
        "n_cnn_embeddings": train_embeddings.shape[1],
        "n_combined_features": train_combined.shape[1],
    }
    
    results_path = config.output_dir / "logs" / "cnn_sequence_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    
    # CNN 모델 저장
    cnn_model_dir = config.output_dir / "models"
    cnn_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cnn_model.state_dict(), cnn_model_dir / "cnn1d_encoder.pt")
    logger.info("CNN 모델 저장: %s", cnn_model_dir / "cnn1d_encoder.pt")
    
    return results


def tune_cnn_hyperparameters(
    config: PipelineConfig | None = None,
    n_trials: int = 20,
) -> dict:
    """CNN 하이퍼파라미터 튜닝."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna가 필요합니다. pip install optuna")
    
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    logger.info("=" * 60)
    logger.info("CNN 하이퍼파라미터 튜닝 시작")
    logger.info("=" * 60)
    
    # 데이터 준비
    train_df, test_df = load_train_test(config)
    y_true = load_numpy(config.output_dir / "processed" / "train_y.npy")
    train_sequences = _extract_sequences(train_df)
    
    # 탭 피처 준비
    train_tabular, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_tabular, config)
    tabular_cols = [col for col in feature_cols if not col.startswith("latent_")]
    folds = metadata["folds"]
    
    def objective(trial):
        # CNN 하이퍼파라미터
        embedding_dim = trial.suggest_int("embedding_dim", 32, 128, step=16)
        conv1_channels = trial.suggest_int("conv1_channels", 16, 64, step=16)
        conv2_channels = trial.suggest_int("conv2_channels", 32, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 16, step=4)
        epochs = trial.suggest_int("epochs", 10, 20)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        
        # CNN 학습
        try:
            cnn_model = _train_cnn_encoder(
                train_sequences,
                y_true,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_dim=embedding_dim,
                conv1_channels=conv1_channels,
                conv2_channels=conv2_channels,
                dropout=dropout,
                weight_decay=weight_decay,
            )
        except Exception as e:
            logger.warning("CNN 학습 실패: %s", e)
            return 0.0
        
        # CNN 임베딩 추출
        train_embeddings = _extract_cnn_embeddings(cnn_model, train_sequences, batch_size=16)
        
        # 탭 피처 + CNN 임베딩 결합
        train_tabular_features = train_tabular[tabular_cols].values.astype(np.float32)
        train_combined = np.hstack([train_tabular_features, train_embeddings])
        
        # CatBoost로 평가
        try:
            import catboost as cb
        except ImportError:
            return 0.0
        
        # 카테고리 컬럼 자동 감지 (결합된 배열 기준 인덱스)
        cat_features = [
            i for i, col in enumerate(tabular_cols)
            if train_tabular[col].dtype in ["object", "category", "int64"] and train_tabular[col].nunique() < 100
        ]
        if not cat_features:
            cat_features = None
        
        pos_weight = (y_true == 0).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 1.0
        oof = np.zeros_like(y_true, dtype=np.float32)
        
        for fold_str, idx_dict in folds.items():
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_combined[train_idx]
            X_valid = train_combined[valid_idx]
            y_train = y_true[train_idx]
            y_valid = y_true[valid_idx]
            
            model = cb.CatBoostClassifier(
                iterations=300,
                depth=7,
                learning_rate=0.05,
                min_data_in_leaf=20,
                subsample=0.8,
                colsample_bylevel=0.8,
                scale_pos_weight=pos_weight,
                random_seed=config.random_seed,
                verbose=False,
                early_stopping_rounds=50,
            )
            
            if cat_features is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    cat_features=cat_features,
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                )
            
            oof[valid_idx] = model.predict_proba(X_valid, ntree_end=model.best_iteration_)[:, 1]
        
        roc_auc = roc_auc_score(y_true, oof)
        return roc_auc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", study.best_value)
    
    # 최적 파라미터로 최종 실험 실행
    logger.info("\n최적 파라미터로 최종 모델 학습 중...")
    return _run_cnn_with_params(config, best_params)


def _run_cnn_with_params(config: PipelineConfig, params: dict) -> dict:
    """최적 파라미터로 CNN 실험 실행."""
    train_df, test_df = load_train_test(config)
    y_true = load_numpy(config.output_dir / "processed" / "train_y.npy")
    train_sequences = _extract_sequences(train_df)
    test_sequences = _extract_sequences(test_df)
    
    # CNN 학습
    cnn_model = _train_cnn_encoder(
        train_sequences,
        y_true,
        epochs=params.get("epochs", 15),
        batch_size=params.get("batch_size", 8),
        learning_rate=params.get("learning_rate", 1e-3),
        embedding_dim=params.get("embedding_dim", 64),
        conv1_channels=params.get("conv1_channels", 32),
        conv2_channels=params.get("conv2_channels", 64),
        dropout=params.get("dropout", 0.0),
        weight_decay=params.get("weight_decay", 0.0),
    )
    
    # 임베딩 추출
    train_embeddings = _extract_cnn_embeddings(cnn_model, train_sequences)
    test_embeddings = _extract_cnn_embeddings(cnn_model, test_sequences)
    
    # 탭 피처 + CNN 임베딩 결합
    train_tabular, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_tabular, config)
    tabular_cols = [col for col in feature_cols if not col.startswith("latent_")]
    
    train_tabular_features = train_tabular[tabular_cols].values.astype(np.float32)
    train_combined = np.hstack([train_tabular_features, train_embeddings])
    
    test_tabular = pd.read_pickle(config.output_dir / "processed" / "test_tree_ready.pkl")
    test_tabular_features = test_tabular[tabular_cols].values.astype(np.float32)
    test_combined = np.hstack([test_tabular_features, test_embeddings])
    
    # CatBoost 학습
    import catboost as cb
    
    folds = metadata["folds"]
    oof = np.zeros_like(y_true, dtype=np.float32)
    pos_weight = (y_true == 0).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 1.0
    
    # 카테고리 컬럼 자동 감지 (결합된 배열 기준 인덱스)
    cat_features = [
        i for i, col in enumerate(tabular_cols)
        if train_tabular[col].dtype in ["object", "category", "int64"] and train_tabular[col].nunique() < 100
    ]
    if not cat_features:
        cat_features = None
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        if cat_features is not None:
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features=cat_features,
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
            )
        
        oof[valid_idx] = model.predict_proba(X_valid, ntree_end=model.best_iteration_)[:, 1]
        
        model_path = model_dir / f"catboost_cnn_tuned_fold{fold}.joblib"
        joblib.dump(model, model_path)
    
    # 성능 평가
    roc_auc = roc_auc_score(y_true, oof)
    ap = average_precision_score(y_true, oof)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("튜닝된 CNN + CatBoost 결과")
    logger.info("=" * 60)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("AP: %.4f", ap)
    
    # 결과 저장
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_cnn_catboost_tuned.npy", oof)
    
    results = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "params": params,
    }
    
    results_path = config.output_dir / "logs" / "cnn_tuning_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    
    # CNN 모델 저장
    cnn_model_dir = config.output_dir / "models"
    cnn_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cnn_model.state_dict(), cnn_model_dir / "cnn1d_encoder_tuned.pt")
    
    return results


if __name__ == "__main__":
    from .config import PipelineConfig
    
    config = PipelineConfig()
    run_cnn_sequence_experiment(config)

