"""트리 모델 하이퍼파라미터 튜닝."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import setup_logger

logger = setup_logger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


def _load_tree_data(config: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """트리 모델용 전처리된 데이터 로딩."""
    processed_dir = config.output_dir / "processed"
    train_df = pd.read_pickle(processed_dir / "train_tree_ready.pkl")
    y = load_numpy(processed_dir / "train_y.npy")
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    
    # feature_generation.py에서 생성한 피처 병합
    features_dir = config.output_dir / "features"
    train_features_path = features_dir / "train_features.pkl"
    if train_features_path.exists():
        train_features = pd.read_pickle(train_features_path)
        # train_tree_ready에 이미 있는 피처는 제외 (중복 방지)
        existing_cols = set(train_df.columns)
        new_cols = [col for col in train_features.columns if col not in existing_cols]
        if new_cols:
            # 인덱스로 병합 (인덱스가 같은 경우)
            if len(train_df) == len(train_features) and train_df.index.equals(train_features.index):
                train_df = train_df.join(train_features[new_cols], how="left")
                logger.info("생성된 피처 %d개 병합 완료 (인덱스 기준)", len(new_cols))
            else:
                # ID 컬럼으로 병합 시도
                id_col = config.id_column
                if id_col in train_df.columns and id_col in train_features.columns:
                    train_df = train_df.merge(
                        train_features[[id_col] + new_cols],
                        on=id_col,
                        how="left"
                    )
                    logger.info("생성된 피처 %d개 병합 완료 (ID 컬럼 기준)", len(new_cols))
                else:
                    logger.warning("피처 병합 실패: 인덱스나 ID 컬럼이 일치하지 않음")
    
    return train_df, y, metadata


def _get_feature_columns(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    """ID와 타겟을 제외한 피처 컬럼 리스트."""
    exclude = {config.id_column, config.target_column}
    return [col for col in df.columns if col not in exclude]


def tune_lgbm(config: PipelineConfig, n_trials: int = 50) -> dict:
    """LightGBM 하이퍼파라미터 튜닝."""
    if lgb is None:
        raise ImportError("lightgbm이 설치되지 않았습니다.")
    
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna가 없어 간단한 grid search로 대체합니다.")
        return _grid_search_lgbm(config)
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "random_state": config.random_seed,
            "verbose": -1,
        }
        
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params["scale_pos_weight"] = pos_weight
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            oof[valid_idx] = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        return roc_auc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", study.best_value)
    
    # 최적 파라미터로 최종 모델 학습
    return _train_with_params_lgbm(config, best_params)


def _grid_search_lgbm(config: PipelineConfig) -> dict:
    """간단한 grid search (Optuna 없을 때)."""
    if lgb is None:
        raise ImportError("lightgbm이 설치되지 않았습니다.")
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    # 간단한 grid
    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [6, 7, 8],
        "num_leaves": [31, 50, 70],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_data_in_leaf": [15, 20, 30],
    }
    
    best_score = 0
    best_params = None
    
    from itertools import product
    
    for params_combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params_combo))
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params.update({
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": pos_weight,
            "random_state": config.random_seed,
            "verbose": -1,
        })
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            oof[valid_idx] = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        if roc_auc > best_score:
            best_score = roc_auc
            best_params = params.copy()
            logger.info("새 최고 점수: %.4f (params: %s)", roc_auc, params)
    
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", best_score)
    
    return _train_with_params_lgbm(config, best_params)


def _train_with_params_lgbm(config: PipelineConfig, params: dict) -> dict:
    """주어진 파라미터로 최종 모델 학습."""
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    oof = np.zeros_like(y, dtype=np.float32)
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_str, idx_dict in folds.items():
        fold = int(fold_str)
        train_idx = np.array(idx_dict["train_idx"])
        valid_idx = np.array(idx_dict["valid_idx"])
        
        X_train = train_df.loc[train_idx, feature_cols]
        X_valid = train_df.loc[valid_idx, feature_cols]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        
        model_path = model_dir / f"lgbm_tuned_fold{fold}.joblib"
        joblib.dump(model, model_path)
        logger.info("튜닝된 모델 저장: %s", model_path)
        
        oof[valid_idx] = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
    
    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    logger.info("튜닝 후 OOF ROC-AUC: %.4f | AP: %.4f", roc_auc, ap)
    
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_lgbm_tuned.npy", oof)
    
    metrics = {"roc_auc": roc_auc, "average_precision": ap, "params": params}
    metrics_path = config.output_dir / "logs"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / "train_metrics_lgbm_tuned.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    
    return metrics


def tune_catboost(config: PipelineConfig, n_trials: int = 50) -> dict:
    """CatBoost 하이퍼파라미터 튜닝."""
    if cb is None:
        raise ImportError("catboost가 설치되지 않았습니다.")
    
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna가 없어 간단한 grid search로 대체합니다.")
        return _grid_search_catboost(config)
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    # 카테고리 컬럼 자동 감지
    cat_features = [
        i for i, col in enumerate(feature_cols)
        if train_df[col].dtype in ["object", "category", "int64"] and train_df[col].nunique() < 100
    ]
    
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "random_seed": config.random_seed,
            "verbose": False,
            "early_stopping_rounds": 50,
        }
        
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params["scale_pos_weight"] = pos_weight
        if cat_features:
            params["cat_features"] = cat_features
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
            )
            oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        return roc_auc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", study.best_value)
    
    return _train_with_params_catboost(config, best_params)


def _grid_search_catboost(config: PipelineConfig) -> dict:
    """간단한 grid search (Optuna 없을 때)."""
    if cb is None:
        raise ImportError("catboost가 설치되지 않았습니다.")
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    cat_features = [
        i for i, col in enumerate(feature_cols)
        if train_df[col].dtype in ["object", "category", "int64"] and train_df[col].nunique() < 100
    ]
    
    param_grid = {
        "iterations": [300, 500, 700],
        "depth": [6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_data_in_leaf": [15, 20, 30],
    }
    
    best_score = 0
    best_params = None
    
    from itertools import product
    
    for params_combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params_combo))
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params.update({
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "scale_pos_weight": pos_weight,
            "random_seed": config.random_seed,
            "verbose": False,
            "early_stopping_rounds": 50,
        })
        if cat_features:
            params["cat_features"] = cat_features
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
            )
            oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        if roc_auc > best_score:
            best_score = roc_auc
            best_params = params.copy()
            logger.info("새 최고 점수: %.4f (params: %s)", roc_auc, params)
    
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", best_score)
    
    return _train_with_params_catboost(config, best_params)


def _train_with_params_catboost(config: PipelineConfig, params: dict) -> dict:
    """주어진 파라미터로 최종 CatBoost 모델 학습."""
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    oof = np.zeros_like(y, dtype=np.float32)
    
    cat_features = [
        i for i, col in enumerate(feature_cols)
        if train_df[col].dtype in ["object", "category", "int64"] and train_df[col].nunique() < 100
    ]
    if cat_features and "cat_features" not in params:
        params["cat_features"] = cat_features
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_str, idx_dict in folds.items():
        fold = int(fold_str)
        train_idx = np.array(idx_dict["train_idx"])
        valid_idx = np.array(idx_dict["valid_idx"])
        
        X_train = train_df.loc[train_idx, feature_cols]
        X_valid = train_df.loc[valid_idx, feature_cols]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
        )
        
        model_path = model_dir / f"catboost_tuned_fold{fold}.joblib"
        joblib.dump(model, model_path)
        logger.info("튜닝된 모델 저장: %s", model_path)
        
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
    
    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    logger.info("튜닝 후 OOF ROC-AUC: %.4f | AP: %.4f", roc_auc, ap)
    
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_catboost_tuned.npy", oof)
    
    metrics = {"roc_auc": roc_auc, "average_precision": ap, "params": params}
    metrics_path = config.output_dir / "logs"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / "train_metrics_catboost_tuned.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    
    return metrics


def tune_xgb(config: PipelineConfig, n_trials: int = 50) -> dict:
    """XGBoost 하이퍼파라미터 튜닝."""
    if xgb is None:
        raise ImportError("xgboost가 설치되지 않았습니다.")
    
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna가 없어 간단한 grid search로 대체합니다.")
        return _grid_search_xgb(config)
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": config.random_seed,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
        }
        
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params["scale_pos_weight"] = pos_weight
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        return roc_auc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", study.best_value)
    
    return _train_with_params_xgb(config, best_params)


def _grid_search_xgb(config: PipelineConfig) -> dict:
    """간단한 grid search (Optuna 없을 때)."""
    if xgb is None:
        raise ImportError("xgboost가 설치되지 않았습니다.")
    
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    
    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [3, 5, 7],
    }
    
    best_score = 0
    best_params = None
    
    from itertools import product
    
    for params_combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params_combo))
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params.update({
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": pos_weight,
            "random_state": config.random_seed,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
        })
        
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_str, idx_dict in folds.items():
            fold = int(fold_str)
            train_idx = np.array(idx_dict["train_idx"])
            valid_idx = np.array(idx_dict["valid_idx"])
            
            X_train = train_df.loc[train_idx, feature_cols]
            X_valid = train_df.loc[valid_idx, feature_cols]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        
        roc_auc = roc_auc_score(y, oof)
        if roc_auc > best_score:
            best_score = roc_auc
            best_params = params.copy()
            logger.info("새 최고 점수: %.4f (params: %s)", roc_auc, params)
    
    logger.info("최적 파라미터: %s", best_params)
    logger.info("최적 ROC-AUC: %.4f", best_score)
    
    return _train_with_params_xgb(config, best_params)


def _train_with_params_xgb(config: PipelineConfig, params: dict) -> dict:
    """주어진 파라미터로 최종 XGBoost 모델 학습."""
    train_df, y, metadata = _load_tree_data(config)
    feature_cols = _get_feature_columns(train_df, config)
    folds = metadata["folds"]
    oof = np.zeros_like(y, dtype=np.float32)
    
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_str, idx_dict in folds.items():
        fold = int(fold_str)
        train_idx = np.array(idx_dict["train_idx"])
        valid_idx = np.array(idx_dict["valid_idx"])
        
        X_train = train_df.loc[train_idx, feature_cols]
        X_valid = train_df.loc[valid_idx, feature_cols]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        
        model_path = model_dir / f"xgb_tuned_fold{fold}.joblib"
        joblib.dump(model, model_path)
        logger.info("튜닝된 모델 저장: %s", model_path)
        
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
    
    roc_auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    logger.info("튜닝 후 OOF ROC-AUC: %.4f | AP: %.4f", roc_auc, ap)
    
    predictions_dir = config.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(predictions_dir / "oof_probs_xgb_tuned.npy", oof)
    
    metrics = {"roc_auc": roc_auc, "average_precision": ap, "params": params}
    metrics_path = config.output_dir / "logs"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / "train_metrics_xgb_tuned.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    
    return metrics


def tune_all_models(config: PipelineConfig | None = None, n_trials: int = 50) -> dict[str, dict]:
    """모든 트리 모델을 튜닝하고 비교."""
    config = config or PipelineConfig()
    results = {}
    
    for model_name, tune_func in [("lgbm", tune_lgbm), ("xgb", tune_xgb), ("catboost", tune_catboost)]:
        try:
            logger.info("=" * 50)
            logger.info("하이퍼파라미터 튜닝: %s", model_name.upper())
            logger.info("=" * 50)
            metrics = tune_func(config, n_trials=n_trials)
            results[model_name] = metrics
        except Exception as e:
            logger.error("%s 튜닝 실패: %s", model_name, e)
            results[model_name] = {"error": str(e)}
    
    logger.info("=" * 50)
    logger.info("튜닝 후 모델 비교 결과")
    logger.info("=" * 50)
    for name, metrics in results.items():
        if "error" not in metrics:
            logger.info("%s: ROC-AUC=%.4f, AP=%.4f", name, metrics["roc_auc"], metrics["average_precision"])
        else:
            logger.info("%s: 실패 (%s)", name, metrics["error"])
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "catboost", "all"], default="all")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    config = PipelineConfig()
    if args.model == "all":
        tune_all_models(config, n_trials=args.n_trials)
    elif args.model == "lgbm":
        tune_lgbm(config, n_trials=args.n_trials)
    elif args.model == "xgb":
        tune_xgb(config, n_trials=args.n_trials)
    elif args.model == "catboost":
        tune_catboost(config, n_trials=args.n_trials)

