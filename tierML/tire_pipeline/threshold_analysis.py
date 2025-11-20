"""3단계: Threshold & Calibration 분석.

1. OOF 예측 기준으로 threshold 0.1~0.9로 F1/Precision/Recall/AP 곡선 분석
2. 비즈니스 기준에 맞는 threshold 선택
3. Isotonic/Platt 캘리브레이션 효과 비교
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PipelineConfig
from .data_utils import load_numpy, save_numpy
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


def analyze_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """다양한 threshold에서 성능 지표 계산."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "threshold": threshold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "n_positive": y_pred.sum(),
        })
    
    df = pd.DataFrame(results)
    df["ap"] = average_precision_score(y_true, y_probs)
    df["roc_auc"] = roc_auc_score(y_true, y_probs)
    
    return df


def calibrate_probs(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    method: str = "isotonic",  # "isotonic" or "platt"
) -> np.ndarray:
    """확률 캘리브레이션."""
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_probs, y_true)
        calibrated = calibrator.predict(y_probs)
    elif method == "platt":
        # Platt scaling (sigmoid)
        lr = LogisticRegression()
        lr.fit(y_probs.reshape(-1, 1), y_true)
        calibrated = lr.predict_proba(y_probs.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.clip(calibrated, 0, 1)


def run_threshold_analysis(
    config: PipelineConfig | None = None,
    feature_type: str = "C",  # A, B, C, D 중 선택
) -> dict:
    """Threshold 및 캘리브레이션 분석 실행."""
    config = config or PipelineConfig()
    config.ensure_output_dirs()
    
    # OOF 예측 로드
    predictions_dir = config.output_dir / "predictions"
    oof_path = predictions_dir / f"oof_probs_ablation_{feature_type}.npy"
    
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF 예측 파일이 없습니다: {oof_path}")
    
    y_probs = load_numpy(oof_path)
    y_true = load_numpy(config.output_dir / "processed" / "train_y.npy")
    
    logger.info("=" * 60)
    logger.info("Threshold & Calibration 분석 시작")
    logger.info("=" * 60)
    logger.info("피처 타입: %s", feature_type)
    logger.info("데이터 크기: %d", len(y_true))
    logger.info("불량 비율: %.4f", y_true.mean())
    
    # 1. 원본 확률로 threshold 분석
    logger.info("")
    logger.info("-" * 60)
    logger.info("1. 원본 확률 threshold 분석")
    logger.info("-" * 60)
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_df = analyze_thresholds(y_true, y_probs, thresholds)
    
    logger.info("최고 F1: %.4f (threshold=%.2f)", 
                threshold_df["f1"].max(), 
                threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"])
    logger.info("최고 Precision: %.4f (threshold=%.2f)", 
                threshold_df["precision"].max(), 
                threshold_df.loc[threshold_df["precision"].idxmax(), "threshold"])
    logger.info("최고 Recall: %.4f (threshold=%.2f)", 
                threshold_df["recall"].max(), 
                threshold_df.loc[threshold_df["recall"].idxmax(), "threshold"])
    
    # 결과 저장
    results_dir = config.output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(results_dir / f"threshold_analysis_{feature_type}.csv", index=False)
    logger.info("결과 저장: %s", results_dir / f"threshold_analysis_{feature_type}.csv")
    
    # 2. 캘리브레이션 비교
    logger.info("")
    logger.info("-" * 60)
    logger.info("2. 캘리브레이션 효과 비교")
    logger.info("-" * 60)
    
    calibration_results = {}
    
    # Isotonic
    logger.info("Isotonic 캘리브레이션 적용 중...")
    y_probs_isotonic = calibrate_probs(y_true, y_probs, method="isotonic")
    threshold_df_isotonic = analyze_thresholds(y_true, y_probs_isotonic, thresholds)
    
    best_f1_iso = threshold_df_isotonic["f1"].max()
    best_thresh_iso = threshold_df_isotonic.loc[threshold_df_isotonic["f1"].idxmax(), "threshold"]
    logger.info("Isotonic 최고 F1: %.4f (threshold=%.2f)", best_f1_iso, best_thresh_iso)
    
    calibration_results["isotonic"] = {
        "best_f1": float(best_f1_iso),
        "best_threshold": float(best_thresh_iso),
        "roc_auc": float(roc_auc_score(y_true, y_probs_isotonic)),
        "ap": float(average_precision_score(y_true, y_probs_isotonic)),
    }
    
    # Platt
    logger.info("Platt 캘리브레이션 적용 중...")
    y_probs_platt = calibrate_probs(y_true, y_probs, method="platt")
    threshold_df_platt = analyze_thresholds(y_true, y_probs_platt, thresholds)
    
    best_f1_platt = threshold_df_platt["f1"].max()
    best_thresh_platt = threshold_df_platt.loc[threshold_df_platt["f1"].idxmax(), "threshold"]
    logger.info("Platt 최고 F1: %.4f (threshold=%.2f)", best_f1_platt, best_thresh_platt)
    
    calibration_results["platt"] = {
        "best_f1": float(best_f1_platt),
        "best_threshold": float(best_thresh_platt),
        "roc_auc": float(roc_auc_score(y_true, y_probs_platt)),
        "ap": float(average_precision_score(y_true, y_probs_platt)),
    }
    
    # 원본
    best_f1_orig = threshold_df["f1"].max()
    best_thresh_orig = threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"]
    calibration_results["original"] = {
        "best_f1": float(best_f1_orig),
        "best_threshold": float(best_thresh_orig),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "ap": float(average_precision_score(y_true, y_probs)),
    }
    
    # 결과 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("캘리브레이션 비교 결과")
    logger.info("=" * 60)
    for method, metrics in calibration_results.items():
        logger.info(
            "%s: F1=%.4f (threshold=%.2f), ROC-AUC=%.4f, AP=%.4f",
            method,
            metrics["best_f1"],
            metrics["best_threshold"],
            metrics["roc_auc"],
            metrics["ap"],
        )
    
    # 캘리브레이션된 확률 저장
    save_numpy(predictions_dir / f"calibrated_isotonic_{feature_type}.npy", y_probs_isotonic)
    save_numpy(predictions_dir / f"calibrated_platt_{feature_type}.npy", y_probs_platt)
    
    # 전체 결과 저장
    all_results = {
        "feature_type": feature_type,
        "threshold_analysis": threshold_df.to_dict("records"),
        "calibration_comparison": calibration_results,
    }
    
    results_path = results_dir / f"threshold_calibration_results_{feature_type}.jsonl"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    logger.info("전체 결과 저장: %s", results_path)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-type", choices=["A", "B", "C", "D"], default="C")
    args = parser.parse_args()
    
    run_threshold_analysis(feature_type=args.feature_type)

