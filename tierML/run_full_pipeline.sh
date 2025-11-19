#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Users/yangjinmo/tierML"
cd "$PROJECT_ROOT"

echo "[1/7] 데이터 분석"
python -m tire_pipeline.cli --step analysis

echo "[2/7] AutoEncoder 포함 트리 파이프라인"
python -m tire_pipeline.cli --step tree_pipeline

echo "[3/7] NumPy 전처리 (logistic)"
python -m tire_pipeline.cli --step preprocess --model-type logistic

echo "[4/7] 모델 학습 (logistic)"
python -m tire_pipeline.cli --step train --model-type logistic

echo "[5/7] 모델 추론"
python -m tire_pipeline.cli --step infer --model-type logistic

echo "[6/7] 캘리브레이션"
python -m tire_pipeline.cli --step calibrate --model-type logistic

echo "[7/7] 앙상블 및 제출 생성"
python -m tire_pipeline.cli --step ensemble --model-type logistic "$@"

echo "✅ 전체 파이프라인 완료"

