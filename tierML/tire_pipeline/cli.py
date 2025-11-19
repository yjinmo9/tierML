"""파이프라인 CLI."""

from __future__ import annotations

import argparse

from .calibration import run_calibration
from .config import PipelineConfig
from .data_analysis import run_data_analysis
from .ensemble_submission import run_ensemble_and_submission
from .feature_generation import run_feature_generation
from .logging_utils import setup_logger
from .model_inference import run_inference
from .model_training import run_model_training
from .preprocessing import run_preprocessing
from .tree_model_training import compare_tree_models, run_tree_model_training
from .tree_hyperparameter_tuning import tune_all_models, tune_catboost, tune_lgbm, tune_xgb


tree_logger = setup_logger("tree_pipeline")


def run_tree_pipeline(config: PipelineConfig, skip_autoencoder: bool) -> None:
    if not skip_autoencoder:
        try:
            from .autoencoder import train_autoencoder

            tree_logger.info("AutoEncoder 실행")
            train_autoencoder(config)
        except ImportError as exc:
            tree_logger.warning("PyTorch 미설치로 AutoEncoder를 건너뜁니다: %s", exc)
    else:
        tree_logger.info("AutoEncoder 건너뛰기 옵션이 설정되었습니다.")

    tree_logger.info("피처 생성 실행")
    run_feature_generation(config)
    tree_logger.info("전처리 실행")
    run_preprocessing(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="타이어 불량 예측 파이프라인")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/yangjinmo/tierML/5-ai-and-datascience-competition",
        help="CSV 데이터가 위치한 디렉터리",
    )
    parser.add_argument("--output-dir", type=str, default="/Users/yangjinmo/tierML/artifacts")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tree", "catboost", "lgbm", "xgb", "linear", "logistic", "mlp", "svm", "autoencoder"],
        help="전처리/학습 시 사용할 모델 유형",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        choices=["standard", "minmax"],
        help="비트리 모델용 스케일러 타입",
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="tree_pipeline 실행 시 AutoEncoder 단계를 건너뜁니다.",
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=[
            "analysis",
            "autoencoder",
            "tree_pipeline",
            "features",
            "preprocess",
            "train",
            "train_tree",
            "compare_trees",
            "tune_tree",
            "infer",
            "calibrate",
            "ensemble",
        ],
    )
    parser.add_argument("--extra-preds", type=str, nargs="*", help="추가 앙상블 확률 파일 경로")
    parser.add_argument(
        "--tree-model",
        type=str,
        choices=["lgbm", "xgb", "catboost", "all"],
        default="all",
        help="트리 모델 선택 (train_tree, tune_tree step에서 사용)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="하이퍼파라미터 튜닝 시도 횟수 (tune_tree step에서 사용)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = (
        PipelineConfig()
        .set_data_dir(args.data_dir)
        .override(output_dir=args.output_dir, model_type=args.model_type, scaler_type=args.scaler_type)
    )
    step = args.step
    if step == "analysis":
        run_data_analysis(config)
    elif step == "autoencoder":
        from .autoencoder import train_autoencoder

        train_autoencoder(config)
    elif step == "tree_pipeline":
        run_tree_pipeline(config, skip_autoencoder=args.skip_ae)
    elif step == "features":
        run_feature_generation(config)
    elif step == "preprocess":
        run_preprocessing(config)
    elif step == "train":
        run_model_training(config)
    elif step == "train_tree":
        run_tree_model_training(config, model_name=args.tree_model)
    elif step == "compare_trees":
        compare_tree_models(config)
    elif step == "tune_tree":
        if args.tree_model == "all":
            tune_all_models(config, n_trials=args.n_trials)
        elif args.tree_model == "lgbm":
            tune_lgbm(config, n_trials=args.n_trials)
        elif args.tree_model == "xgb":
            tune_xgb(config, n_trials=args.n_trials)
        elif args.tree_model == "catboost":
            tune_catboost(config, n_trials=args.n_trials)
    elif step == "infer":
        run_inference(config)
    elif step == "calibrate":
        run_calibration(config)
    elif step == "ensemble":
        run_ensemble_and_submission(config, args.extra_preds)
    else:
        raise ValueError(f"알 수 없는 step: {step}")


if __name__ == "__main__":
    main()

