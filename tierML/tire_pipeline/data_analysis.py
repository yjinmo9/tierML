"""1단계: 데이터 분석."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import PipelineConfig
from .data_utils import load_train_test, save_markdown_report
from .logging_utils import log_time, setup_logger

logger = setup_logger(__name__)


def _describe_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    stats = []
    for col in df.columns:
        series = df[col]
        stats.append(
            {
                "dataset": name,
                "column": col,
                "dtype": str(series.dtype),
                "n_unique": series.nunique(dropna=True),
                "missing_ratio": float(series.isna().mean()),
                "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
                "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else None,
            }
        )
    return pd.DataFrame(stats)


def _summarize_high_dim(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return {}
    values = df[cols]
    return {
        "prefix": prefix,
        "n_cols": len(cols),
        "mean_mean": float(values.mean(axis=1).mean()),
        "mean_std": float(values.std(axis=1).mean()),
        "global_std": float(values.stack().std()),
    }


def _value_counts_section(df: pd.DataFrame, columns: List[str], top_k: int = 5) -> str:
    parts = []
    for col in columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(top_k)
        formatted = "\n".join(f"- {idx}: {count}" for idx, count in vc.items())
        parts.append(f"### {col}\n{formatted}")
    return "\n\n".join(parts)


def run_data_analysis(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    config.ensure_output_dirs()

    with log_time(logger, "데이터 로딩"):
        train_df, test_df = load_train_test(config)

    with log_time(logger, "컬럼 통계 산출"):
        combined_stats = pd.concat(
            [_describe_columns(train_df, "train"), _describe_columns(test_df, "test")],
            ignore_index=True,
        )
        stats_path = config.output_dir / "reports" / "column_stats.parquet"
        combined_stats.to_parquet(stats_path, index=False)
        logger.info("컬럼 통계 저장: %s", stats_path)

    with log_time(logger, "고차원 요약"):
        prefixes = ["x", "y", "p"]
        hi_sections = []
        for prefix in prefixes:
            summary = _summarize_high_dim(train_df, prefix)
            if summary:
                hi_sections.append(
                    f"- `{prefix}` 계열 {summary['n_cols']}개 | 평균의 평균 "
                    f"{summary['mean_mean']:.4f} | 평균 표준편차 {summary['mean_std']:.4f}"
                )
        high_dim_section = "\n".join(hi_sections) if hi_sections else "해당 없음"

    expected_cols = ["Mass_Pilot", "Plant", "NG"]
    categorical_section = _value_counts_section(train_df, expected_cols)

    target_series = train_df[config.target_column]
    if pd.api.types.is_numeric_dtype(target_series):
        imbalance = float(target_series.mean())
    else:
        positive_tokens = {"ng", "1", "true", "bad"}
        target_binary = target_series.astype(str).str.lower().isin(positive_tokens).astype(float)
        imbalance = float(target_binary.mean())
    summary_sections = {
        "데이터 개요": (
            f"- train: {len(train_df):,}행 x {train_df.shape[1]}열\n"
            f"- test: {len(test_df):,}행 x {test_df.shape[1]}열\n"
            f"- 불량 비율(NG=1): {imbalance:.4f}"
        ),
        "주요 범주형 분포": categorical_section or "해당 컬럼이 없습니다.",
        "고차원 좌표 요약": high_dim_section,
    }

    report_path = config.output_dir / "reports" / "eda_summary.md"
    save_markdown_report(report_path, summary_sections)


if __name__ == "__main__":
    run_data_analysis()

