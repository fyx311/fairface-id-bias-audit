"""End-to-end evaluation routines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import ensure_out_dir, read_score_file, train_test_split_by_group
from .metrics import fairness_summary, overall_metrics, per_group_metrics
from .thresholding import select_best_threshold, select_group_thresholds


def evaluate_score_dataframe(df: pd.DataFrame,
                             out_dir: str | Path,
                             threshold_metric: str = "balanced_accuracy",
                             test_size: float = 0.4,
                             seed: int = 42) -> dict:
    """Evaluate baseline global threshold and group-calibrated thresholds."""
    out_dir = ensure_out_dir(out_dir)
    dev_df, test_df = train_test_split_by_group(df, test_size=test_size, seed=seed)

    global_threshold, dev_global_metrics = select_best_threshold(dev_df, metric=threshold_metric)
    group_thresholds = select_group_thresholds(dev_df, metric=threshold_metric)

    global_per_group = per_group_metrics(test_df, threshold=global_threshold)
    calibrated_per_group = per_group_metrics(test_df, threshold_by_group=group_thresholds)
    global_fairness = fairness_summary(global_per_group)
    calibrated_fairness = fairness_summary(calibrated_per_group)

    global_overall = overall_metrics(test_df, threshold=global_threshold)
    calibrated_overall = overall_metrics(test_df, threshold_by_group=group_thresholds)

    global_per_group.to_csv(out_dir / "global_per_group_metrics.csv", index=False)
    calibrated_per_group.to_csv(out_dir / "calibrated_per_group_metrics.csv", index=False)
    global_fairness.to_csv(out_dir / "global_fairness_summary.csv", index=False)
    calibrated_fairness.to_csv(out_dir / "calibrated_fairness_summary.csv", index=False)
    dev_df.to_csv(out_dir / "dev_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    summary = {
        "global_threshold": global_threshold,
        "group_thresholds": group_thresholds,
        "dev_global_metrics": dev_global_metrics,
        "test_global_overall": global_overall,
        "test_calibrated_overall": calibrated_overall,
        "files": {
            "global_per_group": "global_per_group_metrics.csv",
            "calibrated_per_group": "calibrated_per_group_metrics.csv",
            "global_fairness": "global_fairness_summary.csv",
            "calibrated_fairness": "calibrated_fairness_summary.csv",
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_score_file(scores_csv: str | Path,
                        out_dir: str | Path,
                        group_col: str = "group",
                        label_col: str = "label",
                        score_col: str = "score",
                        threshold_metric: str = "balanced_accuracy",
                        test_size: float = 0.4,
                        seed: int = 42) -> dict:
    """Read and evaluate a score CSV."""
    df = read_score_file(scores_csv, label_col=label_col, score_col=score_col, group_col=group_col)
    return evaluate_score_dataframe(
        df=df,
        out_dir=out_dir,
        threshold_metric=threshold_metric,
        test_size=test_size,
        seed=seed,
    )
