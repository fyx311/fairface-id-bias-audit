#!/usr/bin/env python
"""Evaluate a real face-verification score CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairface_id.pipeline import evaluate_score_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit face-verification scores by demographic group.")
    parser.add_argument("--scores_csv", required=True, help="CSV containing label, score, and group columns.")
    parser.add_argument("--out_dir", default="outputs/eval", help="Output directory.")
    parser.add_argument("--group_col", default="group", help="Group column name.")
    parser.add_argument("--label_col", default="label", help="Binary label column name: 1=same person, 0=different person.")
    parser.add_argument("--score_col", default="score", help="Similarity score column name.")
    parser.add_argument("--threshold_metric", default="balanced_accuracy", help="Metric used to select thresholds.")
    parser.add_argument("--test_size", type=float, default=0.4, help="Fraction for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    summary = evaluate_score_file(
        scores_csv=args.scores_csv,
        out_dir=args.out_dir,
        group_col=args.group_col,
        label_col=args.label_col,
        score_col=args.score_col,
        threshold_metric=args.threshold_metric,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("Evaluation complete.")
    print(f"Output directory: {args.out_dir}")
    print(f"Global threshold: {summary['global_threshold']:.4f}")
    print("Group thresholds:")
    for group, threshold in summary["group_thresholds"].items():
        print(f"  {group}: {threshold:.4f}")


if __name__ == "__main__":
    main()
