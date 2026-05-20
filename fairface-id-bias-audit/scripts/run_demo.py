#!/usr/bin/env python
"""Run a complete demo with synthetic face-verification scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Make local src import work when running from a cloned repository.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairface_id.pipeline import evaluate_score_dataframe
from fairface_id.simulate import simulate_pair_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic fairness-audit demo.")
    parser.add_argument("--out_dir", default="outputs/demo", help="Output directory.")
    parser.add_argument("--n_per_group", type=int, default=600, help="Synthetic pairs per group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = simulate_pair_scores(n_per_group=args.n_per_group, seed=args.seed)
    df.to_csv(out_dir / "synthetic_pairs.csv", index=False)

    summary = evaluate_score_dataframe(df, out_dir=out_dir, seed=args.seed)

    print("Demo complete.")
    print(f"Output directory: {out_dir}")
    print(f"Global threshold: {summary['global_threshold']:.4f}")
    print("Group thresholds:")
    for group, threshold in summary["group_thresholds"].items():
        print(f"  {group}: {threshold:.4f}")
    print("\nOverall metrics:")
    print(json.dumps({
        "global": summary["test_global_overall"],
        "calibrated": summary["test_calibrated_overall"],
    }, indent=2))

    try:
        import matplotlib.pyplot as plt
        global_df = pd.read_csv(out_dir / "global_per_group_metrics.csv")
        calibrated_df = pd.read_csv(out_dir / "calibrated_per_group_metrics.csv")
        plot_df = global_df[["group", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "global"})
        plot_df = plot_df.merge(
            calibrated_df[["group", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "calibrated"}),
            on="group",
            how="inner",
        )
        ax = plot_df.set_index("group").plot(kind="bar", rot=30)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("Group-wise balanced accuracy: global vs calibrated threshold")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / "balanced_accuracy_by_group.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
