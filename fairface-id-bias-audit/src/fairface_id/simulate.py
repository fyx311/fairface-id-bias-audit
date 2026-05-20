"""Synthetic data generator for demonstrating the audit pipeline.

The generated numbers are illustrative only. They do not describe any real
population. The purpose is to verify that the code can detect group-level
performance gaps and produce reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GROUPS = ["African", "Asian", "Caucasian", "Indian"]


def _clip_scores(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def simulate_pair_scores(n_per_group: int = 600,
                         groups: list[str] | None = None,
                         seed: int = 42) -> pd.DataFrame:
    """Create a synthetic pair-score dataset.

    Half the samples in each group are positive pairs and half are negative
    pairs. Positive-pair score distributions differ slightly by group so that a
    single global threshold can create a measurable TPR gap.
    """
    if groups is None:
        groups = DEFAULT_GROUPS
    if n_per_group < 20:
        raise ValueError("n_per_group should be at least 20 for stable demo metrics.")

    rng = np.random.default_rng(seed)
    records = []

    # Synthetic score parameters. These are not claims about real groups.
    positive_means = {
        "Caucasian": 0.78,
        "Asian": 0.74,
        "Indian": 0.71,
        "African": 0.68,
    }
    negative_means = {
        "Caucasian": 0.24,
        "Asian": 0.26,
        "Indian": 0.27,
        "African": 0.29,
    }

    for group in groups:
        n_pos = n_per_group // 2
        n_neg = n_per_group - n_pos
        pos_mean = positive_means.get(group, 0.72)
        neg_mean = negative_means.get(group, 0.27)
        pos_scores = _clip_scores(rng.normal(loc=pos_mean, scale=0.12, size=n_pos))
        neg_scores = _clip_scores(rng.normal(loc=neg_mean, scale=0.12, size=n_neg))

        for score in pos_scores:
            records.append({"group": group, "label": 1, "score": float(score)})
        for score in neg_scores:
            records.append({"group": group, "label": 0, "score": float(score)})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
