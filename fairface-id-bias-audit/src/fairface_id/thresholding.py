"""Threshold selection and calibration methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import binary_metrics


def candidate_thresholds(scores: np.ndarray, n_grid: int = 401) -> np.ndarray:
    """Create threshold candidates spanning observed scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores cannot be empty")
    lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
    if lo == hi:
        return np.array([lo], dtype=float)
    return np.linspace(lo, hi, n_grid)


def select_best_threshold(df: pd.DataFrame,
                          metric: str = "balanced_accuracy",
                          n_grid: int = 401) -> tuple[float, dict[str, float]]:
    """Select one threshold that maximizes the chosen metric."""
    best_threshold = None
    best_metrics = None
    best_value = -np.inf

    for threshold in candidate_thresholds(df["score"].to_numpy(), n_grid=n_grid):
        metrics = binary_metrics(df["label"], df["score"], float(threshold))
        value = metrics.get(metric, float("nan"))
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise ValueError(f"Could not select a valid threshold for metric={metric!r}.")
    return best_threshold, best_metrics


def select_group_thresholds(df: pd.DataFrame,
                            metric: str = "balanced_accuracy",
                            n_grid: int = 401,
                            min_group_size: int = 20) -> dict[str, float]:
    """Select a threshold independently for each group on validation data."""
    thresholds: dict[str, float] = {}
    global_threshold, _ = select_best_threshold(df, metric=metric, n_grid=n_grid)
    for group, part in df.groupby("group", sort=True):
        if len(part) < min_group_size or part["label"].nunique() < 2:
            thresholds[str(group)] = global_threshold
            continue
        threshold, _ = select_best_threshold(part, metric=metric, n_grid=n_grid)
        thresholds[str(group)] = threshold
    return thresholds


def select_fair_global_threshold(df: pd.DataFrame,
                                 performance_metric: str = "balanced_accuracy",
                                 fairness_metric: str = "recall_tpr",
                                 fairness_weight: float = 0.25,
                                 n_grid: int = 401) -> tuple[float, dict[str, float]]:
    """Select one threshold using a simple performance-minus-gap objective.

    Objective = overall performance - fairness_weight * max-min group gap.
    This is an interpretable baseline, not a complete fairness solution.
    """
    from .metrics import per_group_metrics

    best_threshold = None
    best_payload = None
    best_value = -np.inf

    for threshold in candidate_thresholds(df["score"].to_numpy(), n_grid=n_grid):
        overall = binary_metrics(df["label"], df["score"], float(threshold))
        group_df = per_group_metrics(df, threshold=float(threshold))
        group_values = group_df[fairness_metric].astype(float).dropna()
        group_gap = float(group_values.max() - group_values.min()) if not group_values.empty else 0.0
        objective = overall[performance_metric] - fairness_weight * group_gap
        if objective > best_value:
            best_value = objective
            best_threshold = float(threshold)
            best_payload = {**overall, "fairness_gap": group_gap, "objective": objective}

    if best_threshold is None or best_payload is None:
        raise ValueError("Could not select a fair global threshold.")
    return best_threshold, best_payload
