"""Metrics for face-verification performance and fairness auditing."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def predict_from_threshold(scores: Iterable[float], threshold: float) -> np.ndarray:
    """Return 1 when score >= threshold, else 0."""
    return (np.asarray(list(scores), dtype=float) >= threshold).astype(int)


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, int]:
    """Compute TP, TN, FP, FN counts."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(num: float, den: float) -> float:
    """Divide and return NaN if denominator is zero."""
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> dict[str, float]:
    """Compute common binary verification metrics."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    y_pred = predict_from_threshold(y_score_arr, threshold)
    counts = confusion_counts(y_true_arr, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # same as TPR
    specificity = safe_div(tn, tn + fp)  # same as TNR
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (math.isnan(precision) or math.isnan(recall)) else float("nan")
    balanced_accuracy = np.nanmean([recall, specificity])

    result: dict[str, float] = {
        "n": float(len(y_true_arr)),
        "threshold": float(threshold),
        **{k: float(v) for k, v in counts.items()},
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy),
    }

    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true_arr)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        else:
            result["roc_auc"] = float("nan")
    except Exception:
        result["roc_auc"] = float("nan")
    return result


def per_group_metrics(df: pd.DataFrame,
                      threshold: float | None = None,
                      threshold_by_group: dict[str, float] | None = None) -> pd.DataFrame:
    """Compute metrics for each demographic group.

    Use either one global threshold or a dictionary of group-specific thresholds.
    """
    if threshold is None and threshold_by_group is None:
        raise ValueError("Provide either threshold or threshold_by_group.")
    rows = []
    for group, part in df.groupby("group", sort=True):
        group_threshold = threshold_by_group.get(group, threshold) if threshold_by_group else threshold
        metrics = binary_metrics(part["label"], part["score"], float(group_threshold))
        metrics["group"] = group
        rows.append(metrics)
    out = pd.DataFrame(rows)
    return out[["group"] + [c for c in out.columns if c != "group"]]


def overall_metrics(df: pd.DataFrame,
                    threshold: float | None = None,
                    threshold_by_group: dict[str, float] | None = None) -> dict[str, float]:
    """Compute overall metrics for a score DataFrame."""
    if threshold_by_group is not None:
        preds = []
        for row in df.itertuples(index=False):
            th = threshold_by_group.get(str(getattr(row, "group")), threshold if threshold is not None else 0.5)
            preds.append(1 if float(getattr(row, "score")) >= th else 0)
        counts = confusion_counts(df["label"], preds)
        # Reuse binary metrics on predicted labels by using y_pred as scores with threshold 0.5.
        # Counts are then inserted exactly from the group-threshold predictions.
        base = binary_metrics(df["label"], preds, 0.5)
        base.update({k: float(v) for k, v in counts.items()})
        base["threshold"] = float("nan")
        return base
    if threshold is None:
        raise ValueError("threshold is required when threshold_by_group is not provided.")
    return binary_metrics(df["label"], df["score"], threshold)


def fairness_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize max-min gaps across groups for key metrics."""
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "recall_tpr", "fpr", "fnr", "precision", "f1", "roc_auc"]:
        if metric not in group_metrics:
            continue
        series = group_metrics[metric].astype(float).dropna()
        if series.empty:
            continue
        best_group = group_metrics.loc[group_metrics[metric].astype(float).idxmax(), "group"]
        worst_group = group_metrics.loc[group_metrics[metric].astype(float).idxmin(), "group"]
        rows.append({
            "metric": metric,
            "min": float(series.min()),
            "max": float(series.max()),
            "gap_max_minus_min": float(series.max() - series.min()),
            "best_group": best_group,
            "worst_group": worst_group,
        })
    return pd.DataFrame(rows)
