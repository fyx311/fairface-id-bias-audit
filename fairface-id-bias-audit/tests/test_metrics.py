import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fairface_id.metrics import binary_metrics, fairness_summary, per_group_metrics
from fairface_id.thresholding import select_best_threshold


def test_binary_metrics_basic():
    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2]
    m = binary_metrics(y_true, scores, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["fpr"] == 0.0
    assert m["fnr"] == 0.0


def test_group_metrics_and_fairness_summary():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B", "A", "B"],
        "label": [1, 0, 1, 0, 1, 0],
        "score": [0.8, 0.2, 0.6, 0.4, 0.7, 0.5],
    })
    gm = per_group_metrics(df, threshold=0.5)
    fs = fairness_summary(gm)
    assert set(gm["group"]) == {"A", "B"}
    assert "accuracy" in set(fs["metric"])


def test_threshold_selection():
    df = pd.DataFrame({
        "label": [1, 1, 0, 0],
        "score": [0.9, 0.8, 0.3, 0.2],
        "group": ["A", "B", "A", "B"],
    })
    threshold, metrics = select_best_threshold(df, n_grid=25)
    assert 0.3 <= threshold <= 0.8
    assert metrics["balanced_accuracy"] >= 0.99
