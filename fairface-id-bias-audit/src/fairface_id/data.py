"""Data loading helpers for face-verification fairness auditing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_SCORE_COLUMNS = {"label", "score", "group"}
REQUIRED_PAIR_COLUMNS = {"path1", "path2", "label", "group"}


def read_score_file(csv_path: str | Path,
                    label_col: str = "label",
                    score_col: str = "score",
                    group_col: str = "group") -> pd.DataFrame:
    """Read a score CSV and return standard columns: label, score, group.

    Parameters
    ----------
    csv_path:
        CSV file containing at least label, score, and group columns.
    label_col, score_col, group_col:
        Column names in the input file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns label, score, group.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Score file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = {label_col, score_col, group_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    out = df.rename(columns={label_col: "label", score_col: "score", group_col: "group"}).copy()
    out = out[["label", "score", "group"] + [c for c in out.columns if c not in {"label", "score", "group"}]]
    out["label"] = out["label"].astype(int)
    out["score"] = out["score"].astype(float)
    out["group"] = out["group"].astype(str)

    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0 and 1.")
    return out


def read_pair_file(csv_path: str | Path) -> pd.DataFrame:
    """Read a pair CSV with columns path1, path2, label, group."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pair file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_PAIR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required pair columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["group"] = df["group"].astype(str)
    return df


def load_embeddings_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load image embeddings saved by extract_embeddings_from_paths.

    The NPZ file is expected to include arrays named `paths` and `embeddings`.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "paths" not in data or "embeddings" not in data:
        raise ValueError("Embedding NPZ must contain arrays named 'paths' and 'embeddings'.")
    paths = [str(p) for p in data["paths"]]
    embeddings = np.asarray(data["embeddings"], dtype=float)
    return dict(zip(paths, embeddings))


def train_test_split_by_group(df: pd.DataFrame,
                              test_size: float = 0.4,
                              seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified-ish train/test split over group and label.

    This avoids the most common mistake in fairness auditing: evaluating a group
    with too few positive or negative pairs after splitting.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    dev_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, part in df.groupby(["group", "label"], sort=False):
        idx = np.array(part.index)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size))) if len(idx) > 1 else 0
        test_idx = idx[:n_test]
        dev_idx = idx[n_test:]
        if len(dev_idx) > 0:
            dev_parts.append(df.loc[dev_idx])
        if len(test_idx) > 0:
            test_parts.append(df.loc[test_idx])
    dev_df = pd.concat(dev_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return dev_df, test_df


def ensure_out_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def validate_groups(df: pd.DataFrame, min_examples_per_group: int = 10) -> None:
    """Raise a warning-like ValueError when group sizes are too small."""
    counts = df.groupby("group").size()
    small = counts[counts < min_examples_per_group]
    if not small.empty:
        raise ValueError(
            "Some groups are too small for a reliable audit: "
            + ", ".join(f"{g}={n}" for g, n in small.items())
        )


def iter_image_paths_from_pairs(pair_df: pd.DataFrame) -> Iterable[str]:
    """Yield unique image paths from a pair DataFrame."""
    seen: set[str] = set()
    for col in ("path1", "path2"):
        for value in pair_df[col].astype(str):
            if value not in seen:
                seen.add(value)
                yield value
