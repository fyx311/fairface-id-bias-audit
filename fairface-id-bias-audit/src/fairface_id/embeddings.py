"""Face embedding extraction and scoring utilities.

The extraction function imports heavy deep-learning packages lazily so that the
synthetic demo and metric tests can run even on machines without PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors."""
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def score_pairs(pair_df: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    """Add cosine similarity scores to a pair DataFrame.

    Parameters
    ----------
    pair_df:
        DataFrame with path1, path2, label, and group columns.
    embeddings:
        Mapping from image path to embedding vector.
    """
    required = {"path1", "path2", "label", "group"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"pair_df is missing columns: {sorted(missing)}")

    records = []
    for row in pair_df.itertuples(index=False):
        p1 = str(getattr(row, "path1"))
        p2 = str(getattr(row, "path2"))
        if p1 not in embeddings or p2 not in embeddings:
            continue
        records.append({
            "path1": p1,
            "path2": p2,
            "label": int(getattr(row, "label")),
            "group": str(getattr(row, "group")),
            "score": cosine_similarity(embeddings[p1], embeddings[p2]),
        })
    if not records:
        raise ValueError("No scored pairs were produced. Check image paths and embedding keys.")
    return pd.DataFrame(records)


def extract_embeddings_from_paths(image_paths: Iterable[str | Path],
                                  output_npz: str | Path,
                                  device: str = "auto",
                                  batch_size: int = 32,
                                  image_size: int = 160) -> dict[str, np.ndarray]:
    """Extract FaceNet embeddings from image paths and save them as an NPZ file.

    This function uses MTCNN for face cropping and InceptionResnetV1 pretrained
    on VGGFace2 for 512-dimensional embeddings. It is intended for lawful,
    consented academic datasets only.
    """
    try:
        import torch
        from PIL import Image
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError as exc:
        raise ImportError(
            "Embedding extraction requires torch, Pillow, and facenet-pytorch. "
            "Install them with: pip install torch Pillow facenet-pytorch"
        ) from exc

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(image_size=image_size, margin=14, post_process=True, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    paths: list[str] = []
    tensors = []
    failed: list[str] = []

    for raw_path in image_paths:
        path = Path(raw_path)
        try:
            img = Image.open(path).convert("RGB")
            cropped = mtcnn(img)
            if cropped is None:
                failed.append(str(path))
                continue
            paths.append(str(path))
            tensors.append(cropped)
        except Exception:
            failed.append(str(path))

    if not tensors:
        raise ValueError("No faces were detected in the provided images.")

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    embedding_array = l2_normalize(np.vstack(embeddings))

    np.savez_compressed(output_npz, paths=np.array(paths, dtype=object), embeddings=embedding_array, failed=np.array(failed, dtype=object))
    return dict(zip(paths, embedding_array))
