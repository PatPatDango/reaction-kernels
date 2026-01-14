from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Optional, Dict
from collections import Counter
import hashlib
import networkx as nx
import numpy as np
from pathlib import Path
import pickle

def load_precomputed_features(
    precomp_dir: str | Path,
    *,
    feature_key: str,                 # "drf_wl" oder "its_wl"
    subset_ids: list[int] | None = None,
    class_key: str = "classes",       # bei dir heißt es "classes"
) -> tuple[list[Any], list[Any]]:
    """
    Like load_precomputed_features(...), but optionally loads only specific subset PKLs.

    subset_ids=[1,2,3] loads only files starting with:
      subset_001*.pkl, subset_002*.pkl, subset_003*.pkl
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob("subset_*.pkl"))

    if not files:
        raise FileNotFoundError(f"No PKL files found in {precomp_dir}")

    if subset_ids is not None:
        want = {f"subset_{i:03d}" for i in subset_ids}
        files = [fp for fp in files if any(fp.name.startswith(w) for w in want)]

    if not files:
        raise FileNotFoundError(
            f"No matching PKLs after filtering subset_ids={subset_ids} in {precomp_dir}"
        )

    X_all, y_all = [], []
    for fp in files:
        with fp.open("rb") as f:
            obj = pickle.load(f)

        if feature_key not in obj:
            raise KeyError(f"{fp.name}: missing key '{feature_key}'")
        if class_key not in obj or obj[class_key] is None:
            raise KeyError(f"{fp.name}: missing key '{class_key}'")

        X_all.extend(obj[feature_key])
        y_all.extend(obj[class_key])

    return X_all, y_all

def available_subset_ids(precomp_dir: str | Path) -> list[int]:
    """
    Returns all subset IDs that exist as PKL files in the folder.
    Example filename: subset_001.reaction_features_....pkl  -> id=1
    """
    precomp_dir = Path(precomp_dir)
    ids: list[int] = []
    for fp in precomp_dir.glob("subset_*.pkl"):
        try:
            sid = int(fp.name.split("subset_")[1][:3])
            ids.append(sid)
        except Exception:
            continue
    return sorted(set(ids))
