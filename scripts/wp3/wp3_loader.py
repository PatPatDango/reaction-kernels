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
    feature_key: str,          # "drf_wl" oder "its_wl"
    class_key: str = "classes" # bei dir: rxn_class
) -> Tuple[List[Dict[str, int]], List[Any]]:
    """
    Loads precomputed reaction features from a directory of .pkl files.

    Returns:
        X : list of feature Counters (hash -> count)
        y : list of class labels (rxn_class)
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(f"No .pkl files found in {precomp_dir}")

    X_all: List[Dict[str, int]] = []
    y_all: List[Any] = []

    for fp in files:
        with fp.open("rb") as f:
            obj = pickle.load(f)

        if feature_key not in obj:
            raise KeyError(f"{fp.name}: missing key '{feature_key}'")

        if class_key not in obj:
            raise KeyError(f"{fp.name}: missing key '{class_key}'")

        X = obj[feature_key]
        y = obj[class_key]

        if len(X) != len(y):
            raise ValueError(f"{fp.name}: feature/label length mismatch")

        X_all.extend(X)
        y_all.extend(y)

    return X_all, y_all

def load_precomputed_features_select(
    precomp_dir: str | Path,
    *,
    feature_key: str,
    class_key: str = "classes",
    subset_ids: list[int] | None = None,
    pattern: str = "*.pkl",
):
    """
    Load precomputed features from a directory.
    Optionally restrict to specific subset IDs (e.g. [1,2,3] -> subset_001, subset_002, subset_003).
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PKL files found in {precomp_dir}")

    if subset_ids is not None:
        want = {f"subset_{i:03d}" for i in subset_ids}
        files = [fp for fp in files if any(tag in fp.name for tag in want)]

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

def subset_class_profile(pkl_path: Path, class_key="classes"):
    """Return (subset_id, Counter(class->count)) for one PKL."""
    obj = pickle.load(open(pkl_path, "rb"))
    classes = obj[class_key]
    return Counter(classes)

def choose_subsets_with_fixed_classes(index, target_classes, min_per_class=20):
    """
    Pick subset_ids where ALL target_classes are present with at least min_per_class.
    """
    target_classes = set(target_classes)
    good = []
    for sid, cnt in index.items():
        if all(cnt.get(c, 0) >= min_per_class for c in target_classes):
            good.append(sid)
    return sorted(good)

def choose_most_balanced_subsets(index, k=5):
    """
    Pick k subset_ids with the most balanced distribution.
    Balance score = max(counts)-min(counts) (smaller is better).
    """
    scored = []
    for sid, cnt in index.items():
        if len(cnt) == 0:
            continue
        vals = list(cnt.values())
        score = max(vals) - min(vals)
        scored.append((score, sid))
    scored.sort()
    return [sid for _, sid in scored[:k]]

def build_subset_index(precomp_dir: str | Path, pattern="subset_*.pkl", class_key="classes"):
    precomp_dir = Path(precomp_dir)
    index = {}
    for fp in sorted(precomp_dir.glob(pattern)):
        sid = int(fp.name.split("subset_")[1][:3])
        obj = pickle.load(open(fp, "rb"))
        index[sid] = Counter(obj[class_key])
    return index

def choose_subsets_with_at_least_k_common_classes(index, ref_classes, k=2, min_per_class=20):
    ref = set(ref_classes)
    good = []
    for sid, cnt in index.items():
        present = {c for c, n in cnt.items() if n >= min_per_class}
        if len(present & ref) >= k:
            good.append(sid)
    return sorted(good)

def common_classes_across_subsets(index, subset_ids, min_per_class=20):
    common = None
    for sid in subset_ids:
        cnt = index[sid]
        present = {c for c, n in cnt.items() if n >= min_per_class}
        common = present if common is None else (common & present)
    return sorted(common) if common is not None else []