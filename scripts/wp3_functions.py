# wp3_functions.py

from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple


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