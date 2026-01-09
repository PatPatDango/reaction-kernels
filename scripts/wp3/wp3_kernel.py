# wp3_kernel.py

import numpy as np
from collections import Counter
from typing import Callable, Dict, List, Sequence, Any


def kernel_multiset_intersection(a: Counter[str], b: Counter[str]) -> int:
    """
    Multiset kernel: sum over features of min(count_a, count_b).
    """
    if not a or not b:
        return 0

    # iterate over smaller counter for speed
    if len(a) > len(b):
        a, b = b, a

    s = 0
    for k, ca in a.items():
        cb = b.get(k, 0)
        if cb:
            s += min(ca, cb)
    return s


def compute_kernel_matrix(
    X_feats: Sequence[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute an NxN kernel matrix K where K[i,j] = kernel_fn(X[i], X[j]).
    Uses symmetry to compute only upper triangle.
    """
    n = len(X_feats)
    K = np.zeros((n, n), dtype=dtype)

    for i in range(n):
        K[i, i] = kernel_fn(X_feats[i], X_feats[i])
        for j in range(i + 1, n):
            kij = kernel_fn(X_feats[i], X_feats[j])
            K[i, j] = kij
            K[j, i] = kij

    return K


def build_kernel_matrix_from_loaded(
    X_by_mode: Dict[str, List[Counter[str]]],
    y_by_mode: Dict[str, List[Any]] | None = None,
    *,
    mode: str = "edge",
    n: int | None = 200,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    dtype=np.float32,
) -> tuple[np.ndarray, List[Any] | None]:
    """
    Wrapper for already loaded dicts:
      X_drf = {"edge": [...], "vertex": [...], "sp": [...]}
      y_drf = {"edge": [...], "vertex": [...], "sp": [...]}
    """
    if mode not in X_by_mode:
        raise KeyError(f"mode='{mode}' not in X_by_mode. Available: {list(X_by_mode.keys())}")

    X = X_by_mode[mode]
    y = None

    if y_by_mode is not None:
        if mode not in y_by_mode:
            raise KeyError(f"mode='{mode}' not in y_by_mode. Available: {list(y_by_mode.keys())}")
        y = y_by_mode[mode]

    if n is not None:
        X = X[:n]
        if y is not None:
            y = y[:n]

    K = compute_kernel_matrix(X, kernel_fn=kernel_fn, dtype=dtype)
    return K, y


def kernel_matrix_stats(K: np.ndarray) -> Dict[str, float]:
    return {
        "n": float(K.shape[0]),
        "sym_max_abs": float(np.max(np.abs(K - K.T))),
        "diag_min": float(np.min(np.diag(K))),
        "diag_max": float(np.max(np.diag(K))),
        "nonzero_share": float((K > 0).mean()),
        "median": float(np.median(K)),
        "mean": float(K.mean()),
        "max": float(K.max()),
    }