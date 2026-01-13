from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from collections import Counter
import pickle
import re

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

log = logging.getLogger(__name__)

# =========================
# Loader
# =========================

_POSSIBLE_CLASS_KEYS = ("classes", "rxn_class", "labels")


def _detect_class_key(obj: dict) -> str:
    for k in _POSSIBLE_CLASS_KEYS:
        if k in obj and obj[k] is not None:
            return k
    raise KeyError(f"Missing any class key {_POSSIBLE_CLASS_KEYS}. Keys: {list(obj.keys())}")


def load_precomputed_features(
    precomp_dir: str | Path,
    *,
    feature_key: str,
    subset_ids: List[int] | None = None,
    pattern: str = "*.pkl",
) -> Tuple[List[Counter[str]], List[Any]]:
    """
    Lädt vorcomputierte Features und Labels aus einem Ordner.
    - feature_key: z.B. 'drf_wl' oder 'its_wl'
    - subset_ids: optional, lädt nur subset_XXX Dateien
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob(pattern))
    if subset_ids is not None:
        wanted = {f"subset_{i:03d}" for i in subset_ids}
        files = [fp for fp in files if any(tag in fp.name for tag in wanted)]
    if not files:
        raise FileNotFoundError(f"No PKL files in {precomp_dir} (pattern={pattern}, subset_ids={subset_ids})")

    X_all: List[Counter[str]] = []
    y_all: List[Any] = []
    for fp in files:
        try:
            with fp.open("rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            log.warning("Skipping unreadable file: %s (%s)", fp, e)
            continue

        if feature_key not in obj:
            raise KeyError(f"{fp.name}: missing feature '{feature_key}'. Keys={list(obj.keys())}")
        class_key = _detect_class_key(obj)

        X_all.extend(obj[feature_key])
        y_all.extend(obj[class_key])

    if len(X_all) != len(y_all):
        raise ValueError(f"Feature/label length mismatch: X={len(X_all)} y={len(y_all)}")

    return X_all, y_all


def filter_empty(X: List[Counter[str]], y: List[Any]) -> Tuple[List[Counter[str]], List[Any]]:
    """Entfernt leere Counter aus X und die zugehörigen y."""
    keep = [i for i, c in enumerate(X) if len(c) > 0]
    if len(keep) == len(X):
        return X, y
    Xf = [X[i] for i in keep]
    yf = [y[i] for i in keep]
    removed = len(X) - len(Xf)
    log.info("Filtered %d empty samples (kept %d/%d).", removed, len(Xf), len(X))
    return Xf, yf


# =========================
# Kernel
# =========================

def kernel_multiset_intersection(a: Counter[str], b: Counter[str]) -> int:
    """
    Multiset-Intersection: Summe über min(count_a, count_b) für gemeinsame Features.
    """
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    s = 0
    for k, ca in a.items():
        cb = b.get(k, 0)
        if cb:
            s += ca if ca <= cb else cb
    return s


def build_kernel_matrix(
    XA: Sequence[Counter[str]],
    XB: Sequence[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    dtype=np.float32,
) -> np.ndarray:
    """
    Rechteckige Kernel-Matrix K mit K[i,j] = kernel_fn(XA[i], XB[j]).
    """
    K = np.zeros((len(XA), len(XB)), dtype=dtype)
    for i, a in enumerate(XA):
        for j, b in enumerate(XB):
            K[i, j] = kernel_fn(a, b)
    return K


def compute_kernel_matrix(
    X: Sequence[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    dtype=np.float32,
) -> np.ndarray:
    """
    Quadratische symmetrische Kernel-Matrix K mit K[i,j] = kernel_fn(X[i], X[j]).
    """
    n = len(X)
    K = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        kii = kernel_fn(X[i], X[i])
        K[i, i] = kii
        for j in range(i + 1, n):
            kij = kernel_fn(X[i], X[j])
            K[i, j] = kij
            K[j, i] = kij
    return K


def normalize_kernel_inplace(K: np.ndarray, eps: float = 1e-12) -> None:
    """
    Normiert K nach K_ij / sqrt(K_ii*K_jj).
    Modifiziert K in-place.
    """
    diag = np.diag(K).copy()
    diag = np.clip(diag, eps, None)
    s = 1.0 / np.sqrt(diag)
    K *= s[:, None]
    K *= s[None, :]


def normalize_test_kernel_inplace(
    K_test: np.ndarray,
    diag_test: np.ndarray,
    diag_train: np.ndarray,
    eps: float = 1e-12,
) -> None:
    """
    Normiert Test×Train-Kernel:
      K_test[i,j] /= sqrt(diag_test[i] * diag_train[j])
    """
    dt = np.clip(diag_test, eps, None)
    dtr = np.clip(diag_train, eps, None)
    s_test = 1.0 / np.sqrt(dt)
    s_train = 1.0 / np.sqrt(dtr)
    K_test *= s_test[:, None]
    K_test *= s_train[None, :]


def kernel_matrix_stats(K: np.ndarray) -> Dict[str, float]:
    sym = float(np.max(np.abs(K - K.T))) if K.shape[0] == K.shape[1] else float("nan")
    return {
        "n_rows": float(K.shape[0]),
        "n_cols": float(K.shape[1]),
        "sym_max_abs": sym,
        "diag_min": float(np.min(np.diag(K))) if K.shape[0] == K.shape[1] else float("nan"),
        "diag_max": float(np.max(np.diag(K))) if K.shape[0] == K.shape[1] else float("nan"),
        "share_nonzero": float((K > 0).mean()),
        "median": float(np.median(K)),
        "mean": float(K.mean()),
        "max": float(K.max()),
    }


def find_first_nonzero_pair(
    X: Sequence[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    min_overlap: int = 1,
) -> Tuple[int, int, int] | None:
    """
    Findet das erste Paar (i,j) mit kernel(X[i],X[j]) >= min_overlap.
    """
    for i, a in enumerate(X):
        if not a:
            continue
        for j in range(i + 1, len(X)):
            b = X[j]
            if not b:
                continue
            k = kernel_fn(a, b)
            if k >= min_overlap:
                return i, j, k
    return None


# =========================
# Train/Test-Splitting und SVM
# =========================

@dataclass
class SVMResult:
    acc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    K_train: np.ndarray
    K_test: np.ndarray


def build_train_test_kernels(
    X: List[Counter[str]],
    y: List[Any],
    *,
    n: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Erzeugt:
      - K_train (train×train), K_test (test×train)
      - y_train, y_test
      - diag_train, diag_test (Selbst-Kernel der jeweiligen Elemente)
    """
    if n is not None:
        X = X[:n]
        y = y[:n]

    # Leere entfernen für Stabilität
    X, y = filter_empty(X, y)

    y_arr = np.asarray(y)
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y_arr
    )

    X_train = [X[i] for i in idx_train]
    X_test = [X[i] for i in idx_test]
    y_train = y_arr[idx_train]
    y_test = y_arr[idx_test]

    # Kernelmatrizen bauen
    K_train = compute_kernel_matrix(X_train, kernel_fn=kernel_fn, dtype=np.float32)
    K_test = build_kernel_matrix(X_test, X_train, kernel_fn=kernel_fn, dtype=np.float32)

    # Diagonalen (für Normalisierung)
    diag_train = np.diag(K_train).copy()
    diag_test = np.array([kernel_fn(x, x) for x in X_test], dtype=np.float32)

    return K_train, K_test, y_train, y_test, diag_train, diag_test


def train_svm_with_precomputed_kernel(
    X: List[Counter[str]],
    y: List[Any],
    *,
    n: int | None = 600,
    test_size: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    normalize: bool = True,
    verbose: bool = True,
) -> SVMResult:
    """
    Trainiert eine SVC(kernel='precomputed') auf unserem Multiset-Kernel.
    """
    K_train, K_test, y_train, y_test, diag_train, diag_test = build_train_test_kernels(
        X, y, n=n, test_size=test_size, seed=seed, kernel_fn=kernel_fn
    )

    if normalize:
        normalize_kernel_inplace(K_train)
        normalize_test_kernel_inplace(K_test, diag_test, diag_train)

    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    acc = float(accuracy_score(y_test, y_pred))

    if verbose:
        log.info("SVM | n=%d | test_size=%.2f | C=%.3f | acc=%.3f", len(y_train) + len(y_test), test_size, C, acc)

    return SVMResult(acc=acc, y_true=y_test, y_pred=y_pred, K_train=K_train, K_test=K_test)


# =========================
# Subset-Utilities
# =========================

_subset_re = re.compile(r"subset_(\d+)")


def parse_subset_id(name: str) -> int | None:
    m = _subset_re.search(name)
    return int(m.group(1)) if m else None


def available_subset_ids(precomp_dir: str | Path) -> List[int]:
    p = Path(precomp_dir)
    ids = []
    for fp in p.glob("subset_*.pkl"):
        sid = parse_subset_id(fp.name)
        if sid is not None:
            ids.append(sid)
    return sorted(set(ids))


def build_subset_index(precomp_dir: str | Path, class_key_candidates=_POSSIBLE_CLASS_KEYS) -> Dict[int, Counter[Any]]:
    """
    Für jeden subset_XXX.pkl -> Counter(Klasse->Anzahl).
    """
    precomp_dir = Path(precomp_dir)
    index: Dict[int, Counter[Any]] = {}
    for fp in sorted(precomp_dir.glob("subset_*.pkl")):
        sid = parse_subset_id(fp.name)
        if sid is None:
            continue
        with fp.open("rb") as f:
            obj = pickle.load(f)
        # Klassen-Key detektieren
        ck = None
        for k in class_key_candidates:
            if k in obj and obj[k] is not None:
                ck = k
                break
        if ck is None:
            log.warning("No class key in %s", fp)
            continue
        index[sid] = Counter(obj[ck])
    return index


def choose_subsets_with_at_least_k_common_classes(
    index: Dict[int, Counter[Any]],
    ref_classes: Iterable[Any],
    k: int = 2,
    min_per_class: int = 20,
) -> List[int]:
    ref = set(ref_classes)
    good = []
    for sid, cnt in index.items():
        present = {c for c, n in cnt.items() if n >= min_per_class}
        if len(present & ref) >= k:
            good.append(sid)
    return sorted(good)


def choose_most_balanced_subsets(index: Dict[int, Counter[Any]], k: int = 5) -> List[int]:
    scored: List[Tuple[int, int]] = []
    for sid, cnt in index.items():
        if not cnt:
            continue
        vals = list(cnt.values())
        scored.append((max(vals) - min(vals), sid))
    scored.sort()
    return [sid for _, sid in scored[:k]]


def common_classes_across_subsets(index: Dict[int, Counter[Any]], subset_ids: List[int], min_per_class: int = 20) -> List[Any]:
    common = None
    for sid in subset_ids:
        cnt = index[sid]
        present = {c for c, n in cnt.items() if n >= min_per_class}
        common = present if common is None else (common & present)
    return sorted(common) if common is not None else []


# =========================
# Results-Collector und Auswertung
# =========================

class ResultsCollector:
    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        tag: str,
        representation: str,
        mode: str,
        n: int,
        test_size: float,
        C: float,
        seed: int,
        acc: float,
        subset_ids: List[int] | None = None,
        precomp_dir: str | None = None,
        feature_key: str | None = None,
    ) -> None:
        self._rows.append(
            {
                "tag": tag,
                "representation": representation,
                "mode": mode,
                "n": n,
                "test_size": test_size,
                "C": C,
                "seed": seed,
                "accuracy": acc,
                "subset_ids": list(subset_ids) if subset_ids is not None else None,
                "precomp_dir": precomp_dir,
                "feature_key": feature_key,
            }
        )

    def to_frame(self):
        try:
            import pandas as pd  # lazy import
        except Exception as e:
            raise RuntimeError("pandas is required for to_frame().") from e
        return pd.DataFrame(self._rows)

    def save_csv(self, path: str | Path) -> None:
        df = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


# =========================
# Optional: Confusion-Matrix Plot
# =========================

def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """
    Gibt eine Plotly-Figur zurück, falls plotly verfügbar ist; sonst None.
    """
    try:
        import plotly.express as px
    except Exception:
        log.warning("plotly not installed; skipping confusion matrix plot.")
        return None

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, title=title, aspect="auto")
    return fig