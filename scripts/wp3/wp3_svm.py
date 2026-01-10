# scripts/wp3_svm.py
"""
WP3 — SVM training with a custom reaction kernel (multiset intersection)

Ziel:
- Wir haben für jede Reaktion bereits "Features" gespeichert.
- Diese Features sind KEINE normalen Zahlen-Vektoren, sondern Multisets:
    Counter(feature_hash -> count)

- Unser Kernel vergleicht zwei Reaktionen direkt, indem er zählt,
  wie viele Feature-Hashes sie gemeinsam haben (inkl. Häufigkeiten).
  => Multiset-Intersection

Warum "precomputed kernel"?
- sklearn.SVC kann entweder:
  (A) einen Standardkernel benutzen (rbf, poly, ...)
  (B) eine Kernel-Funktion benutzen, die mit Vektoren arbeitet
  (C) ODER wir geben eine fertige Kernel-Matrix K rein ("precomputed")

Da wir keine Vektoren haben, sondern Counter, ist (C) am saubersten:
- K_train ist (n_train x n_train)
- K_test  ist (n_test  x n_train)

Dann trainiert SVC direkt auf der Matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from collections import Counter

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import pickle
from .wp3_loader import load_precomputed_features_select


# ============================================================
# 1) Kernel: Multiset Intersection
# ============================================================

def kernel_multiset_intersection(a: Counter[str], b: Counter[str]) -> int:
    """
    Multiset intersection size:
    Für jedes Feature f, das in beiden vorkommt, zählt min(count_a, count_b).

    Beispiel:
      a = {x:3, y:1}
      b = {x:2, z:5}
      -> intersection = min(3,2)=2 (nur x überlappt)

    Rückgabewert ist ein int (Kernelwert).
    """
    if not a or not b:
        return 0

    # iteriere über die kleinere Counter-Map (Performance)
    if len(a) > len(b):
        a, b = b, a

    s = 0
    for k, ca in a.items():
        cb = b.get(k, 0)
        if cb:
            s += min(ca, cb)
    return s


# ============================================================
# 2) Kernel-Matrizen bauen (Train×Train und Test×Train)
# ============================================================

def build_kernel_matrix(
    XA: List[Counter[str]],
    XB: List[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
) -> np.ndarray:
    """
    Baut eine Kernel-Matrix K, wobei:
      K[i, j] = kernel_fn(XA[i], XB[j])

    Wenn XA == XB, ist K eine symmetrische Train×Train Kernel-Matrix.
    Wenn XA = Test und XB = Train, ist K die Test×Train Matrix.

    Achtung:
    - Für SVC(kernel="precomputed") muss:
        Train: (n_train x n_train)
        Test:  (n_test x n_train)
    """
    K = np.zeros((len(XA), len(XB)), dtype=np.float32)
    for i in range(len(XA)):
        ai = XA[i]
        for j in range(len(XB)):
            K[i, j] = kernel_fn(ai, XB[j])
    return K


def build_train_test_kernels(
    X: List[Counter[str]],
    y: List[Any],
    *,
    test_size: float = 0.2,
    seed: int = 42,
    n: int | None = None,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level helper:
    - optional: nimmt nur die ersten n Samples (für speed)
    - split in train/test (stratify = gleiche Klassenverteilung)
    - baut K_train und K_test

    Rückgabe:
      K_train, K_test, y_train, y_test
    """
    if n is not None:
        X = X[:n]
        y = y[:n]

    y_arr = np.asarray(y)

    # Wir splitten über Indizes, damit X und y sicher aligned bleiben
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_arr,  # wichtig: Klassenverteilung bleibt ähnlich
    )

    X_train = [X[i] for i in idx_train]
    X_test = [X[i] for i in idx_test]
    y_train = y_arr[idx_train]
    y_test = y_arr[idx_test]

    # K_train: train×train
    K_train = build_kernel_matrix(X_train, X_train, kernel_fn=kernel_fn)
    # K_test: test×train
    K_test = build_kernel_matrix(X_test, X_train, kernel_fn=kernel_fn)

    return K_train, K_test, y_train, y_test


# ============================================================
# 3) SVM trainieren und evaluieren
# ============================================================

@dataclass
class SVMResult:
    """
    Ergebnisobjekt, damit du später:
    - Accuracy
    - y_true / y_pred
    - Kernel-Matrizen
    benutzen kannst.
    """
    acc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    K_train: np.ndarray
    K_test: np.ndarray


def train_svm_with_precomputed_kernel(
    X: List[Counter[str]],
    y: List[Any],
    *,
    n: int | None = 600,
    test_size: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    verbose: bool = True,
) -> SVMResult:
    """
    Trainiert eine SVM mit precomputed kernel.

    Schritte:
    1) Split + Kernel matrices bauen
    2) SVC(kernel="precomputed") trainieren
    3) Vorhersage + Accuracy + Report

    Returns:
      SVMResult
    """
    K_train, K_test, y_train, y_test = build_train_test_kernels(
        X, y,
        n=n,
        test_size=test_size,
        seed=seed,
        kernel_fn=kernel_fn,
    )

    # SVC mit precomputed kernel: erwartet K_train statt X_train
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    acc = float(accuracy_score(y_test, y_pred))

    if verbose:
        print(f"SVM (precomputed kernel) | n={len(y_train)+len(y_test)} | test_size={test_size} | C={C}")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred, zero_division=0))

    return SVMResult(
        acc=acc,
        y_true=y_test,
        y_pred=y_pred,
        K_train=K_train,
        K_test=K_test,
    )


# ============================================================
# 4) Convenience: mehrere Modes testen (edge/vertex/sp)
# ============================================================

def run_svm_for_modes(
    X_by_mode: Dict[str, List[Counter[str]]],
    y_by_mode: Dict[str, List[Any]],
    *,
    modes: Tuple[str, ...] = ("edge", "vertex", "sp"),
    n: int | None = 600,
    test_size: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
    verbose_each: bool = False,
) -> Dict[str, SVMResult]:
    """
    Läuft über mehrere Feature-Typen (modes) und trainiert jeweils eine SVM.

    Returns:
      dict: mode -> SVMResult
    """
    out: Dict[str, SVMResult] = {}
    for mode in modes:
        res = train_svm_with_precomputed_kernel(
            X_by_mode[mode],
            y_by_mode[mode],
            n=n,
            test_size=test_size,
            seed=seed,
            C=C,
            verbose=verbose_each,
        )
        out[mode] = res
        print(f"mode={mode:6s} | acc={res.acc:.3f}")
    return out


# ============================================================
# 5) Optional: Confusion matrix plotly helper
# ============================================================

def confusion_matrix_plotly(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """
    Gibt eine Plotly Heatmap zurück.
    Bei vielen Klassen (z.B. 50) wird die Matrix groß – ist trotzdem hilfreich.
    """
    import plotly.express as px

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = px.imshow(cm, title=title, aspect="auto")
    return fig

# ============================================================
# 6) Load precomputed feature sets from disk
# ============================================================

def load_precomputed_features_from_dir(precomp_dir: str | Path, feature_key: str, class_key: str = "classes"):
    """
    Load and concatenate precomputed kernel features from multiple subset files.
    Lädt alle .pkl Dateien in einem Ordner und concatenatet sie.
    Erwartet Keys: feature_key (z.B. 'drf_wl' oder 'its_wl') und class_key ('classes').
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl found in {precomp_dir}")

    X_all, y_all = [], []
    for fp in files:
        with fp.open("rb") as f:
            obj = pickle.load(f)

        if feature_key not in obj:
            raise KeyError(f"{fp.name}: missing key '{feature_key}'. Keys={obj.keys()}")
        if class_key not in obj or obj[class_key] is None:
            raise KeyError(f"{fp.name}: missing key '{class_key}' (classes). Keys={obj.keys()}")

        X_all.extend(obj[feature_key])
        y_all.extend(obj[class_key])

    return X_all, y_all

# ============================================================
# 7) Dataset filtering: number of labels and samples per label
# ============================================================

def filter_dataset_by_classes_and_size(
    X, y,
    *,
    n_labels: int | None = None,
    samples_per_label: int | None = None,
    seed: int = 42,
):
    """
    Create controlled datasets by limiting the number of classes and samples per class.
    Macht aus (X,y) ein kleineres Dataset:
    - optional: nur n_labels Klassen behalten
    - optional: pro Klasse samples_per_label Beispiele ziehen
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)

    # unique labels
    labels = np.unique(y_arr)

    if n_labels is not None:
        if n_labels > len(labels):
            raise ValueError(f"Requested n_labels={n_labels} but only {len(labels)} available.")
        labels = rng.choice(labels, size=n_labels, replace=False)

    # indices pro label sammeln
    selected_idx = []
    for lab in labels:
        idx_lab = np.where(y_arr == lab)[0]
        if samples_per_label is not None:
            if len(idx_lab) < samples_per_label:
                # wenn zu wenig da: nimm alle (oder raise wenn du willst)
                take = idx_lab
            else:
                take = rng.choice(idx_lab, size=samples_per_label, replace=False)
        else:
            take = idx_lab
        selected_idx.extend(take.tolist())

    rng.shuffle(selected_idx)
    X_new = [X[i] for i in selected_idx]
    y_new = [y[i] for i in selected_idx]
    return X_new, y_new

# ============================================================
# 8) Experiment runner: kernel SVM over datasets and splits
# ============================================================

def run_experiments(
    *,
    # Welche Daten?
    precomp_dir: str | Path,
    representation: str,  # "DRF–WL" oder "ITS–WL" (nur für Anzeige)
    feature_key: str,     # "drf_wl" oder "its_wl"
    # Welche Kernel-Variante?
    mode: str,            # "edge" | "vertex" | "sp" (nur fürs Logging)
    # Dataset-Controls
    n_labels: int | None = None,
    samples_per_label: int | None = None,
    # Splits
    test_sizes: Tuple[float, ...] = (0.2, 0.3),
    # SVM params
    C: float = 1.0,
    seed: int = 42,
    # Speed limit (optional zusätzlich)
    n_max: int | None = None,
):
    """
    Run kernel SVM experiments across representations, dataset sizes, and train/test splits.
    Lädt Features aus Ordner, filtert ggf. auf n_labels & samples_per_label,
    und trainiert SVMs für mehrere test_sizes.
    Gibt eine Liste von Ergebnissen zurück.
    """
    # 1) laden
    X, y = load_precomputed_features_from_dir(precomp_dir, feature_key=feature_key)

    # optional: harte Obergrenze (falls du nur schnell testen willst)
    if n_max is not None:
        X = X[:n_max]
        y = y[:n_max]

    # 2) filtern (Klassenanzahl / Größe)
    Xf, yf = filter_dataset_by_classes_and_size(
        X, y, n_labels=n_labels, samples_per_label=samples_per_label, seed=seed
    )

    # 3) mehrere splits testen
    results = []
    for ts in test_sizes:
        res = train_svm_with_precomputed_kernel(
            Xf, yf,
            n=None,                 # wir haben schon gefiltert
            test_size=ts,
            seed=seed,
            C=C,
            verbose=False,
        )
        results.append({
            "representation": representation,
            "mode": mode,
            "precomp_dir": str(precomp_dir),
            "n_total": len(yf),
            "n_labels": len(set(yf)),
            "samples_per_label": samples_per_label,
            "test_size": ts,
            "C": C,
            "acc": res.acc,
        })
        print(f"{representation:7s} | {mode:6s} | labels={len(set(yf)):2d} | n={len(yf):4d} | test={ts:.1f} | acc={res.acc:.3f}")

    return results

# ============================================================
# 9) SVM training from precomputed kernel features of the subsets on disk
# ============================================================

def train_svm_from_precomputed_dir(
    *,
    precomp_dir: str | Path,
    feature_key: str,
    subset_ids: list[int] | None = None,  # z.B. [1,2,3]
    n: int = 600,
    test_size: float = 0.2,
    C: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Notebook-friendly wrapper:
    - lädt Features aus einem Ordner (optional nur bestimmte subsets)
    - trainiert SVM mit precomputed kernel
    """
    X, y = load_precomputed_features_select(
        precomp_dir,
        feature_key=feature_key,
        subset_ids=subset_ids,
    )

    return train_svm_with_precomputed_kernel(
        X, y,
        n=n,
        test_size=test_size,
        C=C,
        seed=seed,
        verbose=verbose,
    )
