# scripts/wp3_svm.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from collections import Counter
import plotly.express as px
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import pickle

#handler funktionen
from .wp3_loader import load_precomputed_features
from .wp3_kernel import (
    kernel_multiset_intersection,
    compute_kernel_matrix,
)
@dataclass
class SVMResult:
    """
    Ergebnisobjekt, mit:Accuracy, y_true / y_pred, Kernel-Matrizen
    """
    acc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    K_train: np.ndarray
    K_test: np.ndarray

# ============================================================
# 1) Train/Test split + build Kernel matrices 
# ============================================================
def build_train_test_kernels(
    X: List[Counter[str]],
    y: List[Any],
    *,
    test_size: float = 0.2,
    seed: int = 42,
    n: int | None = None,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if n is not None:
        X = X[:n]
        y = y[:n]

    y_arr = np.asarray(y)

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_arr,
    )

    X_train = [X[i] for i in idx_train]
    X_test  = [X[i] for i in idx_test]
    y_train = y_arr[idx_train]
    y_test  = y_arr[idx_test]

    # ✅ train×train Kernel-Matrix (deine compute_kernel_matrix nimmt nur X)
    K_train = compute_kernel_matrix(X_train, kernel_fn=kernel_fn)

    # ✅ test×train Matrix manuell berechnen (weil compute_kernel_matrix kein Y hat)
    K_test = np.zeros((len(X_test), len(X_train)), dtype=K_train.dtype)
    for i, xi in enumerate(X_test):
        for j, xj in enumerate(X_train):
            K_test[i, j] = kernel_fn(xi, xj)

    return K_train, K_test, y_train, y_test

# ============================================================
# 2) SVM trainieren und evaluieren
# ============================================================
def train_svm_with_precomputed_kernel(
    X: List[Counter[str]],
    y: List[Any],
    *,
    n: int | None = 600,
    test_size: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    verbose: bool = True, #prints an oder aus 
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
    #training und test kernel matrices bauen
    K_train, K_test, y_train, y_test = build_train_test_kernels(
        X, y,
        n=n,
        test_size=test_size,
        seed=seed,
        kernel_fn=kernel_fn,
    )

    # SVC mit precomputed kernel: erwartet K_train statt X_train
    clf = SVC(kernel="precomputed", C=C)    #SVM erstellen C = Strenge / Bestrafung von Fehlern -> c = schwacher Wert = mehr Fehler erlaubt
    clf.fit(K_train, y_train)               #SVM mit Trainings-Kernel-Matrix und Labels trainieren -> Welche Trainingssamples gehören zu welcher Klasse – basierend auf Ähnlichkeiten

    y_pred = clf.predict(K_test)            #Vorhersagen für Testset machen basierend auf Test-Kernel-Matrix
    acc = float(accuracy_score(y_test, y_pred)) #Accuracy berechnen -> Anteil der korrekten Vorhersagen im Testset

    if verbose:
        print(f"SVM (precomputed kernel) | n={len(y_train)+len(y_test)} | test_size={test_size} | C={C}")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred, zero_division=0))
    
    return SVMResult(
        acc=acc,            #Accuracy
        y_true=y_test,      #echte Labels im Test
        y_pred=y_pred,      #vorhergesagte Labels
        K_train=K_train,    #Trainings-Kernel-Matrix    (für spätere Analysen)
        K_test=K_test,      #Test-Kernel-Matrix         (für spätere Analysen)
    )

# ============================================================
# 3) Convenience: mehrere Modes testen (edge/vertex/sp)
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
    Zum einstellen der Modes: edge / vertex / sp
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
# 4) Optional: Confusion matrix plotly helper
# ============================================================
def confusion_matrix_plotly(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """
    Gibt eine Plotly Heatmap zurück.
    y-true: wahre Klassen
    y-pred: vorhergesagte Klassen
    """
    labels = np.unique(np.concatenate([y_true, y_pred])) #alle labels zusammenfügen und doppelte entfernen
    cm = confusion_matrix(y_true, y_pred, labels=labels) #Confusion Matrix berechnen

    fig = px.imshow(cm, title=title, aspect="auto")     #Heatmap der Confusion Matrix erstellen
    return fig                                          #Rückgabe der Plotly-Figur

# ============================================================
# 5) Dataset filtering: create smaller datasets by class count and size
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
    - optional: Wie viele Reaktionen pro Klasse?
    """
    rng = np.random.default_rng(seed)   #Zufallszahlengenerator für Reproduzierbarkeit
    y_arr = np.asarray(y)               #y in numpy array konvertieren für einfaches Indexing    

    # unique labels
    labels = np.unique(y_arr)           #alle einzigartigen Klassenlabels finden

    if n_labels is not None:               #wenn n_labels gesetzt ist
        if n_labels > len(labels):         #prüfen ob n_labels nicht größer ist als die verfügbaren Klassen
            raise ValueError(f"Requested n_labels={n_labels} but only {len(labels)} available.")    #Fehlermeldung wenn zu viele Klassen angefragt wurde
        labels = rng.choice(labels, size=n_labels, replace=False) #zufällig n_labels Klassen auswählen ohne Ersatz

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
    return X_new, y_new     #Rückgabe des gefilterten Datasets

# ============================================================
# 6) Experiment runner: kernel SVM over datasets, splits and subsets
#   mit mehhreren Runs 
# ===========================================================
def run_SVM(
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
    subset_ids: list | None = None,
    # Speed limit (optional zusätzlich)
    n_max: int | None = None,
):
    """
    Run kernel SVM experiments across representations, dataset sizes, and train/test splits.
    Lädt Features aus Ordner, filtert ggf. auf n_labels & samples_per_label,
    und trainiert SVMs für mehrere test_sizes.
    Gibt eine Liste von Ergebnissen zurück.
    """
    # 1) laden -> hier nach dem MODE schauen
    #X, y = load_precomputed_features(precomp_dir, feature_key=feature_key)
    #path = Path(precomp_dir) / f"_{mode}"  #angepasster Pfad je nach Mode
    X, y = load_precomputed_features(
        precomp_dir,
        feature_key=feature_key,
        subset_ids=subset_ids,
    )

    # optional: harte Obergrenze für n
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
# 7) SVM runner: kernel SVM over datasets, splits and subsets
#   mit einem Run
# ===========================================================

def train_svm_from_datasets(
    *,
    precomp_dir,
    feature_key,
    subset_ids=None,
    n=600,
    test_size=0.2,
    C=1.0,
    seed=42,
    verbose=False,
):
    """
    Single-run helper (genau das was du in Sections 4-7 brauchst):
    - lädt Features aus precomp_dir (optional nur subset_ids)
    - nimmt max. n Samples
    - trainiert SVM mit precomputed kernel
    - gibt SVMResult zurück
    """
    X, y = load_precomputed_features(
        precomp_dir,
        feature_key=feature_key,
        subset_ids=subset_ids,
    )

    n_eff = min(n, len(y))
    if n_eff < 2:
        raise ValueError(
            f"Too few samples loaded: {n_eff}. "
            f"Check subset_ids/feature_key/dir."
        )

    return train_svm_with_precomputed_kernel(
        X, y,
        n=n_eff,
        test_size=test_size,
        seed=seed,
        C=C,
        verbose=verbose,
    )

# ============================================================
# 7) Results sichern nachdem SVM durchgelaufen ist
# ============================================================
@dataclass
class ResultLogger:
    """Sammelt Experiment-Ergebnisse (Notebook-friendly)."""
    results: list[dict[str, Any]] = field(default_factory=list)

    def add_result(
        self,
        tag: str,
        kernel: str,
        mode: str,
        n: int,
        test_size: float,
        C: float,
        seed: int,
        res: Any,
        subset_ids=None,
        k=None,
        **extra,
    ) -> None:
        # accuracy robust holen (SVMResult hat .acc)
        acc = None
        if isinstance(res, dict):
            acc = res.get("accuracy", None)
            if acc is None:
                acc = res.get("acc", None)
        else:
            acc = getattr(res, "accuracy", None)
            if acc is None:
                acc = getattr(res, "acc", None)

        if acc is None:
            raise ValueError(
                "Could not extract accuracy from res "
                "(need dict['accuracy'/'acc'] or .accuracy/.acc)."
            )

        self.results.append({
            "tag": tag,
            "kernel": kernel,
            "mode": mode,
            "n": int(n),
            "test_size": float(test_size),
            "C": float(C),
            "seed": int(seed),
            "accuracy": float(acc),
            "subset_ids": list(subset_ids) if subset_ids is not None else None,
            "k": int(k) if k is not None else None,
            **extra,
        })