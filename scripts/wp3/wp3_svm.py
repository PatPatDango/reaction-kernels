# scripts/wp3_svm.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Iterable, Optional
import time
from collections import Counter
import plotly.express as px
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from pathlib import Path
import pickle
import pandas as pd

from scripts.wp3.tryout import class_counts

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
    f1_macro: float
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
    f1m = f1_score(y_test, y_pred, average="macro")

    if verbose:
        print(f"SVM (precomputed kernel) | n={len(y_train)+len(y_test)} | test_size={test_size} | C={C}")
        print("Accuracy:", acc)
        print("F1-macro:", f1m)
        print(classification_report(y_test, y_pred, zero_division=0))
    
    return SVMResult(
        acc=acc,            #Accuracy
        f1_macro=f1m,       #F1-macro
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
        f1_macro = None
        if isinstance(res, dict):
            f1_macro = res.get("f1_macro", None)
        else:
            f1_macro = getattr(res, "f1_macro", None)

        self.results.append({
            "tag": tag,
            "kernel": kernel,
            "mode": mode,
            "n": int(n),
            "test_size": float(test_size),
            "C": float(C),
            "seed": int(seed),
            "accuracy": float(acc),
            "f1_macro": float(f1_macro) if f1_macro is not None else None,
            "subset_ids": list(subset_ids) if subset_ids is not None else None,
            "k": int(k) if k is not None else None,
            **extra,
        })

    
def _detect_feature_key(d: Dict[str, Any]) -> str:
    for k in ("drf_wl", "its_wl", "features", "wl_features"):
        if k in d and isinstance(d[k], list):
            return k
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], (Counter, dict, set, list)):
            return k
    raise KeyError("Could not detect feature key in pickle.")

def _normalize_method(meta: Dict[str, Any], method_hint: Optional[str]) -> str:
    if method_hint:
        return method_hint.upper()
    t = (meta.get("type") or "").lower()
    if "drf" in t:
        return "DRF"
    if "its" in t:
        return "ITS"
    return "UNKNOWN"


def _drop_errors(
    features: List[Any],
    rsmi: List[str],
    classes: List[Any],
    errors: List[Dict[str, Any]],
) -> Tuple[List[Any], List[str], List[Any], int]:
    if not errors:
        return features, rsmi, classes, 0
    bad_ix = {e.get("index") for e in errors if "index" in e}
    keep = [i for i in range(len(features)) if i not in bad_ix]
    return (
        [features[i] for i in keep],
        [rsmi[i] for i in keep],
        [classes[i] for i in keep],
        len(bad_ix),
    )

def stratified_downsample(
    features: List[Any],
    y_raw: List[Any],
    rsmi: Optional[List[str]] = None,
    n_per_class: Optional[int] = None,
    random_state: int = 42,
    dedupe_rsmi: bool = True,
) -> Tuple[List[Any], List[Any], Optional[List[str]], Dict[Any, int]]:
    """
    Downsample to at most n_per_class items per class (stratified).
    Optionally deduplicate by rsmi per class.
    """
    if n_per_class is None:
        return features, y_raw, rsmi, class_counts(y_raw)

    df = pd.DataFrame({"idx": np.arange(len(y_raw)), "y": y_raw})
    if rsmi is not None:
        df["rsmi"] = rsmi

    rng = np.random.RandomState(random_state)
    sel_idx: List[int] = []

    for cls, grp in df.groupby("y"):
        if dedupe_rsmi and "rsmi" in df.columns:
            grp = grp.drop_duplicates(subset=["rsmi"], keep="first")
        take = min(n_per_class, len(grp))
        if take == 0:
            continue
        sel_idx.extend(rng.choice(grp["idx"].values, size=take, replace=False))

    sel_idx = sorted(sel_idx)
    f2 = [features[i] for i in sel_idx]
    y2 = [y_raw[i] for i in sel_idx]
    r2 = [rsmi[i] for i in sel_idx] if rsmi is not None else None
    return f2, y2, r2, class_counts(y2)

def load_pickles_from_dir(
    dir_path: str | Path,
    *,
    method_hint: Optional[str] = None,
    drop_failed: bool = True,
) -> List[Dict[str, Any]]:
    dir_path = Path(dir_path)
    out = []
    for pkl in sorted(dir_path.glob("*.pkl")):
        with open(pkl, "rb") as fh:
            d = pickle.load(fh)

        meta = d.get("meta", {})
        fkey = _detect_feature_key(d)
        features = d[fkey]
        rsmi = d.get("rsmi", [])
        y = d.get("classes", [])
        errors = d.get("errors", [])

        if drop_failed:
            features, rsmi, y, nerr = _drop_errors(features, rsmi, y, errors)
        else:
            nerr = len(errors or [])

        method = _normalize_method(meta, method_hint)

        mode = meta.get("mode")
        if not mode:
            name = pkl.stem
            if name.rsplit("_", 1)[-1] in {"edge", "vertex", "sp"}:
                mode = name.rsplit("_", 1)[-1]
            else:
                mode = "unknown"

        out.append(
            dict(
                features=features,
                y_raw=y,
                rsmi=rsmi,
                meta=meta,
                file=pkl,
                method=method,
                mode=mode,
                errors_dropped=nerr,
            )
        )
    return out


def compute_safe_test_size(
    y_raw: List[Any],
    requested: float,
    min_test_per_class: int = 1,
    min_train_per_class: int = 1,
) -> Tuple[Optional[float], Dict[str, Any]]:
    counts = class_counts(y_raw)
    if len(counts) < 2:
        return None, dict(reason="single_class", counts=counts)

    min_c = min(counts.values())
    if min_c < (min_test_per_class + min_train_per_class):
        return None, dict(reason="insufficient_samples_per_class", counts=counts)

    low = min_test_per_class / min_c
    high = 1 - (min_train_per_class / min_c)

    if low >= high:
        return None, dict(reason="no_feasible_split_interval", counts=counts, low=low, high=high)

    eps = 1e-9
    safe = requested
    if requested < low:
        safe = min(high - eps, low + eps)
    elif requested > high:
        safe = max(low + eps, high - eps)

    return safe, dict(reason="ok", counts=counts, low=low, high=high, requested=requested, adjusted=(safe != requested))


from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence

def _ensure_counters(
    features: Sequence[Any],
    to_counter_fn: Optional[Callable[[Any], Counter]] = None,
) -> List[Counter]:
    """
    Convert a sequence of feature objects to a list of collections.Counter.

    Behavior:
    - If an element is already a Counter -> keep as-is.
    - If to_counter_fn is provided, try it first.
    - Otherwise try `Counter(feature)` (works for mappings, iterables of tokens, lists).
    - Fall back to trying common conversion methods (feature.to_counter(), feature.as_counter(), feature.items()).
    - If conversion fails, raises TypeError describing the failing index/type.

    Returns:
      List[Counter] with same length and order as `features`.
    """
    out: List[Counter] = []
    for i, f in enumerate(features):
        # already good
        if isinstance(f, Counter):
            out.append(f)
            continue

        # user-provided converter
        if to_counter_fn is not None:
            try:
                c = to_counter_fn(f)
                if not isinstance(c, Counter):
                    c = Counter(c)
                out.append(c)
                continue
            except Exception:
                # fall through to other attempts
                pass

        # try direct Counter construction (works for mappings and iterables)
        try:
            c = Counter(f)
            out.append(c)
            continue
        except Exception:
            pass

        # try mapping-like `.items()`
        try:
            items = dict(f.items())  # will raise if no .items()
            out.append(Counter(items))
            continue
        except Exception:
            pass

        # try common custom methods
        for attr in ("to_counter", "as_counter", "as_dict"):
            fn = getattr(f, attr, None)
            if callable(fn):
                try:
                    cc = fn()
                    if not isinstance(cc, Counter):
                        cc = Counter(cc)
                    out.append(cc)
                    break
                except Exception:
                    continue
        else:
            # no break -> conversion failed
            raise TypeError(
                f"Cannot convert feature at index {i} (type {type(f)!r}) to collections.Counter. "
                "Provide a `to_counter_fn` or pass features in a standard form (Counter, mapping, or iterable of tokens)."
            )

    return out

def train_eval_svm_precomputed(
    features: List[Any],
    y_raw: List[Any],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
    kernel_fn = None,
) -> Dict[str, Any]:
    """
    Train / eval using precomputed kernel.
    Uses compute_kernel_matrix(...) to build K_train (square). Builds K_test manually
    (test x train) using kernel_fn. Optionally normalizes the kernel (cosine-like).
    Returns a dict compatible with run_split_sweep_all (keys: acc, f1_macro, timings, ...).
    """
    if kernel_fn is None:
        # fallback to the provided kernel function name
        try:
            kernel_fn = kernel_multiset_intersection
        except Exception:
            raise RuntimeError("No kernel_fn provided and kernel_multiset_intersection not importable")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    idx = np.arange(len(features))
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # prepare features as Counters once
    X_counters = _ensure_counters(features)
    X_train = [X_counters[i] for i in tr_idx]
    X_test  = [X_counters[i] for i in te_idx]

    timings: Dict[str, float] = {}

    # K_train: square matrix for training set
    t0 = time.perf_counter()
    # compute_kernel_matrix expects Counters and returns an NxN matrix
    K_train = compute_kernel_matrix(X_train, kernel_fn=kernel_fn)
    timings["k_train_time"] = time.perf_counter() - t0

    # compute diagonals for normalization if needed
    if kernel_normalize:
        diag_train = np.fromiter((kernel_fn(x, x) for x in X_train), dtype=np.float64, count=len(X_train))
        diag_test = np.fromiter((kernel_fn(x, x) for x in X_test), dtype=np.float64, count=len(X_test))
    else:
        diag_train = None
        diag_test = None

    # K_test: test x train
    t0 = time.perf_counter()
    K_test = np.zeros((len(X_test), len(X_train)), dtype=K_train.dtype)
    for i, xi in enumerate(X_test):
        for j, xj in enumerate(X_train):
            K_test[i, j] = kernel_fn(xi, xj)
    timings["k_test_time"] = time.perf_counter() - t0

    # optional normalization (cosine-like)
    if kernel_normalize:
        eps = 1e-12
        diag_train_clipped = np.clip(diag_train, eps, None)
        diag_test_clipped = np.clip(diag_test, eps, None)

        # normalize K_train in-place
        dprod_train = np.sqrt(diag_train_clipped[:, None] * diag_train_clipped[None, :])
        K_train = (K_train / dprod_train).astype(K_train.dtype, copy=False)

        # normalize K_test
        dprod_test = np.sqrt(diag_test_clipped[:, None] * diag_train_clipped[None, :])
        K_test = (K_test / dprod_test).astype(K_test.dtype, copy=False)

    # train SVM (precomputed)
    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
    t0 = time.perf_counter()
    clf.fit(K_train, y_train)
    timings["fit_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(K_test)
    timings["predict_time"] = time.perf_counter() - t0
    timings["total_time"] = sum(timings.values())

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    return dict(
        acc=acc,
        f1_macro=f1m,
        cm=cm,
        y_test=y_test,
        y_pred=y_pred,
        label_encoder=le,
        timings=timings,
        report_df=report_df,
        tr_idx=tr_idx,
        te_idx=te_idx,
        C=C,
        class_weight=class_weight,
        test_size=test_size,
        kernel_normalize=kernel_normalize,
        approach="kernel",
    )




    
def gather_capped_dataset(
    dirs: List[Tuple[Path, str]],
    *,
    method: str,           # "DRF" or "ITS"
    mode: Optional[str],   # "edge" | "vertex" | "sp" or None to accept mixed
    n_per_class: int,
    random_state: int = 42,
    dedupe_rsmi: bool = True,
    max_files_per_dir: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate across provided directories and cap to n_per_class per class.
    Returns a dict with features, y_raw, rsmi, class_counts, n_classes, n_samples, missing_classes.
    """
    all_features: List[Any] = []
    all_labels: List[Any] = []
    all_rsmi: List[str] = []

    for d, m in dirs:
        datasets = load_pickles_from_dir(d, method_hint=method, drop_failed=True)
        # Filter by mode if requested
        if mode:
            datasets = [ds for ds in datasets if ds["mode"] == mode]
        if max_files_per_dir is not None:
            datasets = datasets[:max_files_per_dir]

        for ds in datasets:
            all_features.extend(ds["features"])
            all_labels.extend(ds["y_raw"])
            all_rsmi.extend(ds["rsmi"])

    if not all_features:
        raise ValueError("No features found when aggregating. Check directories and mode.")

    # Downsample per class to n_per_class
    feats, y2, r2, counts2 = stratified_downsample(
        all_features, all_labels, all_rsmi, n_per_class=n_per_class, random_state=random_state, dedupe_rsmi=dedupe_rsmi
    )

    # Track missing classes (that didn't reach n_per_class)
    desired_classes = sorted(set(all_labels))
    have_classes = sorted(counts2.keys())
    missing = [c for c in desired_classes if c not in counts2]
    too_small = {c: class_counts(all_labels).get(c, 0) for c in desired_classes if counts2.get(c, 0) < n_per_class}

    return dict(
        features=feats,
        y_raw=y2,
        rsmi=r2,
        class_counts=counts2,
        n_classes=len(set(y2)),
        n_samples=len(y2),
        desired_classes=desired_classes,
        have_classes=have_classes,
        missing_classes=missing,
        classes_below_cap=too_small,
        method=method,
        mode=mode or "mixed",
        n_per_class=n_per_class,
    )

def run_split_sweep_all(
    drf_dirs: List[Tuple[Path, str]],
    its_dirs: List[Tuple[Path, str]],
    *,
    n_per_class: int,                 # z.B. 100 oder 200
    dataset_size_label: str,          # Beschriftung, z.B. "50×200"
    splits: Iterable[float],          # z.B. [0.1, 0.2, 0.3, 0.4]
    seeds: Iterable[int],             # z.B. range(5)
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
    use_linear_if_n_ge: int = 3000,
    l2_normalize_features: bool = True,
    max_files_per_dir: Optional[int] = None,
    verbose: bool = True,             # NEW: toggle progress logging
) -> pd.DataFrame:
    """
    Build an aggregated dataset per (method x mode) capped at n_per_class and sweep over
    test_size (splits) and random seeds. Automatically picks linear SVM for large n.
    Returns a DataFrame with all measurement points.

    Progress logging:
    - If verbose=True, prints per-run status, run time and ETA.
    """
    rows = []

    # Compute expected total tasks for ETA/progress
    n_method_dirs = len(drf_dirs) + len(its_dirs)
    n_splits = max(1, len(list(splits)))
    n_seeds = max(1, len(list(seeds)))
    total_tasks = n_method_dirs * n_splits * n_seeds

    def _format_time(sec: float) -> str:
        if sec is None or np.isnan(sec):
            return "N/A"
        sec = int(round(sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    start_all = time.perf_counter()
    completed = 0

    def process(method_dirs: List[Tuple[Path, str]], method_name: str):
        nonlocal completed, start_all
        for (d, mode) in method_dirs:
            agg = gather_capped_dataset(
                [(d, mode)],
                method=method_name,
                mode=mode,
                n_per_class=n_per_class,
                random_state=42,
                dedupe_rsmi=True,
                max_files_per_dir=max_files_per_dir,
            )

            total_n = agg["n_samples"]
            features = agg["features"]
            y_raw = agg["y_raw"]

            for ts in splits:
                for seed in seeds:
                    # safety: adjust test size if needed
                    safe_ts, _ = compute_safe_test_size(y_raw, requested=float(ts))

                    t0 = time.perf_counter()

                    
                    res = train_eval_svm_precomputed(
                        features=features,
                        y_raw=y_raw,
                        test_size=safe_ts if safe_ts is not None else float(ts),
                        random_state=seed,
                        C=C,
                        class_weight=class_weight,
                        kernel_normalize=kernel_normalize,
                    )
                    run_time = time.perf_counter() - t0

                    completed += 1
                    elapsed_all = time.perf_counter() - start_all
                    avg_per_task = elapsed_all / completed if completed else np.nan
                    remaining_tasks = max(0, total_tasks - completed)
                    eta_seconds = avg_per_task * remaining_tasks if not np.isnan(avg_per_task) else np.nan

                    rows.append(dict(
                        method=method_name,
                        mode=mode,
                        dataset_size=dataset_size_label,
                        n_samples=total_n,
                        n_classes=agg["n_classes"],
                        n_per_class=n_per_class,
                        split=float(ts),
                        seed=int(seed),
                        acc=res["acc"],
                        f1_macro=res["f1_macro"],
                        approach=res["approach"],
                        C=C,
                        class_weight=str(class_weight),
                        kernel_normalize=kernel_normalize,
                        l2_normalize_features=l2_normalize_features,
                        k_train_time=res["timings"].get("k_train_time", np.nan),
                        k_test_time=res["timings"].get("k_test_time", np.nan),
                        fit_time=res["timings"].get("fit_time", np.nan),
                        predict_time=res["timings"].get("predict_time", np.nan),
                        total_time=res["timings"]["total_time"],
                    ))

                    if verbose:
                        print(
                            f"[{completed}/{total_tasks}] {method_name}/{mode} split={ts} seed={seed} n={total_n} "
                            f"approach={res['approach']} acc={res['acc']:.3f} run={run_time:.2f}s "
                            f"elapsed={_format_time(elapsed_all)} ETA={_format_time(eta_seconds)}"
                        )

    process(drf_dirs, "DRF")
    process(its_dirs, "ITS")

    if verbose:
        total_elapsed = time.perf_counter() - start_all
        print(f"[DONE] Completed {completed}/{total_tasks} runs in {_format_time(total_elapsed)}")

    return pd.DataFrame(rows)
