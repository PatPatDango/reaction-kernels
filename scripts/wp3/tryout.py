from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import plotly.express as px
import plotly.graph_objects as go


# -------------------------
# Loading precomputed sets
# -------------------------

def _detect_feature_key(d: Dict[str, Any]) -> str:
    """
    Find the list-like field that contains per-reaction feature multisets.
    Tries common keys used in precompute scripts.
    """
    for k in ("drf_wl", "its_wl", "features", "wl_features"):
        if k in d and isinstance(d[k], list):
            return k
    # Fallback: first list-valued key with Counters or dict-like items
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], (Counter, dict, set, list)):
            return k
    raise KeyError("Could not detect feature key in pickle.")


def _normalize_method(meta: Dict[str, Any], method_hint: Optional[str]) -> str:
    """
    Derive method name (DRF or ITS) from meta['type'] or provided hint.
    """
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
    """
    Removes rows that failed during precompute (if any).
    """
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


def load_pickles_from_dir(
    dir_path: str | Path,
    *,
    method_hint: Optional[str] = None,
    drop_failed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load all *.pkl files from a directory and return a list of dataset dicts:
    - features: List[Counter-like] (per reaction)
    - y_raw: List[label] (as in pickle)
    - rsmi: List[str]
    - meta: Dict
    - file: Path
    - method: "DRF" | "ITS" | "UNKNOWN"
    - mode: vertex | edge | sp (best-effort from meta or filename)
    - errors_dropped: int
    """
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
            # best effort: parse from filename "..._h{h}_{mode}.pkl"
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


# -------------------------
# Multiset kernel
# -------------------------

def to_counter(x: Any) -> Counter:
    """
    Normalize feature container to Counter.
    - Counter -> as is
    - dict    -> Counter(dict)
    - set     -> Counter({k: 1})
    - list    -> Counter(list)  (multiplicities preserved if present)
    """
    if isinstance(x, Counter):
        return x
    if isinstance(x, dict):
        return Counter(x)
    if isinstance(x, set):
        return Counter({k: 1 for k in x})
    if isinstance(x, list):
        return Counter(x)
    raise TypeError(f"Unsupported feature container type: {type(x)}")


def multiset_inner_product(a: Any, b: Any) -> int:
    """
    Inner product between two multisets (as Counters): sum over min counts on shared keys.
    This is the kernel value k(a, b).
    """
    ca = to_counter(a)
    cb = to_counter(b)
    # Iterate over smaller dict for speed
    if len(ca) > len(cb):
        ca, cb = cb, ca
    s = 0
    for k, va in ca.items():
        vb = cb.get(k)
        if vb:
            s += va if va < vb else vb
    return s


def compute_kernel_matrix(
    X: List[Any],
    Y: Optional[List[Any]] = None,
    *,
    normalize: bool = True,
    return_diagonals: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute Gram matrix K for lists of feature multisets X and Y.
    - If Y is None -> square matrix on X.
    - If normalize -> K_ij /= sqrt(K_xx_i * K_yy_j).

    Returns: (K, diag_X, diag_Y) if return_diagonals else (K, None, None).
    """
    if Y is None:
        Y = X

    n, m = len(X), len(Y)
    K = np.zeros((n, m), dtype=np.float64)

    # Precompute diagonals (self inner products) for normalization
    diag_X = np.fromiter((multiset_inner_product(x, x) for x in X), dtype=np.float64, count=n)
    diag_Y = np.fromiter((multiset_inner_product(y, y) for y in Y), dtype=np.float64, count=m)

    for i in range(n):
        xi = X[i]
        for j in range(m):
            K[i, j] = multiset_inner_product(xi, Y[j])

    if normalize:
        # Avoid division by zero: clip to tiny epsilon
        eps = 1e-12
        dprod = np.sqrt(np.clip(diag_X, eps, None)[:, None] * np.clip(diag_Y, eps, None)[None, :])
        K = K / dprod

    if return_diagonals:
        return K, diag_X, diag_Y
    return K, None, None


# -------------------------
# SVM training + evaluation
# -------------------------

def train_eval_svm_precomputed(
    features: List[Any],
    y_raw: List[Any],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,  # e.g., "balanced" or None
    kernel_normalize: bool = True,
) -> Dict[str, Any]:
    """
    Train/test split, compute K_train and K_test, train SVM (precomputed kernel),
    and evaluate. Returns metrics, timings, encoders, and confusion matrix.
    """
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    idx = np.arange(len(features))
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = [features[i] for i in tr_idx]
    X_test = [features[i] for i in te_idx]

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    K_train, _, _ = compute_kernel_matrix(X_train, None, normalize=kernel_normalize)
    timings["k_train_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    K_test, _, _ = compute_kernel_matrix(X_test, X_train, normalize=kernel_normalize)
    timings["k_test_time"] = time.perf_counter() - t0

    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
    t0 = time.perf_counter()
    clf.fit(K_train, y_train)
    timings["fit_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(K_test)
    timings["predict_time"] = time.perf_counter() - t0
    timings["total_time"] = sum(timings.values())

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

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
    )


# -------------------------
# Benchmark runners
# -------------------------

def run_benchmark_on_dir(
    dir_path: str | Path,
    *,
    method: str,  # "DRF" or "ITS"
    mode: str,    # "edge" | "vertex" | "sp"
    dataset_size: str,  # "small" | "big" (for labeling)
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Iterate all pickles in dir_path and run train/eval SVM. Returns:
    - results_df: one row per pickle, with metrics and timings
    - details: list of raw result dicts (per pickle) for further inspection
    """
    datasets = load_pickles_from_dir(dir_path, method_hint=method, drop_failed=True)
    rows = []
    details = []

    for ds in datasets:
        res = train_eval_svm_precomputed(
            features=ds["features"],
            y_raw=ds["y_raw"],
            test_size=test_size,
            random_state=random_state,
            C=C,
            class_weight=class_weight,
            kernel_normalize=kernel_normalize,
        )
        details.append(res)

        n_samples = len(ds["features"])
        n_classes = len(set(ds["y_raw"]))

        row = dict(
            file=str(ds["file"].name),
            path=str(ds["file"]),
            method=method,
            mode=mode,
            dataset_size=dataset_size,
            n_samples=n_samples,
            n_classes=n_classes,
            errors_dropped=ds["errors_dropped"],
            acc=res["acc"],
            f1_macro=res["f1_macro"],
            C=res["C"],
            class_weight=str(res["class_weight"]),
            test_size=res["test_size"],
            kernel_normalize=res["kernel_normalize"],
            k_train_time=res["timings"]["k_train_time"],
            k_test_time=res["timings"]["k_test_time"],
            fit_time=res["timings"]["fit_time"],
            predict_time=res["timings"]["predict_time"],
            total_time=res["timings"]["total_time"],
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df, details


def run_full_benchmark(
    drf_dirs: List[Tuple[Path, str]],
    its_dirs: List[Tuple[Path, str]],
    *,
    dataset_size: str,  # "small" or "big"
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
) -> pd.DataFrame:
    """
    Run benchmarks for DRF and ITS over provided (dir, mode) tuples.
    Returns aggregated results dataframe over all pickles.
    """
    all_results = []

    # DRF
    for (d, mode) in drf_dirs:
        df, _ = run_benchmark_on_dir(
            d,
            method="DRF",
            mode=mode,
            dataset_size=dataset_size,
            test_size=test_size,
            random_state=random_state,
            C=C,
            class_weight=class_weight,
            kernel_normalize=kernel_normalize,
        )
        all_results.append(df)

    # ITS
    for (d, mode) in its_dirs:
        df, _ = run_benchmark_on_dir(
            d,
            method="ITS",
            mode=mode,
            dataset_size=dataset_size,
            test_size=test_size,
            random_state=random_state,
            C=C,
            class_weight=class_weight,
            kernel_normalize=kernel_normalize,
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# -------------------------
# Plotly helpers
# -------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion matrix") -> go.Figure:
    """
    Heatmap of confusion matrix.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            hovertemplate="Predicted %{x}<br>True %{y}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=500,
    )
    return fig


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric: str = "acc",
    facet_by: Optional[str] = "dataset_size",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Grouped bar chart: metric by mode, colored by method (DRF/ITS).
    One bar per (method, mode, file); aggregated mean with error bars (std) per group.
    """
    # Aggregate to mean +/- std per method×mode×facet
    group_cols = ["method", "mode"]
    if facet_by:
        group_cols.append(facet_by)
    agg = (
        results_df.groupby(group_cols)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["title"] = title or f"{metric.upper()} by method/mode" + (f" ({facet_by})" if facet_by else "")

    fig = px.bar(
        agg,
        x="mode",
        y="mean",
        color="method",
        barmode="group",
        error_y="std",
        facet_col=(facet_by if facet_by else None),
        title=agg["title"].iloc[0],
        labels={"mean": metric.upper(), "mode": "Base kernel"},
    )
    fig.update_layout(height=500, legend_title_text="Method")
    return fig


def plot_runtime_bars(
    results_df: pd.DataFrame,
    stack_components: Tuple[str, ...] = ("k_train_time", "fit_time", "k_test_time", "predict_time"),
    facet_by: Optional[str] = "dataset_size",
    title: str = "Runtime breakdown (s) by method/mode",
) -> go.Figure:
    """
    Stacked bars per method×mode showing runtime components.
    """
    base = results_df.copy()
    base["group"] = base["method"] + " | " + base["mode"]
    agg = base.groupby(["group"] + ([facet_by] if facet_by else [])).agg({k: "mean" for k in stack_components}).reset_index()

    fig = go.Figure()
    for comp in stack_components:
        fig.add_trace(
            go.Bar(
                x=agg["group"] if not facet_by else agg["group"].astype(str) + " (" + agg[facet_by].astype(str) + ")",
                y=agg[comp],
                name=comp,
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Method | Mode" + (f" ({facet_by})" if facet_by else ""),
        yaxis_title="Seconds",
        height=500,
    )
    return fig