from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize as sk_normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import plotly.express as px
import plotly.graph_objects as go

# external helpers (existing in repo)
from wp3_kernel import kernel_multiset_intersection, compute_kernel_matrix as wp3_compute_kernel_matrix, kernel_matrix_stats
from wp3_svm import build_train_test_kernels  # build_train_test_kernels returns K_train, K_test, y_train, y_test

# SciPy sparse optional
try:
    from scipy.sparse import csr_matrix
except Exception:
    csr_matrix = None


# -------------------------
# Loading precomputed sets (original loader kept)
# -------------------------

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


def load_pickles_from_dir(
    dir_path: str | Path,
    *,
    method_hint: Optional[str] = None,
    drop_failed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load pickled subset files and return list of dicts with keys:
      features, y_raw, rsmi, meta, file, method, mode, errors_dropped
    This preserves the original tryout.agreement with gather_capped_dataset.
    """
    dir_path = Path(dir_path)
    out = []
    for pkl in sorted(dir_path.glob("*.pkl")):
        with open(pkl, "rb") as fh:
            d = pd.read_pickle(fh) if False else __import__("pickle").load(fh)  # use pickle.load

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


# -------------------------
# Small helpers
# -------------------------

def class_counts(labels: List[Any]) -> Dict[Any, int]:
    return pd.Series(labels).value_counts().to_dict()


def compute_safe_test_size(
    y_raw: List[Any],
    requested: float,
    min_test_per_class: int = 1,
    min_train_per_class: int = 1,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Returns (safe_test_size, info) or (None, info) if splitting is impossible.
    Ensures at least min_test_per_class in test and min_train_per_class in train per class.
    """
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


# -------------------------
# Multiset kernel adapters
# -------------------------

def to_counter(x: Any) -> Counter:
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
    ca = to_counter(a)
    cb = to_counter(b)
    return kernel_multiset_intersection(ca, cb)


def compute_kernel_matrix(
    X: List[Any],
    Y: Optional[List[Any]] = None,
    *,
    normalize: bool = True,
    return_diagonals: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Adapter using wp3_kernel internals. Accepts lists of feature-like objects (Counters or lists/dicts).
    Returns (K, diag_X, diag_Y) if return_diagonals else (K, None, None).
    """
    Xc = [Counter(x) for x in X]
    if Y is None:
        # Use wp3's compute for symmetric case
        K = wp3_compute_kernel_matrix(Xc, kernel_fn=kernel_multiset_intersection)
        diag = np.diag(K).astype(float)
        if normalize:
            eps = 1e-12
            dprod = np.sqrt(np.clip(diag, eps, None)[:, None] * np.clip(diag, eps, None)[None, :])
            K = K / dprod
        if return_diagonals:
            return K, diag, diag
        return K, None, None
    else:
        Yc = [Counter(y) for y in Y]
        K = np.zeros((len(Xc), len(Yc)), dtype=np.float64)
        for i, xi in enumerate(Xc):
            for j, yj in enumerate(Yc):
                K[i, j] = kernel_multiset_intersection(xi, yj)
        if normalize:
            diag_X = np.array([kernel_multiset_intersection(xi, xi) for xi in Xc], dtype=float)
            diag_Y = np.array([kernel_multiset_intersection(yj, yj) for yj in Yc], dtype=float)
            eps = 1e-12
            dprod = np.sqrt(np.clip(diag_X, eps, None)[:, None] * np.clip(diag_Y, eps, None)[None, :])
            K = K / dprod
            if return_diagonals:
                return K, diag_X, diag_Y
        if return_diagonals:
            return K, None, None
        return K, None, None


# -------------------------
# Sparse features for linear SVM
# -------------------------

def features_to_sparse_matrix(features: List[Any]) -> Tuple["csr_matrix", List[str]]:
    """
    Convert list of Counter-like feature multisets into a sparse CSR matrix (rows=samples, cols=feature keys).
    """
    if csr_matrix is None:
        raise ImportError("scipy is required for sparse matrix conversion. Install scipy first.")

    vocab: Dict[str, int] = {}
    row_list: List[int] = []
    col_list: List[int] = []
    data_list: List[float] = []

    for i, f in enumerate(features):
        c = to_counter(f)
        for k, v in c.items():
            if k not in vocab:
                vocab[k] = len(vocab)
            j = vocab[k]
            row_list.append(i)
            col_list.append(j)
            data_list.append(float(v))

    X = csr_matrix((data_list, (row_list, col_list)), shape=(len(features), len(vocab)), dtype=np.float64)
    feature_names = [None] * len(vocab)
    for k, j in vocab.items():
        feature_names[j] = k
    return X, feature_names


# -------------------------
# Downsampling per class
# -------------------------

def stratified_downsample(
    features: List[Any],
    y_raw: List[Any],
    rsmi: Optional[List[str]] = None,
    n_per_class: Optional[int] = None,
    random_state: int = 42,
    dedupe_rsmi: bool = True,
) -> Tuple[List[Any], List[Any], Optional[List[str]], Dict[Any, int]]:
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


# -------------------------
# Aggregation across dirs/files to build capped dataset
# -------------------------

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
    Uses load_pickles_from_dir to preserve per-file meta information.
    """
    all_features: List[Any] = []
    all_labels: List[Any] = []
    all_rsmi: List[str] = []

    for d, m in dirs:
        datasets = load_pickles_from_dir(d, method_hint=method, drop_failed=True)
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

    feats, y2, r2, counts2 = stratified_downsample(
        all_features, all_labels, all_rsmi, n_per_class=n_per_class, random_state=random_state, dedupe_rsmi=dedupe_rsmi
    )

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


# -------------------------
# Adapter: precomputed-kernel SVM trainer (uses wp3 build helper)
# -------------------------

def train_eval_svm_precomputed(
    features: List[Any],
    y_raw: List[Any],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
) -> Dict[str, Any]:
    """
    Uses wp3_svm.build_train_test_kernels to build kernels (split+kernels) then trains SVC(kernel='precomputed').
    Returns results in a dict compatible with previous code.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    # build_train_test_kernels expects list[Counter], so convert
    Xc = [Counter(x) for x in features]
    K_train, K_test, y_train, y_test = build_train_test_kernels(
        Xc,
        list(y),
        test_size=test_size,
        seed=random_state,
        n=None,
        kernel_fn=kernel_multiset_intersection,
    )
    timings["k_train_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
    clf.fit(K_train, y_train)
    timings["fit_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(K_test)
    timings["predict_time"] = time.perf_counter() - t0
    timings["total_time"] = sum(timings.values())

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0)
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
        tr_idx=None,
        te_idx=None,
        C=C,
        class_weight=class_weight,
        test_size=test_size,
        kernel_normalize=kernel_normalize,
        approach="kernel",
    )


# -------------------------
# Linear SVM path (unchanged)
# -------------------------

def train_eval_linear_svm(
    features: List[Any],
    y_raw: List[Any],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    l2_normalize_features: bool = True,
) -> Dict[str, Any]:
    if csr_matrix is None:
        raise ImportError("scipy is required for the linear SVM path. Install scipy first.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_sparse, feature_names = features_to_sparse_matrix(features)

    if l2_normalize_features:
        X_sparse = sk_normalize(X_sparse, norm="l2", axis=1, copy=False)

    idx = np.arange(X_sparse.shape[0])
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = X_sparse[tr_idx, :]
    X_test = X_sparse[te_idx, :]

    timings: Dict[str, float] = {}
    clf = LinearSVC(C=C, class_weight=class_weight, dual=True)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    timings["fit_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    timings["predict_time"] = time.perf_counter() - t0
    timings["total_time"] = sum(timings.values())

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0)
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
        l2_normalize_features=l2_normalize_features,
        n_features=X_sparse.shape[1],
        approach="linear",
    )


# -------------------------
# Aggregated benchmark runner (unchanged API)
# -------------------------

def run_aggregated_benchmark(
    drf_dirs: List[Tuple[Path, str]],
    its_dirs: List[Tuple[Path, str]],
    *,
    n_per_class: int,
    dataset_size_label: str,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel_normalize: bool = True,
    use_linear_if_n_ge: int = 3000,
    l2_normalize_features: bool = True,
    max_files_per_dir: Optional[int] = None,
) -> pd.DataFrame:
    results = []

    def process(method_dirs: List[Tuple[Path, str]], method_name: str):
        for (d, mode) in method_dirs:
            agg = gather_capped_dataset(
                [(d, mode)],
                method=method_name,
                mode=mode,
                n_per_class=n_per_class,
                random_state=random_state,
                dedupe_rsmi=True,
                max_files_per_dir=max_files_per_dir,
            )
            total_n = agg["n_samples"]
            if total_n >= use_linear_if_n_ge:
                res = train_eval_linear_svm(
                    features=agg["features"],
                    y_raw=agg["y_raw"],
                    test_size=test_size,
                    random_state=random_state,
                    C=C,
                    class_weight=class_weight,
                    l2_normalize_features=l2_normalize_features,
                )
            else:
                res = train_eval_svm_precomputed(
                    features=agg["features"],
                    y_raw=agg["y_raw"],
                    test_size=test_size,
                    random_state=random_state,
                    C=C,
                    class_weight=class_weight,
                    kernel_normalize=kernel_normalize,
                )

            results.append(dict(
                method=method_name,
                mode=mode,
                dataset_size=dataset_size_label,
                n_samples=agg["n_samples"],
                n_classes=agg["n_classes"],
                n_per_class=n_per_class,
                acc=res["acc"],
                f1_macro=res["f1_macro"],
                approach=res["approach"],
                C=C,
                class_weight=str(class_weight),
                test_size=test_size,
                kernel_normalize=kernel_normalize,
                l2_normalize_features=l2_normalize_features,
                fit_time=res["timings"].get("fit_time", np.nan),
                predict_time=res["timings"].get("predict_time", np.nan),
                k_train_time=res["timings"].get("k_train_time", np.nan),
                k_test_time=res["timings"].get("k_test_time", np.nan),
                total_time=res["timings"]["total_time"],
            ))

    process(drf_dirs, "DRF")
    process(its_dirs, "ITS")

    return pd.DataFrame(results)


# -------------------------
# Pairwise and plotting helpers (unchanged)
# -------------------------

from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC as _SVC
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.metrics import accuracy_score as _acc_score, f1_score as _f1_score

def _subset_two_classes(
    features: List[Any],
    y_raw: List[Any],
    class_a: Any,
    class_b: Any,
    n_per_class: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[List[Any], List[Any]]:
    idx = [i for i, y in enumerate(y_raw) if y in (class_a, class_b)]
    f_sub = [features[i] for i in idx]
    y_sub = [y_raw[i] for i in idx]

    if n_per_class is not None:
        f_sub, y_sub, _, _ = stratified_downsample(
            f_sub, y_sub, rsmi=None, n_per_class=n_per_class, random_state=random_state, dedupe_rsmi=False
        )
    return f_sub, y_sub


def compute_pairwise_accuracy_matrix(
    features: List[Any],
    y_raw: List[Any],
    *,
    class_order: Optional[Sequence[Any]] = None,
    n_per_class: Optional[int] = 200,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    approach: str = "linear",
    kernel_normalize: bool = True,
    l2_normalize_features: bool = True,
) -> pd.DataFrame:
    classes = sorted(set(y_raw)) if class_order is None else list(class_order)
    rows = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ca, cb = classes[i], classes[j]
            f_pair, y_pair = _subset_two_classes(features, y_raw, ca, cb, n_per_class=n_per_class, random_state=random_state)
            cnt = class_counts(y_pair)
            if len(cnt) < 2 or min(cnt.values()) < 2:
                continue

            if approach == "linear":
                X_sparse, _ = features_to_sparse_matrix(f_pair)
                if l2_normalize_features:
                    X_sparse = sk_normalize(X_sparse, norm="l2", axis=1, copy=False)
                le = LabelEncoder()
                y_enc = le.fit_transform(y_pair)
                idx = np.arange(X_sparse.shape[0])
                tr, te, y_tr, y_te = train_test_split(idx, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)
                clf = LinearSVC(C=C, class_weight=None, dual=True)
                clf.fit(X_sparse[tr, :], y_tr)
                y_pred = clf.predict(X_sparse[te, :])

            elif approach == "kernel":
                le = LabelEncoder()
                y_enc = le.fit_transform(y_pair)
                idx = np.arange(len(f_pair))
                tr, te, y_tr, y_te = train_test_split(idx, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)
                K_train, _, _ = compute_kernel_matrix([f_pair[k] for k in tr], None, normalize=kernel_normalize)
                K_test, _, _ = compute_kernel_matrix([f_pair[k] for k in te], [f_pair[k] for k in tr], normalize=kernel_normalize)
                clf = SVC(kernel="precomputed", C=C)
                clf.fit(K_train, y_tr)
                y_pred = clf.predict(K_test)
            else:
                raise ValueError("approach must be 'linear' or 'kernel'.")

            acc = accuracy_score(y_te, y_pred)
            f1m = f1_score(y_te, y_pred, average="macro")
            rows.append(dict(
                class_a=ca, class_b=cb,
                acc=acc, f1_macro=f1m,
                n_a=cnt.get(ca, 0), n_b=cnt.get(cb, 0),
            ))

    return pd.DataFrame(rows)


def plot_pairwise_accuracy_heatmap(
    pair_df: pd.DataFrame,
    title: str = "Pairwise accuracy heatmap",
    class_order: Optional[List[Any]] = None,
    show_all_ticks: bool = True,
    height: int = 900,
    width: int = 1400,
    tick_font_size: int = 9,
    tick_angle: int = -45,
) -> go.Figure:
    if pair_df.empty:
        return go.Figure()

    classes = sorted(set(pair_df["class_a"]).union(pair_df["class_b"])) if class_order is None else list(class_order)
    mat = pd.DataFrame(np.nan, index=classes, columns=classes, dtype=float)

    for _, r in pair_df.iterrows():
        a, b, v = r["class_a"], r["class_b"], r["acc"]
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
            mat.loc[b, a] = v

    np.fill_diagonal(mat.values, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        colorscale="Viridis",
        zmin=0.0, zmax=1.0,
        hovertemplate="Class %{y} vs %{x}<br>ACC %{z:.3f}<extra></extra>",
    ))

    xaxis_cfg = dict(title="Class", automargin=True)
    yaxis_cfg = dict(title="Class", automargin=True)

    if show_all_ticks:
        xaxis_cfg.update(dict(
            type="category",
            categoryorder="array",
            categoryarray=classes,
            tickmode="array",
            tickvals=classes,
            ticktext=classes,
            tickangle=tick_angle,
            tickfont=dict(size=tick_font_size),
        ))
        yaxis_cfg.update(dict(
            type="category",
            categoryorder="array",
            categoryarray=classes,
            tickmode="array",
            tickvals=classes,
            ticktext=classes,
            tickfont=dict(size=tick_font_size),
        ))

    fig.update_layout(
        title=title,
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        height=height,
        width=width,
        margin=dict(l=60, r=20, t=60, b=120),
    )
    return fig


def plot_class_difficulty_bars(pair_df: pd.DataFrame, metric: str = "acc", title: str = "Per-class difficulty (mean pairwise ACC)") -> go.Figure:
    if pair_df.empty:
        return go.Figure()

    stats = defaultdict(list)
    for _, r in pair_df.iterrows():
        a, b, v = r[["class_a", "class_b", metric]]
        stats[a].append(v)
        stats[b].append(v)

    cls = []
    mean_v = []
    std_v = []
    for k, arr in stats.items():
        cls.append(k)
        mean_v.append(np.nanmean(arr))
        std_v.append(np.nanstd(arr))

    df = pd.DataFrame({"class": cls, "mean": mean_v, "std": std_v}).sort_values("mean", ascending=True)
    fig = px.bar(df, x="class", y="mean", error_y="std", title=title, labels={"mean": metric.upper()})
    fig.update_layout(height=600, xaxis_tickangle=-45)
    return fig


def fig2_style_pair_plot_kernelpca(
    features: List[Any],
    y_raw: List[Any],
    *,
    class_a: Any,
    class_b: Any,
    n_per_class: int = 200,
    random_state: int = 42,
    C_linear_2d: float = 1.0,
    kernel_normalize: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    f_pair, y_pair = _subset_two_classes(features, y_raw, class_a, class_b, n_per_class=n_per_class, random_state=random_state)
    if len(set(y_pair)) < 2:
        raise ValueError("Pair must contain both classes.")

    K, _, _ = compute_kernel_matrix(f_pair, None, normalize=kernel_normalize)
    kpca = KernelPCA(n_components=2, kernel="precomputed", random_state=random_state)
    Z = kpca.fit_transform(K)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_pair)
    tr, te, y_tr, y_te = train_test_split(np.arange(len(y_enc)), y_enc, test_size=0.2, random_state=random_state, stratify=y_enc)

    clf2d = SVC(kernel="linear", C=C_linear_2d)
    clf2d.fit(Z[tr, :], y_tr)

    x_min, x_max = Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5
    y_min, y_max = Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = clf2d.predict(grid).reshape(gx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 300),
        y=np.linspace(y_min, y_max, 300),
        z=zz,
        showscale=False,
        colorscale=[[0, "#0b5"], [1, "#fd0"]],
        opacity=0.30,
        contours_coloring="heatmap",
        hoverinfo="skip",
    ))

    colors = {0: "#0b5", 1: "#a5f"}
    names = {0: f"class {class_a}", 1: f"class {class_b}"}
    for lab in [0, 1]:
        sel = (y_enc == lab)
        fig.add_trace(go.Scatter(
            x=Z[sel, 0], y=Z[sel, 1],
            mode="markers",
            marker=dict(color=colors[lab], size=8),
            name=names[lab],
            hovertemplate=f"{names[lab]}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
        ))

    sv = clf2d.support_vectors_
    fig.add_trace(go.Scatter(
        x=sv[:, 0], y=sv[:, 1],
        mode="markers",
        marker=dict(color="#ffcc99", size=10, line=dict(color="#ff9900", width=2)),
        name="support vectors",
        hovertemplate="SV<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
    ))

    ttl = title or f"Fig-2 style (n={n_per_class}) | classes: {class_a} vs {class_b}"
    fig.update_layout(
        title=ttl,
        xaxis_title="KernelPCA-1",
        yaxis_title="KernelPCA-2",
        height=700,
        legend=dict(orientation="h"),
    )
    return fig


# -------------------------
# Split-sweep / plotting helpers (kept)
# -------------------------

def plot_split_metric_line(
    sweep_df: pd.DataFrame,
    metric: str = "acc",
    facet_by: Optional[str] = "dataset_size",
    color_by: str = "method",
    line_dash_by: str = "mode",
    title: Optional[str] = None,
) -> go.Figure:
    if sweep_df.empty:
        return go.Figure()

    group_cols = ["split", color_by, line_dash_by]
    if facet_by:
        group_cols.append(facet_by)

    agg = (
        sweep_df.groupby(group_cols)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("split")
    )

    fig = px.line(
        agg,
        x="split",
        y="mean",
        color=color_by,
        line_dash=line_dash_by,
        facet_col=(facet_by if facet_by else None),
        markers=True,
        title=title or f"{metric.upper()} vs Test split",
        labels={"mean": metric.upper(), "split": "Test split"},
    )
    for _, row in agg.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["split"], row["split"]],
            y=[row["mean"] - (row["std"] or 0), row["mean"] + (row["std"] or 0)],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)"),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_layout(height=500, legend_title_text=color_by.capitalize())
    return fig


def plot_split_runtime_area(
    sweep_df: pd.DataFrame,
    components: Tuple[str, ...] = ("k_train_time", "fit_time", "k_test_time", "predict_time"),
    facet_by: Optional[str] = "dataset_size",
    color_by: str = "method",
    line_dash_by: str = "mode",
    title: str = "Runtime components vs Test split",
) -> go.Figure:
    if sweep_df.empty:
        return go.Figure()

    base = sweep_df.copy()
    for comp in components:
        if comp not in base.columns:
            base[comp] = np.nan

    group_cols = ["split", color_by, line_dash_by]
    if facet_by:
        group_cols.append(facet_by)

    agg = base.groupby(group_cols).agg({k: "mean" for k in components}).reset_index().sort_values("split")
    for comp in components:
        agg[comp] = agg[comp].fillna(0.0)

    fig = go.Figure()
    palette = {
        "k_train_time": "#1f77b4",
        "fit_time": "#ff7f0e",
        "k_test_time": "#2ca02c",
        "predict_time": "#d62728",
    }

    for comp in components:
        for (meth, md) in sorted(set(zip(agg[color_by], agg[line_dash_by]))):
            sub = agg[(agg[color_by] == meth) & (agg[line_dash_by] == md)]
            name = f"{meth} | {md} | {comp}"
            fig.add_trace(go.Scatter(
                x=sub["split"], y=sub[comp],
                mode="lines+markers",
                name=name,
                line=dict(color=palette.get(comp, None), dash="solid"),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Test split",
        yaxis_title="Seconds (mean over seeds)",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_split_metric_box(
    sweep_df: pd.DataFrame,
    metric: str = "acc",
    facet_by: Optional[str] = "dataset_size",
    color_by: str = "method",
    box_group_by: str = "mode",
    title: Optional[str] = None,
) -> go.Figure:
    if sweep_df.empty:
        return go.Figure()

    fig = px.box(
        sweep_df,
        x="split",
        y=metric,
        color=color_by,
        facet_col=(facet_by if facet_by else None),
        points="all",
        title=title or f"{metric.upper()} distribution vs Test split",
        labels={metric: metric.upper(), "split": "Test split"},
    )
    fig.update_traces(boxmean=True)
    fig.update_layout(height=500, legend_title_text=color_by.capitalize())
    return fig