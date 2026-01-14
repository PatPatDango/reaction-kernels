from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display as ipy_display  # <-- Neu: explizit importieren

# Local imports (wie im Notebook)
from scripts.wp3.wp3_loader import (
    load_precomputed_features,
    available_subset_ids,
    load_precomputed_features_select,
)
from scripts.wp3.wp3_kernel import (
    build_kernel_matrix_from_loaded,
    kernel_matrix_stats,
    kernel_multiset_intersection,
)
from scripts.wp3.wp3_svm import train_svm_from_precomputed_dir, train_svm_with_precomputed_kernel
from scripts.wp3.wp3_plots import (
    fig2_style_svm_from_kernel,
    plot_experiment_dashboard,
    plot_heatmaps_by_k,
    plot_difference_heatmap,
    plot_slope_drf_vs_its,
    plot_drf_minus_its_bar,
    plot_drf_vs_its_dots,
    plot_accuracy_by_k,
)
from scripts.wp3.wp3_subset_handler_new import (
    make_option1_soft_shared_k_classes_config,
    print_option1_config,
    sanitize_subset_ids,
     make_option_quota_all_classes_config,
)

# -------------------------------
# Repo/Verzeichnis-Helfer
# -------------------------------
def find_repo_root() -> Tuple[Path, Path]:
    root = next(
        (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir() or (p / "scripts").is_dir()),
        None,
    )
    if root is None:
        raise RuntimeError("Repo-Root not found (expected folder 'data' or 'scripts').")
    return root, root / "data"


def make_dir_maps(data_dir: Path, size: str = "small") -> Tuple[Dict[str, Path], Dict[str, Path]]:
    if size not in {"small", "big"}:
        raise ValueError("size must be 'small' or 'big'")

    drf_base = data_dir / f"drf_{size}"
    its_base = data_dir / f"its_{size}"

    drf = {
        "edge": drf_base / "precomputed_drf_edge",
        "vertex": drf_base / "precomputed_drf_vertex",
        "sp": drf_base / "precomputed_drf_sp",
    }
    its = {
        "edge": its_base / "precomputed_its_edge",
        "vertex": its_base / "precomputed_its_vertex",
        "sp": its_base / "precomputed_its_sp",
    }
    return drf, its


def list_precomputed_dir_tuples(data_dir: Path, family: str, size: str = "big") -> List[Tuple[Path, str]]:
    if family not in {"drf", "its"}:
        raise ValueError("family must be 'drf' or 'its'")
    base = data_dir / f"{family}_{size}"
    prefix = f"precomputed_{family}_"
    return [
        (base / f"{prefix}edge", "edge"),
        (base / f"{prefix}vertex", "vertex"),
        (base / f"{prefix}sp", "sp"),
    ]


# -------------------------------
# Laden/Kernel/Hilfsfunktionen
# -------------------------------
def load_features_from_dir_tuples(
    dir_tuples: Sequence[Tuple[Path, str]],
    feature_key: str,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    X_map: Dict[str, List] = {}
    y_map: Dict[str, List] = {}

    for path, mode in dir_tuples:
        assert path.exists(), f"Pfad nicht gefunden: {path}"
        X, y = load_precomputed_features(path, feature_key=feature_key)
        X_map[mode] = X
        y_map[mode] = y
        print(f"Loaded {feature_key} ({mode}) from {path}")
        print("  #reactions:", len(X), " #classes:", len(set(y)))
    return X_map, y_map


def find_first_nonzero_kernel_pair(
    X: Sequence,
    kernel_func=kernel_multiset_intersection,
) -> Optional[Tuple[int, int, float]]:
    for i in range(len(X)):
        if len(X[i]) == 0:
            continue
        for j in range(i + 1, len(X)):
            if len(X[j]) == 0:
                continue
            k = kernel_func(X[i], X[j])
            if k > 0:
                print("Found non-zero kernel at:", i, j, "value:", k)
                return i, j, float(k)
    print("No non-zero kernel pair found.")
    return None


def build_kernel_and_stats(
    X_map: Dict[str, List],
    y_map: Dict[str, List],
    mode: str,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    K, y_small = build_kernel_matrix_from_loaded(X_map, y_map, mode=mode, n=n)
    stats = kernel_matrix_stats(K)
    return K, np.asarray(y_small), stats


def plot_kernel_heatmap(K: np.ndarray, title: str):
    fig = px.imshow(K, title=title, aspect="auto")
    return fig


def upper_triangle_values(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    return K[np.triu_indices(n, k=1)]


def histogram_two_kernels(K_a: np.ndarray, K_b: np.ndarray, name_a: str, name_b: str):
    vals_a = upper_triangle_values(K_a)
    vals_b = upper_triangle_values(K_b)
    fig = px.histogram(
        x=[vals_a, vals_b],
        labels={"value": "Kernel value", "variable": "Kernel"},
        nbins=50,
        opacity=0.6,
        title=f"Distribution of Kernel Values: {name_a} vs {name_b}",
    )
    fig.data[0].name = name_a
    fig.data[1].name = name_b
    return fig


def count_pkls(p: Path) -> int:
    return len(list(p.glob("*.pkl")))


def print_pkl_counts(drf_dirs: Dict[str, Path], its_dirs: Dict[str, Path]) -> None:
    print("\n--- PKL counts ---")
    for mode, p in drf_dirs.items():
        print("DRF", mode, ":", count_pkls(p))
    for mode, p in its_dirs.items():
        print("ITS", mode, ":", count_pkls(p))


def safe_subset_ids(
    option_subset_ids: Optional[Iterable[str]],
    drf_dir: Path,
    its_dir: Path,
    take: int = 20,
) -> List[str]:
    option_subset_ids = list(option_subset_ids) if option_subset_ids else []
    if len(option_subset_ids) > 0:
        return option_subset_ids
    common = sorted(set(available_subset_ids(drf_dir)) & set(available_subset_ids(its_dir)))
    if not common:
        raise FileNotFoundError("No common subset_*.pkl between DRF and ITS dirs.")
    return common[:take]


# -------------------------------
# Result-Logging
# -------------------------------
@dataclass
class ResultsLog:
    rows: List[dict] = field(default_factory=list)

    @staticmethod
    def _extract_accuracy(res: Union[dict, object]) -> float:
        if isinstance(res, dict):
            for k in ("accuracy", "acc", "score"):
                if k in res and res[k] is not None:
                    return float(res[k])
            if "metrics" in res and isinstance(res["metrics"], dict):
                for k in ("accuracy", "acc", "score"):
                    if k in res["metrics"] and res["metrics"][k] is not None:
                        return float(res["metrics"][k])
        for attr in ("accuracy", "acc", "score"):
            if hasattr(res, attr):
                val = getattr(res, attr)
                if val is not None:
                    return float(val)
        if hasattr(res, "__dict__"):
            d = res.__dict__
            for k in ("accuracy", "acc", "score"):
                if k in d and d[k] is not None:
                    return float(d[k])
        raise ValueError(f"Could not extract accuracy from res. type={type(res)}")

    def add(
        self,
        tag: str,
        kernel: str,
        mode: str,
        n: int,
        test_size: float,
        C: float,
        seed: int,
        res: Union[dict, object],
        subset_ids: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        **extra,
    ) -> float:
        acc = self._extract_accuracy(res)
        self.rows.append(
            {
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
            }
        )
        return acc

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# -------------------------------
# Experiment-Abschnitte (S4–S7)
# -------------------------------
def run_section4_baseline(
    results: ResultsLog,
    drf_dirs: Dict[str, Path],
    its_dirs: Dict[str, Path],
    k_values: Sequence[int] = (1, 2),
    n: int = 600,
    test_size: float = 0.2,
    C: float = 1.0,
    seed: int = 42,
    take_subsets: int = 30,
    min_per_class: int = 5,
    ref_scan: int = 180,
    verbose: bool = False,
) -> pd.DataFrame:
    tag = "S4_baseline"
    for k in k_values:
        opt1 = make_option1_soft_shared_k_classes_config(
            drf_edge_dir=drf_dirs["edge"],
            its_edge_dir=its_dirs["edge"],
            k=k,
            take_subsets=take_subsets,
            min_per_class=min_per_class,
            ref_scan=ref_scan,
        )
        subset_ids = safe_subset_ids(opt1["subset_ids"], drf_dirs["edge"], its_dirs["edge"], take=20)
        if verbose:
            print(f"[S4 | k={k}] using {len(subset_ids)} subsets")

        res = train_svm_from_precomputed_dir(
            precomp_dir=drf_dirs["edge"],
            feature_key="drf_wl",
            subset_ids=subset_ids,
            n=n,
            test_size=test_size,
            C=C,
            seed=seed,
            verbose=False,
        )
        results.add(tag, "DRF–WL", "edge", n, test_size, C, seed, res, subset_ids=subset_ids, k=k)

        res = train_svm_from_precomputed_dir(
            precomp_dir=its_dirs["edge"],
            feature_key="its_wl",
            subset_ids=subset_ids,
            n=n,
            test_size=test_size,
            C=C,
            seed=seed,
            verbose=False,
        )
        results.add(tag, "ITS–WL", "edge", n, test_size, C, seed, res, subset_ids=subset_ids, k=k)

    return results.to_frame()


def run_section5_modes(
    results: ResultsLog,
    drf_dirs: Dict[str, Path],
    its_dirs: Dict[str, Path],
    k_values: Sequence[int] = (1, 2),
    modes: Sequence[str] = ("edge", "vertex", "sp"),
    n: int = 600,
    test_size: float = 0.2,
    C: float = 1.0,
    seed: int = 42,
    take_subsets: int = 30,
    min_per_class: int = 5,
    ref_scan: int = 180,
    verbose: bool = False,
) -> pd.DataFrame:
    tag = "S5_modes"
    for k in k_values:
        opt1 = make_option1_soft_shared_k_classes_config(
            drf_edge_dir=drf_dirs["edge"],
            its_edge_dir=its_dirs["edge"],
            k=k,
            take_subsets=take_subsets,
            min_per_class=min_per_class,
            ref_scan=ref_scan,
        )
        subset_ids = safe_subset_ids(opt1["subset_ids"], drf_dirs["edge"], its_dirs["edge"], take=20)
        if verbose:
            print(f"[S5 | k={k}] using {len(subset_ids)} subsets")

        for mode in modes:
            res = train_svm_from_precomputed_dir(
                precomp_dir=drf_dirs[mode],
                feature_key="drf_wl",
                subset_ids=subset_ids,
                n=n,
                test_size=test_size,
                C=C,
                seed=seed,
                verbose=False,
            )
            results.add(tag, "DRF–WL", mode, n, test_size, C, seed, res, subset_ids=subset_ids, k=k)

            res = train_svm_from_precomputed_dir(
                precomp_dir=its_dirs[mode],
                feature_key="its_wl",
                subset_ids=subset_ids,
                n=n,
                test_size=test_size,
                C=C,
                seed=seed,
                verbose=False,
            )
            results.add(tag, "ITS–WL", mode, n, test_size, C, seed, res, subset_ids=subset_ids, k=k)
    return results.to_frame()


def run_section6_size_sweep(
    results: ResultsLog,
    drf_dirs: Dict[str, Path],
    its_dirs: Dict[str, Path],
    k_values: Sequence[int] = (1, 2),
    n_values: Sequence[int] = (200, 600, 1200),
    test_size: float = 0.2,
    C: float = 1.0,
    seed: int = 42,
    take_subsets: int = 30,
    min_per_class: int = 5,
    ref_scan: int = 180,
    verbose: bool = False,
) -> pd.DataFrame:
    tag = "S6_size"
    for k in k_values:
        opt1 = make_option1_soft_shared_k_classes_config(
            drf_edge_dir=drf_dirs["edge"],
            its_edge_dir=its_dirs["edge"],
            k=k,
            take_subsets=take_subsets,
            min_per_class=min_per_class,
            ref_scan=ref_scan,
        )
        subset_ids = safe_subset_ids(opt1["subset_ids"], drf_dirs["edge"], its_dirs["edge"], take=20)
        if verbose:
            print(f"[S6 | k={k}] using {len(subset_ids)} subsets")

        for n in n_values:
            res = train_svm_from_precomputed_dir(
                precomp_dir=drf_dirs["edge"],
                feature_key="drf_wl",
                subset_ids=subset_ids,
                n=n,
                test_size=test_size,
                C=C,
                seed=seed,
                verbose=False,
            )
            results.add(tag, "DRF–WL", "edge", n, test_size, C, seed, res, subset_ids=subset_ids, k=k)

            res = train_svm_from_precomputed_dir(
                precomp_dir=its_dirs["edge"],
                feature_key="its_wl",
                subset_ids=subset_ids,
                n=n,
                test_size=test_size,
                C=C,
                seed=seed,
                verbose=False,
            )
            results.add(tag, "ITS–WL", "edge", n, test_size, C, seed, res, subset_ids=subset_ids, k=k)
    return results.to_frame()


def run_section7_split_sweep(
    results: ResultsLog,
    drf_dirs: Dict[str, Path],
    its_dirs: Dict[str, Path],
    k: int,
    n: int = 600,
    test_sizes: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    C: float = 1.0,
    seed: int = 42,
    subset_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    tag = "S7_split"
    print(f"\n[Section 7] Using {'ALL' if subset_ids is None else len(subset_ids)} subsets")

    for ts in test_sizes:
        res = train_svm_from_precomputed_dir(
            precomp_dir=drf_dirs["edge"],
            feature_key="drf_wl",
            subset_ids=subset_ids,
            n=n,
            test_size=ts,
            C=C,
            seed=seed,
            verbose=False,
        )
        results.add(tag, "DRF–WL", "edge", n, ts, C, seed, res, subset_ids=subset_ids, k=k)

        res = train_svm_from_precomputed_dir(
            precomp_dir=its_dirs["edge"],
            feature_key="its_wl",
            subset_ids=subset_ids,
            n=n,
            test_size=ts,
            C=C,
            seed=seed,
            verbose=False,
        )
        results.add(tag, "ITS–WL", "edge", n, ts, C, seed, res, subset_ids=subset_ids, k=k)

    return results.to_frame()


# -------------------------------
# Auswertung/Plots
# -------------------------------
def quick_summary_and_dashboard(df_results: pd.DataFrame, title_prefix: str = "WP3 (k=1 vs k=2)"):
    print("Rows:", len(df_results))
    ipy_display(df_results.head(10))  # <-- Explizit IPython.display verwenden

    gb = (
        df_results.groupby(["mode", "k", "n", "test_size"])["kernel"]
        .apply(lambda s: {x.upper() for x in s.astype(str)})
        .sort_index()
    )
    missing_pairs = gb[~gb.apply(lambda u: {"DRF", "ITS"}.issubset(u))]
    print("Kombinationen ohne beide Kernel:", len(missing_pairs))
    ipy_display(missing_pairs.head(20))  # <-- ebenso hier

    view = df_results[df_results["tag"].isin(["S4_baseline", "S5_modes", "S6_size", "S7_split"])][
        ["tag", "kernel", "mode", "k", "n", "test_size", "accuracy", "subset_ids"]
    ].sort_values(["tag", "k", "kernel", "mode", "n", "test_size"])
    ipy_display(view.head(20))  # <-- und hier

    figs = plot_experiment_dashboard(df_results, title_prefix=title_prefix)
    return figs




def sample_exact_counts(
    X: Sequence,
    y: Sequence,
    target_counts: Dict[str, int],
    *,
    seed: int = 42,
) -> Tuple[List, List]:
    """
    Zieht pro Klasse bis zu target_counts[class] Beispiele (zufällig, ohne Überschuss).
    Gibt X_sel, y_sel zurück (gemischt).
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(list(map(str, y)))

    selected_idx: List[int] = []
    for cls, target in target_counts.items():
        if target <= 0:
            continue
        idx = np.where(y_arr == str(cls))[0]
        if len(idx) == 0:
            continue
        take = min(target, len(idx))
        pick = rng.choice(idx, size=take, replace=False)
        selected_idx.extend(pick.tolist())

    rng.shuffle(selected_idx)
    X_sel = [X[i] for i in selected_idx]
    y_sel = [y[i] for i in selected_idx]
    return X_sel, y_sel

def prepare_quota_dataset_and_train(
    *,
    drf_edge_dir: Path,
    its_edge_dir: Path,
    per_class_target: Union[int, Dict[str, int]] = 1000,
    max_per_class: int = 1000,
    feature_key_drf: str = "drf_wl",
    feature_key_its: str = "its_wl",
    test_size: float = 0.2,
    C: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
):
    """
    1) Wähle gemeinsame subset_ids greedily, bis Zielmengen pro Klasse erfüllt (so gut wie möglich)
    2) Lade DRF/ITS Features aus diesen Subsets
    3) Sampele pro Klasse bis zu target_counts
    4) Trainiere SVMs mit precomputed kernel (stratifizierter Split)
    Rückgabe: (res_drf, res_its, cfg)
    """
    cfg = make_option_quota_all_classes_config(
        drf_edge_dir=drf_edge_dir,
        its_edge_dir=its_edge_dir,
        per_class_target=per_class_target,
        max_per_class=max_per_class,
    )
    subset_ids = cfg["subset_ids"]
    target_counts = cfg["target_counts"]

    if verbose:
        print("Quota config:", cfg["name"])
        print("n_subsets:", len(subset_ids))
        print("deficits (remaining):", {k: v for k, v in cfg["deficits"].items() if v > 0})

    # Laden
    X_drf, y_drf = load_precomputed_features_select(
        drf_edge_dir, feature_key=feature_key_drf, subset_ids=subset_ids
    )
    X_its, y_its = load_precomputed_features_select(
        its_edge_dir, feature_key=feature_key_its, subset_ids=subset_ids
    )

    # Pro Klasse exact counts ziehen
    X_drf_q, y_drf_q = sample_exact_counts(X_drf, y_drf, target_counts, seed=seed)
    X_its_q, y_its_q = sample_exact_counts(X_its, y_its, target_counts, seed=seed)

    # Train/Test (SVM mit precomputed kernel kümmert sich um Stratify)
    res_drf = train_svm_with_precomputed_kernel(
        X_drf_q, y_drf_q, n=None, test_size=test_size, seed=seed, C=C, verbose=verbose
    )
    res_its = train_svm_with_precomputed_kernel(
        X_its_q, y_its_q, n=None, test_size=test_size, seed=seed, C=C, verbose=verbose
    )

    return res_drf, res_its, cfg