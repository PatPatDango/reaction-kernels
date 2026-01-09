# ================================
# WP2: Pre-compute feature hash sets
# ================================
# This code computes (hashed) DRF and WL–DRF feature Counters for *all* reactions
# in a dataset (or in multiple subset TSVs) and saves them to disk, so you don't
# have to recompute them later for kernels.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle

import pandas as pd
from collections import Counter
from .wp2_functions import its_wl_feature_sets_per_iter_from_rsmi
from .wp2_functions import drf_wl_features_from_rsmi

# --- import your feature functions ---
# adjust the import path/module name to your project setup
# (e.g., from src.wp2_functions import ...)
from .wp2_functions import (
    drf_features_from_rsmi,
    drf_wl_features_from_rsmi,
)

# -----------------------
# Helpers
# -----------------------

def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...], purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {purpose} column. Tried {candidates}. Available: {list(df.columns)}")

def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)

def load_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

# -----------------------
# Core precompute function
# -----------------------

def precompute_features_for_dataframe(
    df: pd.DataFrame,
    *,
    rxn_col: Optional[str] = None,
    label_col: Optional[str] = None,
    # feature configs
    drf_mode: str = "edge",         # "vertex" | "edge" | "sp"
    wl_h: int = 3,
    wl_mode: str = "edge",          # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    # behavior
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dict containing:
      - 'meta': configuration + column names
      - 'labels': list of class labels (if label_col exists)
      - 'rsmi': list of reaction SMILES strings (same order)
      - 'drf': list[Counter[str]] (same order)
      - 'wl_drf': list[Counter[str]] (same order)

    Each Counter maps feature_hash -> count (multiset).
    """

    # auto-detect columns if not provided
    if rxn_col is None:
        if "clean_rxn" not in df.columns:
            raise KeyError(
            "Required column 'clean_rxn' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    rxn_col = "clean_rxn"
    if label_col is None:
        # label column is optional; if none found, we proceed without it
        try:
            label_col = _pick_col(df, ("rxn_class", "rxn class", "class", "label"), "class label")
        except KeyError:
            label_col = None

    rsmi_list = df[rxn_col].astype(str).tolist()
    labels_list = df[label_col].tolist() if label_col is not None else None

    drf_list: list[Counter[str]] = []
    wl_drf_list: list[Counter[str]] = []
    errors: list[dict] = []

    # progress bar (optional)
    try:
        from tqdm.auto import tqdm  # type: ignore
        it = tqdm(range(len(rsmi_list)), desc="Precomputing features")
    except Exception:
        it = range(len(rsmi_list))

    for i in it:
        rsmi = rsmi_list[i]
        try:
            # DRF (no WL)
            drf = drf_features_from_rsmi(
                rsmi,
                mode=drf_mode,
                include_edge_labels_in_sp=include_edge_labels_in_sp,
                hash_labels=True,          # your code uses hash_labels for base Φ
                digest_size=digest_size,
                raw_labels=False,
            )

            # WL–DRF (WL labels + DRF)
            wl_drf = drf_wl_features_from_rsmi(
                rsmi,
                h=wl_h,
                mode=wl_mode,
                include_edge_labels_in_sp=include_edge_labels_in_sp,
                hash_node_labels=hash_node_labels,
                hash_features=hash_features,
                digest_size=digest_size,
            )

            drf_list.append(drf)
            wl_drf_list.append(wl_drf)

        except Exception as e:
            err = {
                "index": i,
                "rsmi": rsmi,
                "error": repr(e),
            }
            errors.append(err)

            if stop_on_error:
                raise

            # keep alignment: append empty counters for failed items
            drf_list.append(Counter())
            wl_drf_list.append(Counter())

    out: Dict[str, Any] = {
        "meta": {
            "rxn_col": rxn_col,
            "label_col": label_col,
            "drf_mode": drf_mode,
            "wl_h": wl_h,
            "wl_mode": wl_mode,
            "include_edge_labels_in_sp": include_edge_labels_in_sp,
            "hash_node_labels": hash_node_labels,
            "hash_features": hash_features,
            "digest_size": digest_size,
            "n_rows": len(df),
            "n_errors": len(errors),
        },
        "rsmi": rsmi_list,
        "drf": drf_list,
        "wl_drf": wl_drf_list,
        "errors": errors,
    }
    if labels_list is not None:
        out["labels"] = labels_list

    return out

# -----------------------
# Convenience: precompute and save
# -----------------------

def precompute_and_save(
    input_tsv: str | Path,
    out_path: str | Path,
    *,
    rxn_col: Optional[str] = None,
    label_col: Optional[str] = None,
    drf_mode: str = "edge",
    wl_h: int = 3,
    wl_mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    digest_size: int = 16,
) -> Path:
    df = load_tsv(input_tsv)
    feats = precompute_features_for_dataframe(
        df,
        rxn_col=rxn_col,
        label_col=label_col,
        drf_mode=drf_mode,
        wl_h=wl_h,
        wl_mode=wl_mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        digest_size=digest_size,
    )
    save_pickle(feats, out_path)
    return Path(out_path)

def precompute_all_subsets_in_dir_drf_wl(
    subsets_dir: str | Path,
    out_dir: str | Path,
    *,
    rxn_col: str = "clean_rxn",
    class_col: str = "rxn_class",
    pattern: str = "*.tsv",
    h: int = 3,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> None:
    """
    Pre-compute DRF–WL feature representations for all subset TSV files.

    For each reaction:
      - build educt and product graphs
      - apply WL (0..h)
      - compute DRF over WL features (union/sum over iterations inside the Counter)
      - store as Counter(feature_hash -> count)

    Results are saved as .pkl files, one per subset.
    """
    subsets_dir = Path(subsets_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(subsets_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {subsets_dir}")

    for f in files:
        print(f"[+] DRF–WL precompute for {f.name}")
        df = pd.read_csv(f, sep="\t")

        if rxn_col not in df.columns:
            raise KeyError(f"Column '{rxn_col}' not found in {f.name}")

        if class_col not in df.columns:
            raise KeyError(f"Column '{class_col}' not found in {f.name}")

        features = []
        errors = []

        for idx, rsmi in enumerate(df[rxn_col].astype(str)):
            try:
                # ✅ FIX: drf_wl_features_from_rsmi returns ONE Counter
                total = drf_wl_features_from_rsmi(
                    rsmi,
                    h=h,
                    mode=mode,
                    include_edge_labels_in_sp=include_edge_labels_in_sp,
                    hash_node_labels=hash_node_labels,
                    hash_features=hash_features,
                    digest_size=digest_size,
                )
                features.append(total)

            except Exception as e:
                errors.append({"index": idx, "rsmi": rsmi, "error": repr(e)})
                features.append(Counter())

        out = {
            "meta": {
                "type": "DRF–WL",
                "rxn_col": rxn_col,
                "class_col": class_col,
                "h": h,
                "mode": mode,
                "include_edge_labels_in_sp": include_edge_labels_in_sp,
                "hash_node_labels": hash_node_labels,
                "hash_features": hash_features,
                "digest_size": digest_size,
                "n_rows": len(df),
                "n_errors": len(errors),
            },
            "rsmi": df[rxn_col].tolist(),
            "classes": df[class_col].tolist(),
            "drf_wl": features,
            "errors": errors,
        }

        out_path = out_dir / f"{f.stem}.reaction_features_drf_wl_h{h}_{mode}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(out, fh)

        print(f"    → saved to {out_path.name} | errors: {len(errors)}/{len(df)}")

    print("[✓] DRF–WL precompute finished.")
# ================================
# Example usage in your notebook
# ================================

# (A) Precompute for one dataset TSV
# out = precompute_and_save(
#     input_tsv="schneider50k_clean.tsv",
#     out_path="precomputed/schneider50k.drf_wl_h3_edge.pkl",
#     drf_mode="edge",
#     wl_h=3,
#     wl_mode="edge",
# )

# (B) Precompute for all subset TSVs you saved (e.g. subsets_small/)
# precompute_all_subsets_in_dir(
#     subsets_dir="subsets_small",
#     out_dir="precomputed_subsets",
#     pattern="subset*.tsv",
#     drf_mode="edge",
#     wl_h=3,
#     wl_mode="edge",
# )

def precompute_all_subsets_in_dir_its_wl(
    subsets_dir: str | Path,
    out_dir: str | Path,
    *,
    rxn_col: str = "clean_rxn",
    class_col: str = "rxn_class",
    pattern: str = "*.tsv",
    h: int = 3,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> None:
    """
    Pre-compute ITS–WL feature representations for all subset TSV files.

    For each reaction:
      - build ITS graph
      - compute WL features for iterations 0..h
      - take the union over all iterations
      - store as Counter(feature_hash -> count)

    Results are saved as .pkl files, one per subset.
    """
    subsets_dir = Path(subsets_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(subsets_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {subsets_dir}")

    for f in files:
        print(f"[+] ITS–WL precompute for {f.name}")
        df = pd.read_csv(f, sep="\t")

        if rxn_col not in df.columns:
            raise KeyError(f"Column '{rxn_col}' not found in {f.name}")
        
        if class_col not in df.columns:
            raise KeyError(f"Column '{class_col}' not found in {f.name}")

        features = []
        errors = []

        for idx, rsmi in enumerate(df[rxn_col].astype(str)):
            try:
                _, total = its_wl_feature_sets_per_iter_from_rsmi(
                    rsmi,
                    h=h,
                    mode=mode,
                    include_edge_labels_in_sp=include_edge_labels_in_sp,
                    hash_node_labels=hash_node_labels,
                    hash_features=hash_features,
                    digest_size=digest_size,
                )
                features.append(total)
            except Exception as e:
                errors.append({"index": idx, "rsmi": rsmi, "error": repr(e)})
                features.append(Counter())

        out = {
            "meta": {
                "type": "ITS–WL",
                "rxn_col": rxn_col,
                "h": h,
                "mode": mode,
                "hash_node_labels": hash_node_labels,
                "hash_features": hash_features,
                "digest_size": digest_size,
                "n_rows": len(df),
                "n_errors": len(errors),
            },
            "rsmi": df[rxn_col].tolist(),
            "classes": df[class_col].tolist() if class_col in df.columns else None,
            "its_wl": features,
            "errors": errors,
        }

        out_path = out_dir / f"{f.stem}.reaction_features_its_wl_h{h}_{mode}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(out, fh)

        print(f"    → saved to {out_path.name}")

    print("[✓] ITS–WL precompute finished.")