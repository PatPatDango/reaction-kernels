# scripts/wp3/wp3_subset_handler.py
"""
WP3 Subset Handler — Option 1 (ONLY)

Option 1 = "Pick subsets robustly so DRF and ITS are comparable"

We support two practical Option-1 variants (both are robust and avoid n_samples=0):

A) COMMON-SUBSETS ONLY (no class constraints)
   - take the subset_ids that exist in BOTH DRF and ITS folders
   - good if you just want lots of data fast

B) SOFT SHARED k-CLASSES (recommended)
   - first choose common subset_ids (exists for both)
   - then choose k "must-have" classes from those subsets (top-k frequent)
   - then keep only subset_ids where each must-have class has at least min_per_class
   - each subset may contain additional classes (soft/realistic)

This file contains EVERYTHING you need for Option 1:
- scanning subset IDs
- indexing class counts per subset
- choosing subset IDs for option 1
- sanity helpers to ensure non-empty after filtering

Your SVM / kernel code should just call:
  cfg = make_option1_soft_shared_k_classes_config(...)
  subset_ids = cfg["subset_ids"]
  allowed_classes = cfg["target_classes"]  # use when you filter classes

Note:
- This expects your PKLs to have key "classes" (list of rxn_class labels).
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Any
import pickle


# ============================================================
# 1) File / subset helpers
# ============================================================

def _subset_id_from_filename(name: str) -> Optional[int]:
    """
    Parse subset id from e.g.:
      subset_001.reaction_features_drf_wl_h3_edge.pkl -> 1
    """
    if "subset_" not in name:
        return None
    try:
        return int(name.split("subset_")[1][:3])
    except Exception:
        return None


def available_subset_ids(precomp_dir: str | Path) -> List[int]:
    """
    Return all subset IDs present as subset_*.pkl in a directory.
    """
    precomp_dir = Path(precomp_dir)
    ids: List[int] = []
    for fp in precomp_dir.glob("subset_*.pkl"):
        sid = _subset_id_from_filename(fp.name)
        if sid is not None:
            ids.append(sid)
    return sorted(set(ids))


def list_subset_pkls(precomp_dir: str | Path, subset_ids: Optional[Sequence[int]] = None) -> List[Path]:
    """
    List subset_*.pkl files in a directory. Optionally restrict to subset_ids.
    """
    precomp_dir = Path(precomp_dir)
    files = sorted(precomp_dir.glob("subset_*.pkl"))
    if not files:
        return []

    if subset_ids is None:
        return files

    want_prefixes = {f"subset_{i:03d}" for i in subset_ids}
    return [fp for fp in files if any(fp.name.startswith(p) for p in want_prefixes)]


def load_pkl(fp: str | Path) -> Dict[str, Any]:
    fp = Path(fp)
    with fp.open("rb") as f:
        return pickle.load(f)


def common_subset_ids(drf_dir: str | Path, its_dir: str | Path, *, take_subsets: Optional[int] = None) -> List[int]:
    """
    Subset IDs that exist in BOTH directories.
    """
    drf_ids = set(available_subset_ids(drf_dir))
    its_ids = set(available_subset_ids(its_dir))
    common = sorted(drf_ids & its_ids)
    if take_subsets is not None:
        common = common[:take_subsets]
    return common


# ============================================================
# 2) Build a fast subset index: subset_id -> {class: count}
# ============================================================

def build_subset_index(
    precomp_dir: str | Path,
    *,
    class_key: str = "classes",
    subset_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, int]]:
    """
    Reads each subset_*.pkl and counts classes inside it.
    Returns:
      index[subset_id] = { "1.2.4": 20, "7.9.2": 20, ... }
    """
    precomp_dir = Path(precomp_dir)
    files = list_subset_pkls(precomp_dir, subset_ids=subset_ids)
    if not files:
        raise FileNotFoundError(f"No PKL files found in {precomp_dir}")

    index: Dict[int, Dict[str, int]] = {}
    for fp in files:
        sid = _subset_id_from_filename(fp.name)
        if sid is None:
            continue
        obj = load_pkl(fp)
        if class_key not in obj or obj[class_key] is None:
            raise KeyError(f"{fp.name}: missing key '{class_key}'")
        counts = Counter(map(str, obj[class_key]))
        index[sid] = dict(counts)
    return index


def subset_has_all_classes(
    subset_counts: Dict[str, int],
    must_have: Sequence[str],
    *,
    min_per_class: int = 1,
) -> bool:
    """
    True iff every class in must_have appears at least min_per_class in this subset.
    """
    for c in must_have:
        if subset_counts.get(str(c), 0) < min_per_class:
            return False
    return True


def choose_subsets_with_must_have_classes(
    index: Dict[int, Dict[str, int]],
    must_have: Sequence[str],
    *,
    min_per_class: int = 1,
) -> List[int]:
    """
    Return subset_ids where each must-have class appears >= min_per_class.
    """
    out: List[int] = []
    for sid, counts in index.items():
        if subset_has_all_classes(counts, must_have, min_per_class=min_per_class):
            out.append(sid)
    return sorted(out)


# ============================================================
# 3) Option 1A: "just common subsets"
# ============================================================

def make_option1_common_only_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    take_subsets: int = 20,
) -> Dict[str, Any]:
    """
    OPTION 1A (simple):
    - choose subset_ids that exist in BOTH folders
    - no class requirements

    Good when you want: "just run experiments now".
    """
    common_ids = common_subset_ids(drf_edge_dir, its_edge_dir)
    if not common_ids:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge) dirs.")

    subset_ids = common_ids[:take_subsets]

    return {
        "name": f"opt1_common_only_take{len(subset_ids)}",
        "subset_ids": subset_ids,
        "target_classes": None,  # no filtering by classes
        "take_subsets": take_subsets,
    }


# ============================================================
# 4) Option 1B: Soft shared k-classes (recommended)
# ============================================================

def _top_k_classes_from_index(
    index: Dict[int, Dict[str, int]],
    subset_ids: Sequence[int],
    *,
    k: int,
) -> List[str]:
    """
    Aggregate counts over chosen subset_ids and take top-k classes.
    """
    pool = Counter()
    for sid in subset_ids:
        pool.update(index.get(sid, {}).keys())
    # NOTE: we only count presence across subsets, not occurrences.
    # That makes it robust for "which classes appear often in subsets".
    return [c for c, _ in pool.most_common(k)]


def make_option1_soft_shared_k_classes_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    k: int = 2,                # how many must-have classes
    take_subsets: int = 20,    # how many subsets to keep at most
    min_per_class: int = 10,   # each must-have class must have >= this many samples in a subset
    ref_scan: int = 50,        # how many common subsets to look at when choosing top-k classes
    ref_side: str = "its",     # choose must-have classes based on "its" or "drf"
    class_key: str = "classes",
) -> Dict[str, Any]:
    """
    OPTION 1B (soft / realistic):
    1) Find common subset ids (exist in BOTH DRF and ITS dirs)
    2) From the first ref_scan common subsets, choose k must-have classes that are most common
       (based on ref_side = 'its' or 'drf')
    3) Keep subset_ids where those must-have classes exist with >= min_per_class
       on BOTH DRF and ITS (so comparison is fair)
    4) Take up to take_subsets subset_ids

    This usually yields MANY more subsets than "exactly same classes everywhere",
    while still guaranteeing at least k common classes across all chosen subsets.
    """
    drf_edge_dir = Path(drf_edge_dir)
    its_edge_dir = Path(its_edge_dir)

    # (1) common subset ids
    common_ids = common_subset_ids(drf_edge_dir, its_edge_dir)
    if not common_ids:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge) dirs.")

    # (2) build indexes for the common ids
    drf_index = build_subset_index(drf_edge_dir, class_key=class_key, subset_ids=common_ids)
    its_index = build_subset_index(its_edge_dir, class_key=class_key, subset_ids=common_ids)

    ref_ids = common_ids[: min(ref_scan, len(common_ids))]

    # choose must-have classes based on ref_side
    if ref_side.lower() == "drf":
        must_have = _top_k_classes_from_index(drf_index, ref_ids, k=k)
    else:
        must_have = _top_k_classes_from_index(its_index, ref_ids, k=k)

    if len(must_have) < k:
        # fallback: take whatever exists
        must_have = list(dict.fromkeys(must_have))  # unique-preserve-order

    # (3) choose subsets where must-have classes exist strongly on BOTH sides
    drf_ok = choose_subsets_with_must_have_classes(drf_index, must_have, min_per_class=min_per_class)
    its_ok = choose_subsets_with_must_have_classes(its_index, must_have, min_per_class=min_per_class)

    subset_ids = sorted(set(drf_ok) & set(its_ok))[:take_subsets]

    return {
        "name": f"opt1_soft_shared_k{k}_min{min_per_class}_refScan{min(ref_scan, len(common_ids))}_{ref_side}",
        "subset_ids": subset_ids,
        "target_classes": must_have,   # <-- filter allowed/target classes to these if you want stable experiments
        "k": k,
        "min_per_class": min_per_class,
        "ref_scan": ref_scan,
        "ref_side": ref_side,
        "n_common_available": len(common_ids),
    }


# ============================================================
# 5) Sanity helpers (highly recommended in notebook)
# ============================================================

def print_option1_config(cfg: Dict[str, Any]) -> None:
    print("Option1 config:", cfg.get("name"))
    print("subset_ids (first 20):", (cfg.get("subset_ids") or [])[:20])
    print("n_subsets:", len(cfg.get("subset_ids") or []))
    print("target_classes:", cfg.get("target_classes"))


def count_samples_if_filtered(
    precomp_dir: str | Path,
    *,
    subset_ids: Sequence[int],
    allowed_classes: Sequence[str],
    class_key: str = "classes",
) -> int:
    """
    Fast check: how many labels remain after filtering to allowed_classes?
    Uses only label lists from PKLs (no features).
    """
    precomp_dir = Path(precomp_dir)
    allowed = set(map(str, allowed_classes))

    total = 0
    for fp in list_subset_pkls(precomp_dir, subset_ids=subset_ids):
        obj = load_pkl(fp)
        y = obj.get(class_key, None)
        if y is None:
            raise KeyError(f"{fp.name}: missing key '{class_key}'")
        total += sum(1 for lab in y if str(lab) in allowed)
    return total


def ensure_nonempty_or_raise(
    precomp_dir: str | Path,
    *,
    subset_ids: Sequence[int],
    allowed_classes: Optional[Sequence[str]],
    min_needed: int = 20,
    class_key: str = "classes",
) -> None:
    """
    Raises a ValueError if filtering would leave too few samples.
    """
    if allowed_classes is None:
        return
    kept = count_samples_if_filtered(
        precomp_dir,
        subset_ids=subset_ids,
        allowed_classes=allowed_classes,
        class_key=class_key,
    )
    if kept < min_needed:
        raise ValueError(
            f"Too few samples after filtering: kept={kept} (<{min_needed}). "
            f"subset_ids={list(subset_ids)[:10]}..., allowed_classes={list(allowed_classes)}"
        )