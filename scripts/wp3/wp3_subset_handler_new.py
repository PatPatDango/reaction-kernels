from __future__ import annotations

"""
WP3 Subset Handler — Option 1 (ONLY)

Option 1 = "Pick subsets robustly so DRF and ITS are comparable"

We support two practical Option-1 variants:

A) COMMON-SUBSETS ONLY (no class constraints)
   - take the subset_ids that exist in BOTH DRF and ITS folders

B) SOFT SHARED k-CLASSES (recommended)
   - first choose common subset_ids (exists for both)
   - then choose k "must-have" classes (chosen from ref_side) from those subsets
   - then keep only subset_ids where each must-have class has at least min_per_class
     on BOTH sides
"""

from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
import pickle


# -------------------------------
# 1) File / subset helpers
# -------------------------------
def _subset_id_from_filename(name: str) -> Optional[int]:
    if "subset_" not in name:
        return None
    try:
        return int(name.split("subset_")[1][:3])
    except Exception:
        return None


def available_subset_ids(precomp_dir: str | Path) -> List[int]:
    precomp_dir = Path(precomp_dir)
    ids: List[int] = []
    for fp in precomp_dir.glob("subset_*.pkl"):
        sid = _subset_id_from_filename(fp.name)
        if sid is not None:
            ids.append(sid)
    return sorted(set(ids))


def list_subset_pkls(precomp_dir: str | Path, subset_ids: Optional[Sequence[int]] = None) -> List[Path]:
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
    drf_ids = set(available_subset_ids(drf_dir))
    its_ids = set(available_subset_ids(its_dir))
    common = sorted(drf_ids & its_ids)
    if take_subsets is not None:
        common = common[:take_subsets]
    return common


def sanitize_subset_ids(precomp_dir: str | Path, subset_ids: Sequence[int]) -> List[int]:
    avail = set(available_subset_ids(precomp_dir))
    cleaned = [sid for sid in subset_ids if sid in avail]
    missing = [sid for sid in subset_ids if sid not in avail]
    if missing:
        print(f"[WARN] Missing subset ids in {precomp_dir}: {missing}")
    if not cleaned:
        raise ValueError(
            f"After sanitizing, subset_ids is empty.\n"
            f"Available ids (first 50): {sorted(avail)[:50]}"
        )
    return cleaned


# -------------------------------
# 2) Subset index: subset_id -> {class: count}
# -------------------------------
def build_subset_index(
    precomp_dir: str | Path,
    *,
    class_key: str = "classes",
    subset_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, int]]:
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
    out: List[int] = []
    for sid, counts in index.items():
        if subset_has_all_classes(counts, must_have, min_per_class=min_per_class):
            out.append(sid)
    return sorted(out)


# -------------------------------
# 3) Option 1 configs
# -------------------------------
def make_option1_common_only_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    take_subsets: int = 20,
) -> Dict[str, Any]:
    common_ids = common_subset_ids(drf_edge_dir, its_edge_dir)
    if not common_ids:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge) dirs.")
    subset_ids = common_ids[:take_subsets]
    return {
        "name": f"opt1_common_only_take{len(subset_ids)}",
        "subset_ids": subset_ids,
        "target_classes": None,
        "take_subsets": take_subsets,
    }


def _top_k_classes_from_index(
    index: Dict[int, Dict[str, int]],
    subset_ids: Sequence[int],
    *,
    k: int,
) -> List[str]:
    pool = Counter()
    for sid in subset_ids:
        # Präsenz über Subsets (nicht Anzahl Samples) — robust
        pool.update(index.get(sid, {}).keys())
    return [c for c, _ in pool.most_common(k)]


def make_option1_soft_shared_k_classes_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    k: int = 2,
    take_subsets: int = 20,
    min_per_class: int = 10,
    ref_scan: int = 50,
    ref_side: str = "its",
    class_key: str = "classes",
) -> Dict[str, Any]:
    drf_edge_dir = Path(drf_edge_dir)
    its_edge_dir = Path(its_edge_dir)

    common_ids = common_subset_ids(drf_edge_dir, its_edge_dir)
    if not common_ids:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge) dirs.")

    drf_index = build_subset_index(drf_edge_dir, class_key=class_key, subset_ids=common_ids)
    its_index = build_subset_index(its_edge_dir, class_key=class_key, subset_ids=common_ids)

    ref_ids = common_ids[: min(ref_scan, len(common_ids))]
    must_have = _top_k_classes_from_index(
        drf_index if ref_side.lower() == "drf" else its_index,
        ref_ids,
        k=k,
    )
    if len(must_have) < k:
        must_have = list(dict.fromkeys(must_have))  # de-dupe, preserve order

    drf_ok = choose_subsets_with_must_have_classes(drf_index, must_have, min_per_class=min_per_class)
    its_ok = choose_subsets_with_must_have_classes(its_index, must_have, min_per_class=min_per_class)
    subset_ids = sorted(set(drf_ok) & set(its_ok))[:take_subsets]

    return {
        "name": f"opt1_soft_shared_k{k}_min{min_per_class}_refScan{min(ref_scan, len(common_ids))}_{ref_side}",
        "subset_ids": subset_ids,
        "target_classes": must_have,
        "k": k,
        "min_per_class": min_per_class,
        "ref_scan": ref_scan,
        "ref_side": ref_side,
        "n_common_available": len(common_ids),
    }


# -------------------------------
# 4) Sanity helpers
# -------------------------------
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
    



# ============ NEU: Quota-Auswahl über gemeinsame Subsets (DRF & ITS) ============
def _normalize_targets(
    classes: Sequence[str],
    per_class_target: Union[int, Dict[str, int]],
    max_per_class: int = 1000,
) -> Dict[str, int]:
    """
    Erzeugt Zielmengen pro Klasse:
    - Wenn per_class_target ein int ist, setze für alle Klassen diesen Wert (gecapped).
    - Wenn dict, übernehme pro Klasse den Wert (gecapped), fehlende Klassen -> 0.
    """
    targets: Dict[str, int] = {}
    if isinstance(per_class_target, int):
        val = min(max(per_class_target, 0), max_per_class)
        for c in classes:
            targets[str(c)] = val
    else:
        for c in classes:
            t = int(per_class_target.get(str(c), 0))
            targets[str(c)] = min(max(t, 0), max_per_class)
    return targets

def _merge_class_sets(drf_index: Dict[int, Dict[str, int]], its_index: Dict[int, Dict[str, int]]) -> List[str]:
    """
    Liste aller Klassen, die in irgendeinem gemeinsamen Subset auf mindestens einer Seite vorkommen.
    Für Fairness im Quota-Build nehmen wir aber Coverage = min(drf, its) je Klasse/SID.
    """
    classes = set()
    for sid in drf_index:
        classes.update(drf_index[sid].keys())
    for sid in its_index:
        classes.update(its_index[sid].keys())
    return sorted(classes)

def greedy_select_common_subsets_for_quota(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    per_class_target: Union[int, Dict[str, int]] = 1000,
    max_per_class: int = 1000,
    class_key: str = "classes",
    candidate_subset_ids: Optional[Sequence[int]] = None,
    max_subsets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Greedy wählt gemeinsame subset_ids (existieren in DRF & ITS), bis pro Klasse die gewünschte Menge erreicht ist.
    Coverage pro subset_id und Klasse = min(count_DRF, count_ITS) (Fairness: beide Seiten haben genügend).

    Vorgehen:
    - Kandidaten = gemeinsame subset_ids (oder übergeben)
    - Baue Indexe: sid -> {class: count} für DRF und ITS
    - Ziele: targets[class] (int oder dict), gecapped bei max_per_class
    - Solange Defizite existieren:
        - Wähle das subset, das die größte Defizit-Summe reduziert
        - Ziehe Coverage von Defiziten ab
        - Stoppe, wenn keine Verbesserung mehr möglich oder max_subsets erreicht
    Rückgabe:
      {
        'name': 'quota_all_classes',
        'subset_ids': [...],
        'target_counts': {class: target},
        'achieved_counts': {class: achieved},
        'deficits': {class: remaining},
        'n_candidates': int,
      }
    """
    drf_edge_dir = Path(drf_edge_dir)
    its_edge_dir = Path(its_edge_dir)

    # Kandidaten: gemeinsame Subsets
    if candidate_subset_ids is None:
        candidates = common_subset_ids(drf_edge_dir, its_edge_dir)
    else:
        candidates = sorted(set(common_subset_ids(drf_edge_dir, its_edge_dir)) & set(candidate_subset_ids))

    if not candidates:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge) dirs.")

    # Indizes bauen begrenzt auf Kandidaten
    drf_index = build_subset_index(drf_edge_dir, class_key=class_key, subset_ids=candidates)
    its_index = build_subset_index(its_edge_dir, class_key=class_key, subset_ids=candidates)

    classes = _merge_class_sets(drf_index, its_index)
    targets = _normalize_targets(classes, per_class_target, max_per_class=max_per_class)

    # Defizite initialisieren (nur Klassen mit Ziel > 0)
    deficits: Dict[str, int] = {c: t for c, t in targets.items() if t > 0}
    achieved: Dict[str, int] = {c: 0 for c in targets.keys()}

    # Precompute coverage je subset
    coverage_by_sid: Dict[int, Dict[str, int]] = {}
    for sid in candidates:
        cov: Dict[str, int] = {}
        di = drf_index.get(sid, {})
        ii = its_index.get(sid, {})
        # Coverage ist min(counts) je Klasse
        for c in deficits.keys():
            cov[c] = min(di.get(c, 0), ii.get(c, 0))
        coverage_by_sid[sid] = cov

    selected: List[int] = []
    remaining = set(candidates)

    def total_deficit(deficits: Dict[str, int]) -> int:
        return sum(max(d, 0) for d in deficits.values())

    prev_total = total_deficit(deficits)

    while remaining and total_deficit(deficits) > 0:
        # Wähle das subset mit maximaler Defizitreduktion
        best_sid = None
        best_gain = 0
        for sid in list(remaining):
            cov = coverage_by_sid[sid]
            gain = 0
            for c, need in deficits.items():
                if need <= 0:
                    continue
                take = min(need, cov.get(c, 0))
                gain += take
            if gain > best_gain:
                best_gain = gain
                best_sid = sid

        if best_sid is None or best_gain <= 0:
            # keine weitere Verbesserung möglich
            print("[INFO] No further gain from remaining subsets. Stopping.")
            break

        # subset übernehmen
        selected.append(best_sid)
        remaining.remove(best_sid)

        # Defizite aktualisieren und Achieved zählen
        cov = coverage_by_sid[best_sid]
        for c in deficits.keys():
            inc = cov.get(c, 0)
            if inc > 0:
                achieved[c] += inc
                deficits[c] = max(deficits[c] - inc, 0)

        # Abbruchbedingung max_subsets
        if max_subsets is not None and len(selected) >= max_subsets:
            print(f"[INFO] Reached max_subsets={max_subsets}. Stopping.")
            break

        new_total = total_deficit(deficits)
        if new_total >= prev_total:
            print("[WARN] No deficit reduction after selection (unexpected). Stopping to avoid loop.")
            break
        prev_total = new_total

    return {
        "name": "quota_all_classes",
        "subset_ids": selected,
        "target_counts": targets,
        "achieved_counts": achieved,
        "deficits": deficits,
        "n_candidates": len(candidates),
    }

def make_option_quota_all_classes_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    per_class_target: Union[int, Dict[str, int]] = 1000,
    max_per_class: int = 1000,
    class_key: str = "classes",
    candidate_subset_ids: Optional[Sequence[int]] = None,
    max_subsets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Wrapper, der die greedy Auswahl aufruft und eine kompakte Config zurückgibt.
    """
    plan = greedy_select_common_subsets_for_quota(
        drf_edge_dir=drf_edge_dir,
        its_edge_dir=its_edge_dir,
        per_class_target=per_class_target,
        max_per_class=max_per_class,
        class_key=class_key,
        candidate_subset_ids=candidate_subset_ids,
        max_subsets=max_subsets,
    )
    cfg = {
        "name": f"opt_quota_all_classes_{len(plan['subset_ids'])}subs",
        "subset_ids": plan["subset_ids"],
        "target_counts": plan["target_counts"],
        "achieved_counts": plan["achieved_counts"],
        "deficits": plan["deficits"],
        "n_candidates": plan["n_candidates"],
    }
    return cfg