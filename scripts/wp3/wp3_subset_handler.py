from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any, Optional, Sequence
import pickle







# ============================================================
# 1) Tiny IO helpers (subset files + PKL loading)
# ============================================================

def _subset_id(name: str) -> Optional[int]:
    """
    lädt die Subset-ID aus einem Dateinamen.
    """
    if "subset_" not in name:  #wenn subset_ nicht im Namen ist
        return None
    try:
        # split[:3] nimmt die ersten 3 Zeichen (z.B. "001")
        return int(name.split("subset_")[1][:3])
    except Exception:
        return None


def _subset_pkls(precomp_dir: str | Path, subset_ids: Optional[Sequence[int]] = None) -> list[Path]:
    """
    Gibt alle subset_*.pkl Dateien in einem Ordner zurück.
    Optional: nur bestimmte subset_ids laden.

    precomp_dir: z.B. "data/drf_small/precomputed_drf_edge"
    subset_ids: z.B. [1,2,3] -> lädt nur subset_001*, subset_002*, subset_003*
    """
    d = Path(precomp_dir)

    # alle Dateien die so heißen wie subset_*.pkl
    files = sorted(d.glob("subset_*.pkl"))

    # wenn subset_ids nicht angegeben ist -> alle zurückgeben
    if subset_ids is None:
        return files

    # bestimmte subset_ids filtern
    want = {f"subset_{i:03d}" for i in subset_ids}

    # behalten nur Dateien, deren Name mit einem dieser Prefixe startet
    return [fp for fp in files if any(fp.name.startswith(w) for w in want)]


def _load(fp: Path) -> dict[str, Any]:
    """
    Lädt eine einzelne .pkl Datei und gibt das enthaltene Python-Objekt zurück.
    = ein dict mit keys: --> metadaten + features + classes
      - "drf_wl" oder "its_wl"
      - "classes"
      - "meta" ...
    """
    with fp.open("rb") as f:
        return pickle.load(f)


def available_subset_ids(precomp_dir: str | Path) -> list[int]:
    """
    Überprüft und listet alle Subset-IDs auf die da existieren.
    """
    ids = []
    for fp in Path(precomp_dir).glob("subset_*.pkl"):
        sid = _subset_id(fp.name)
        if sid is not None:
            ids.append(sid)
    return sorted(set(ids))  # set => doppelte entfernen


def common_subset_ids(drf_dir: str | Path, its_dir: str | Path, *, take: Optional[int] = None) -> list[int]:
    """
    Findet Subset-IDs die sowohl im DRF-Ordner als auch im ITS-Ordner existieren.
    fair vergleichen: gleiche subset_XXX auf beiden Seite-> Welcher Kernel ist besser?
    take: wenn gesetzt, nehmen wir nur die ersten take IDs (z.B. 20)
    """
    # ids in beiden ordnern holen
    common = sorted(set(available_subset_ids(drf_dir)) & set(available_subset_ids(its_dir)))
    # optional kürzen
    return common[:take] if take is not None else common


# ============================================================
# 2) Build class index per subset
#    -> Für jedes subset: wie oft kommt jede Klasse vor?
# ============================================================

def build_subset_index(
    precomp_dir: str | Path,
    *,
    class_key: str = "classes",
    subset_ids: Optional[Sequence[int]] = None,
) -> dict[int, Counter[str]]:
    """
    index[subset_id] = Counter(class -> count)
    Beispiel: index[1] = {"3.4.1": 20, "6.3.7": 20, ...}
    wichtig für k Klassen Auswahl
    Welche Subsets erfüllen die Bedingungen gleichzeitig für DRF und ITS?
    """
    files = _subset_pkls(precomp_dir, subset_ids=subset_ids)
    if not files:
        raise FileNotFoundError(f"No subset_*.pkl found in {precomp_dir}")

    idx: dict[int, Counter[str]] = {}

    for fp in files:
        # subset_id aus Dateiname ziehen
        sid = _subset_id(fp.name)
        if sid is None:
            continue

        # pkl laden
        obj = _load(fp)

        # prüfen ob der key "classes" drin ist
        if class_key not in obj or obj[class_key] is None:
            raise KeyError(f"{fp.name}: missing key '{class_key}'")

        # Counter macht aus einer Liste von Klassen eine Häufigkeitstabelle
        # map(str, ...) nur zur Sicherheit (falls irgendwo nicht str)
        idx[sid] = Counter(map(str, obj[class_key]))
    return idx

# ============================================================
# 3) K Auswahl : choose subsets that share k must-have classes
#    -> Wir wählen “must-have Klassen” und suchen viele subsets die die enthalten
# ============================================================

def _top_k_classes_by_presence(
    index: dict[int, Counter[str]],
    subset_ids: Sequence[int],
    *,
    k: int,
) -> list[str]:
    """
    Wählt die k Klassen die in den meisten Subsets vorkommen (Presence).
    --> nur ob sie in einem Subset überhaupt vorkommt.
    Idee: Klassen die in vielen Subsets auftauchen, sind “robuster” für Experimente.
    """
    presence = Counter()

    for sid in subset_ids:
        # index[sid].keys() = alle Klassen, die in diesem subset vorkommen
        # presence.update(...) zählt dann: "wie viele Subsets enthalten diese Klasse"
        presence.update(index.get(sid, {}).keys())
    # k häufigste
    return [c for c, _ in presence.most_common(k)]

def _subsets_with_must_have(
    index: dict[int, Counter[str]],
    must_have: Sequence[str],
    *,
    min_per_class: int,
) -> list[int]:
    """
    Gibt subset_ids zurück, die ALLE must-have Klassen enthalten
    UND jede davon mindestens min_per_class mal.

    Beispiel:
      must_have=["3.4.1","6.3.7"], min_per_class=10
      -> subset muss mind. 10 samples von jeder der beiden Klassen enthalten
    """
    must_have = [str(c) for c in must_have]
    ok = []

    for sid, cnt in index.items():
        # cnt ist Counter(class->count) für dieses subset
        if all(cnt.get(c, 0) >= min_per_class for c in must_have):
            ok.append(sid)

    return sorted(ok)

def make_soft_shared_k_classes_config(
    *,
    drf_edge_dir: str | Path,
    its_edge_dir: str | Path,
    k: int = 2,                 # wie viele must-have Klassen
    take_subsets: int = 20,     # max wie viele subsets wir am Ende nehmen
    min_per_class: int = 10,    # jede must-have Klasse muss mind. X mal im subset vorkommen
    ref_scan: int = 50,         # wie viele "frühe" common subsets benutzen wir um must-have zu bestimmen
    ref_side: str = "its",      # "its" oder "drf": wo wir must-have Klassen auswählen
    class_key: str = "classes",
) -> dict[str, Any]:
    """
    viele Subsets, die mindestens k gemeinsame Klassen haben,
    damit DRF und ITS fair verglichen werden können.

    Schritte:
    (1) Finde subset_XXX IDs die in DRF UND ITS existieren (common_ids)
    (2) Lade für diese subsets die Klassenverteilungen (Index bauen)
    (3) Wähle k "must-have" Klassen, die in ref_scan vielen Subsets vorkommen
    (4) Behalte nur subsets, die diese must-have Klassen ausreichend oft enthalten
        -> einmal für DRF, einmal für ITS
    (5) Schnittmenge daraus = Subsets die auf beiden Seiten passen
    """
    # (1) common subset ids
    common_ids = common_subset_ids(drf_edge_dir, its_edge_dir)
    if not common_ids:
        raise FileNotFoundError("No common subset PKLs between DRF(edge) and ITS(edge).")

    # (2) Index bauen: pro subset die Klassenhäufigkeit
    drf_idx = build_subset_index(drf_edge_dir, class_key=class_key, subset_ids=common_ids)
    its_idx = build_subset_index(its_edge_dir, class_key=class_key, subset_ids=common_ids)

    # wir scannen nur die ersten ref_scan subsets als “Basis”
    ref_ids = common_ids[: min(ref_scan, len(common_ids))]

    # (3) must-have Klassen auswählen
    if ref_side.lower() == "drf":
        must_have = _top_k_classes_by_presence(drf_idx, ref_ids, k=k)
    else:
        must_have = _top_k_classes_by_presence(its_idx, ref_ids, k=k)

    # (4) Subsets filtern: müssen must-have Klassen genug oft enthalten
    drf_ok = _subsets_with_must_have(drf_idx, must_have, min_per_class=min_per_class)
    its_ok = _subsets_with_must_have(its_idx, must_have, min_per_class=min_per_class)

    # (5) Subsets nehmen, die auf BEIDEN Seiten ok sind
    subset_ids = sorted(set(drf_ok) & set(its_ok))[:take_subsets]

    # Konfig zurückgeben (praktisch fürs Notebook)
    return {
        "name": f"opt1_soft_k{k}_min{min_per_class}_scan{min(ref_scan, len(common_ids))}_{ref_side}",
        "subset_ids": subset_ids,
        "target_classes": must_have,   # diese Klassen sind garantiert "stabil" (wenn subset_ids nicht leer ist)
        "k": k,
        "min_per_class": min_per_class,
        "ref_scan": ref_scan,
        "ref_side": ref_side,
        "n_common_available": len(common_ids),
    }

def print_k_config(cfg: dict[str, Any]) -> None:
    """
    Print für Notebook.
    """
    print("Option1 config:", cfg.get("name"))                   #name der Config
    print("n_common_available:", cfg.get("n_common_available")) #Anzahl der gemeinsamen Subsets in DRF und ITS
    print("target_classes:", cfg.get("target_classes"))         #die ausgewählten must-have Klassen
    sids = cfg.get("subset_ids") or []                          #die final ausgewählten subset IDs
    print("n_subsets:", len(sids))                              #Anzahl der final ausgewählten Subsets
    print("subset_ids (first 20):", sids[:20])                   #die final ausgewählten subset IDs (erste 20)

def safe_subset_ids(option_subset_ids, drf_dir, its_dir, take=20):
    # prüft ob option_subset_ids gesetzt ist 
    # verhindert leere datasets 
    option_subset_ids = list(option_subset_ids) if option_subset_ids else []
    if len(option_subset_ids) > 0:
        return option_subset_ids

    # fallback: nimm einfach gemeinsame subset ids
    common = sorted(set(available_subset_ids(drf_dir)) & set(available_subset_ids(its_dir)))
    if not common:
        raise FileNotFoundError("No common subset_*.pkl between DRF and ITS dirs.")
    return common[:take]