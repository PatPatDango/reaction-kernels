import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _random_chunk_sizes(total: int, min_n: int, max_n: int, rnd: random.Random) -> List[int]:

    sizes: List[int] = []
    remaining = total
    while remaining > 0:
        if remaining <= max_n:
            # Letzter Chunk: n == remaining (liegt in [min_n, max_n], sonst ist das Setup nicht erfüllbar)
            n = remaining
        else:
            # Wähle n in [min_n, max_n], aber so, dass der verbleibende Rest >= min_n bleibt
            low = min_n
            high = min(max_n, remaining - min_n)
            if low > high:
                # Fallback: min_n nehmen, damit der Rest nicht < min_n wird
                n = min_n
            else:
                n = rnd.randint(low, high)
        sizes.append(n)
        remaining -= n
    return sizes


def _partition_class_rows(grp: pd.DataFrame, min_n: int, max_n: int, rnd: random.Random) -> List[pd.DataFrame]:
    """
    Schuffle Zeilen einer Klasse und teile sie in zufällige Chunks zwischen min_n und max_n auf.
    Alle Zeilen der Klasse werden exakt einmal verwendet.
    """
    # Zufällig durchmischen, damit die Chunks wirklich zufällig sind
    shuffled = grp.sample(frac=1.0, random_state=rnd.randrange(10**9)).reset_index(drop=True)
    sizes = _random_chunk_sizes(len(shuffled), min_n, max_n, rnd)

    parts: List[pd.DataFrame] = []
    start = 0
    for n in sizes:
        end = start + n
        parts.append(shuffled.iloc[start:end].copy())
        start = end
    return parts


def split_all_data_into_subsets(
    df: pd.DataFrame,
    label_col: str,
    min_classes: int = 3,
    max_classes: int = 5,
    min_n: int = 20,
    max_n: int = 200,
    seed: int | None = None,
):
    """
    Teilt den gesamten Datensatz in zufällige Teilsets auf, ohne Überschneidungen:
    - Jede Klasse wird in zufällige Blöcke (20–200 Beispiele) zerlegt.
    - Diese Blöcke werden zu Teilsets gruppiert, wobei jedes Teilset 3–5 verschiedene Klassen enthält.
    Rückgabe: Liste von (subset_df, labels, counts)
    """
    rnd = random.Random(seed)

    # 1) Pro Klasse: in zufällige Blöcke zwischen min_n und max_n zerlegen
    chunks: List[Dict] = []
    for lab, grp in df.groupby(label_col):
        if len(grp) < min_n:
            raise ValueError(
                f"Klasse '{lab}' hat nur {len(grp)} Zeilen, minimal benötigt: {min_n}."
            )
        parts = _partition_class_rows(grp, min_n, max_n, rnd)
        for p in parts:
            chunks.append({"label": lab, "df": p, "count": len(p)})

    # 2) Blöcke zufällig mischen und zu Teilsets bündeln (3–5 Klassen pro Set, keine Klasse doppelt pro Set)
    rnd.shuffle(chunks)
    results = []
    while chunks:
        available_labels = {c["label"] for c in chunks}
        # Zielanzahl Klassen für dieses Set
        k_min = 1 if len(available_labels) < min_classes else min_classes
        k_max = min(max_classes, len(available_labels))
        k = rnd.randint(k_min, k_max)

        selected_labels = set()
        parts = []
        counts: Dict[str, int] = {}

        # Greedy: nimm den nächsten Chunk, dessen Klasse noch nicht im aktuellen Set ist
        i = 0
        while i < len(chunks) and len(selected_labels) < k:
            ch = chunks[i]
            lab = ch["label"]
            if lab not in selected_labels:
                parts.append(ch["df"])
                counts[str(lab)] = counts.get(str(lab), 0) + ch["count"]
                selected_labels.add(lab)
                # Entferne den Chunk aus der Liste
                chunks.pop(i)
            else:
                i += 1

        if not parts:
            # Falls wir nichts hinzufügen konnten (extremer Randfall), abbrechen
            raise RuntimeError("Konnte kein Teilset bilden. Prüfe Parameter und Daten.")

        subset_df = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=rnd.randrange(10**9))
            .reset_index(drop=True)
        )
        results.append((subset_df, sorted(selected_labels), counts))

    # Optional: Validierung, ob wirklich alle Zeilen verwendet wurden
    total_rows = sum(len(r[0]) for r in results)
    if total_rows != len(df):
        raise AssertionError(
            f"Nicht alle Daten wurden verwendet: {total_rows} von {len(df)}."
        )

    return results


def save_subsets_as_tsv(results, out_dir: str | Path, prefix: str = "subset"):
    """
    Speichert jedes Teilset als TSV-Datei. Gibt die Pfade zurück.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, (subset_df, labels, counts) in enumerate(results, start=1):
        p = out / f"{prefix}_{i:03d}.tsv"
        subset_df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    return paths