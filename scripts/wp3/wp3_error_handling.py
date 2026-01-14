# wp3_error_handling.py
from __future__ import annotations

from pathlib import Path
import pickle


def debug_pkl_basic(precomp_dir: str | Path, feature_key: str = "drf_wl"):
    """
    Funktion 1:
    - nimmt einen Ordner
    - lädt die erste .pkl Datei
    - printet Keys, n_errors, first error, first feature
    """
    precomp_dir = Path(precomp_dir)
    pkl = next(precomp_dir.glob("*.pkl"))  # next() ist hier richtig
    obj = pickle.load(open(pkl, "rb"))

    print("Inspecting:", pkl)
    print("Keys:", obj.keys())
    print("n_errors:", obj["meta"]["n_errors"])
    print("First error:", obj["errors"][:1])
    print("First feature:", type(obj[feature_key][0]), obj[feature_key][0])


def debug_pkl_empties(precomp_dir: str | Path, feature_key: str = "drf_wl"):
    """
    Funktion 2:
    - lädt die erste .pkl Datei (sorted)
    - printet Meta + wie viele Counters leer sind
    - sucht das erste nicht-leere Beispiel und zeigt ein Sample
    """
    precomp_dir = Path(precomp_dir)
    pkl = sorted(precomp_dir.glob("*.pkl"))[0]
    print("Inspecting:", pkl)

    with open(pkl, "rb") as f:
        obj = pickle.load(f)

    print("Keys:", obj.keys())
    print("Meta n_rows:", obj["meta"]["n_rows"])
    print("Meta n_errors:", obj["meta"]["n_errors"])
    print("First error (if any):", obj["errors"][:1])

    X = obj[feature_key]
    empty = sum(1 for c in X if len(c) == 0)
    print("Empty counters:", empty, "/", len(X))

    for i, c in enumerate(X):
        if len(c) > 0:
            print("First non-empty at idx:", i, "items:", len(c), "total:", sum(c.values()))
            print("Sample:", list(c.items())[:5])
            break
    else:
        print("ALL COUNTERS ARE EMPTY in this PKL.")


def debug_dir_summary(precomp_dir: str | Path, feature_key: str = "drf_wl"):
    """
    Funktion 3:
    - listet alle .pkl Dateien
    - lädt die erste
    - printet n_errors + empty ratio + total count von feature[0]
    """
    precomp_dir = Path(precomp_dir)
    pkls = list(precomp_dir.glob("*.pkl"))
    print("Found PKLs:", len(pkls))

    pkl = pkls[0]
    obj = pickle.load(open(pkl, "rb"))

    print("Inspecting:", pkl)
    print("n_errors:", obj["meta"]["n_errors"])
    print("empty:", sum(1 for c in obj[feature_key] if len(c) == 0), "/", len(obj[feature_key]))
    print("example total count:", sum(obj[feature_key][0].values()))

def debug_find_nonzero_kernel_pair(X, kernel_fn):
    """
    Findet das erste Paar (i, j) mit:
    - X[i] und X[j] nicht leer
    - kernel_fn(X[i], X[j]) > 0

    Gibt (i, j, kernel_value) aus oder meldet, wenn nichts gefunden wurde.
    """
    n = len(X)

    for i in range(n):
        if len(X[i]) == 0:
            continue
        for j in range(i + 1, n):
            if len(X[j]) == 0:
                continue
            k = kernel_fn(X[i], X[j])
            if k > 0:
                print("Found non-zero kernel at:", i, j, "value:", k)
                return i, j, k

    print("No non-zero kernel pair found.")
    return None

def debug_find_nonempty_pair(X):
    """
    Findet das erste Paar (i, j) mit:
    - X[i] nicht leer
    - X[j] nicht leer

    Gibt (i, j) zurück oder None.
    """
    n = len(X)

    for i in range(n):
        if len(X[i]) == 0:
            continue
        for j in range(i + 1, n):
            if len(X[j]) > 0:
                print("Found non-empty pair at:", i, j)
                print("Sizes:", len(X[i]), len(X[j]))
                return i, j

    print("No non-empty pair found.")
    return None