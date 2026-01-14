# wp3_kernel.py

import numpy as np
from collections import Counter
from typing import Callable, Dict, List, Sequence, Any

####### KERNEL FUNCTIONS #######

def kernel_multiset_intersection(a: Counter[str], b: Counter[str]) -> int:
    """
    Multiset kernel: sum over features of min(count_a, count_b).
    count_a = how often feature appears in reaction a (f1: 2, f2: 3, f3:1)
    count_b = how often feature appears in reaction b (f1: 1, f2: 0, f3:4)
    Vergleich wie ähnlich sind 2 Reaktionen zueiandender? -> wie viele gleiche Dinge in beiden Reaktionen 
    """
   #leere Fall abfangen wenn beide Features Leer sind -> keine Unterschiede
    if not a or not b:
        return 0

    # iterate over smaller counter for speed
    #wenn a größer ist dann nach b iterieren 
    if len(a) > len(b):
        a, b = b, a

    s = 0 #zählen der gemeinsamkeiten 
    for k, ca in a.items(): #für k features die in a vorkommen
        #ob diese auch in b vorkommen
        cb = b.get(k, 0)
        if cb:  #wenn in b auch vorkommt
            s += min(ca, cb) #zählen der gemeinsamen Vorkommen (mindestens da beide Reaktionen das Feature haben)
    return s #summe der gemeinsamen Vorkommen zurückgeben // 0 = komplett unterschiedlich

def compute_kernel_matrix(
    X_feats: Sequence[Counter[str]],
    kernel_fn: Callable[[Counter[str], Counter[str]], int] = kernel_multiset_intersection,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute an NxN kernel matrix K where K[i,j] = kernel_fn(X[i], X[j]).
    Uses symmetry to compute only upper triangle.
    baut eine Tabelle Wie ähnlich ist Reaktion i zu Reaktion j?
    X_feats: Liste von Reaktionen mit ihren Features (Counter)
    kernel_fn: Funktion die die Ähnlichkeit zwischen 2 Reaktionen berechnet (hier Multiset Intersection) -> anpassbar 
    dtype: Datentyp für die Kernel-Matrix (Standard: float32 - Speicheroptimiert)
    """
    n = len(X_feats)    #Anzahl der Reaktionen
    K = np.zeros((n, n), dtype=dtype)  #Initialisierung der NxN Matrix mit Nullen -> nach der Anzahl der Reaktionen

    for i in range(n):  # Schleife über alle Reaktionen
        K[i, i] = kernel_fn(X_feats[i], X_feats[i]) #Diagonalelemente Reaktion mit sich selbst vergleichen -> maximale Ähnlichkeit
        for j in range(i + 1, n):  # Schleife über die oberen Dreieckselemente - i+1 da K[i,j] == K[j,i] -> nicht doppelt berechnen
            kij = kernel_fn(X_feats[i], X_feats[j]) #Berechnung der Ähnlichkeit zwischen Reaktion i und j über die kernel_fn Funktion (hier Multiset Intersection)
            K[i, j] = kij   #Eintragen des Ergebnisses in die Matrix
            K[j, i] = kij   #Symmetrie ausnutzen K[j,i] = K[i,j]
    return K    #Rückgabe der fertigen Kernel-Matrix

def kernel_matrix_stats(K: np.ndarray) -> Dict[str, float]:
    """
   Debugging stats for a kernel matrix. 
   Übersetzen der Kernselmatrix in einige nützliche Statistiken zur Analyse.
    """
    return {
        "n": float(K.shape[0]),     #Anzahl der Reaktionen
        "sym_max_abs": float(np.max(np.abs(K - K.T))), #K - K.T = Unterschied zwischen K[i,j] und K[j,i]
        "diag_min": float(np.min(np.diag(K))),  # Diagonale = K[i,i] --> Minimum der Selbstähnlichkeiten (darf nicht 0)
        "diag_max": float(np.max(np.diag(K))),  # Diagonale = K[i,i] --> Maximum der Selbstähnlichkeiten (gefühl für Skalierung)
        "nonzero_share": float((K > 0).mean()), #Anteil der Elemente in K die größer als 0 sind (zeigt wie viele Reaktionen überhaupt Gemeinsamkeiten haben) (0.05 → sehr sparsam (typisch DRF) // 0.6 → dichter Kernel (typisch ITS))
        "median": float(np.median(K)), #Median der Kernelwerte (zeigt die typische Ähnlichkeit zwischen Reaktionen)
        "mean": float(K.mean()),    #Durchschnitt der Kernelwerte (zeigt die durchschnittliche Ähnlichkeit zwischen Reaktionen) Größer = dichterer Kernel
        "max": float(K.max()),      #Maximaler Kernelwert (zeigt die höchste Ähnlichkeit zwischen zwei verschiedenen Reaktionen)
    }


