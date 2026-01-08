# src/wp3_kernel.py

from typing import Dict

# Multiset kernel implementation for reaction features
def kernel_multiset_intersection(
    c1: Dict[str, int],
    c2: Dict[str, int],
) -> int:
    """
    Multiset kernel as specified in the lab.

    k(G, H) = sum_f min(c_G(f), c_H(f))

    where c_G and c_H are multisets (Counters) of hashed features.
    """
    common = set(c1.keys()) & set(c2.keys())
    return sum(min(c1[f], c2[f]) for f in common)

def kernel_set_intersection(
    c1: Dict[str, int],
    c2: Dict[str, int],
) -> int:
    """
    Set-based kernel (baseline).

    k(G, H) = |S_G ∩ S_H|
    """
    return len(set(c1.keys()) & set(c2.keys()))