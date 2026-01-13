from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Optional, Dict
from collections import Counter
import hashlib
import networkx as nx
from synkit.IO import rsmi_to_graph


# ------------------------------------------------------------
# Labels und Hashing (Core)
# ------------------------------------------------------------

EDGE_LABEL_KEYS = ("label", "bond", "type", "order", "change")


def get_node_label(G: nx.Graph, n: Any) -> str:
    d = G.nodes[n]
    element = str(d.get("element", n))
    charge = int(d.get("charge", 0)) if d.get("charge", 0) is not None else 0
    hcount = int(d.get("hcount", 0)) if d.get("hcount", 0) is not None else 0
    aromatic = "ar" if d.get("aromatic", False) else "al"
    return f"{element}|c{charge}|h{hcount}|{aromatic}"


def get_edge_label(G: nx.Graph, u: Any, v: Any) -> str:
    d = G.edges[u, v]
    for k in EDGE_LABEL_KEYS:
        if k in d and d[k] is not None:
            return str(d[k])
    return ""


def _hash_str(x: str, digest_size: int = 16) -> str:
    return hashlib.blake2b(x.encode("utf-8"), digest_size=digest_size).hexdigest()


# ------------------------------------------------------------
# RSmi → Graphs
# ------------------------------------------------------------

def rsmi_to_educt_product(rsmi: str) -> tuple[nx.Graph, nx.Graph]:
    """
    Gibt (educt_graph, product_graph) zurück, robust gegen 2+ Outputs von synkit.IO.rsmi_to_graph
    """
    out = rsmi_to_graph(rsmi, drop_non_aam=False, use_index_as_atom_map=False)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    raise TypeError("rsmi_to_graph must return at least (educt, product)")


# ------------------------------------------------------------
# Φ-Funktionen (Basis-Features)
# ------------------------------------------------------------

def phi_vertex_list(
    G: nx.Graph,
    *,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> List[str]:
    labels = [get_node_label(G, n) for n in G.nodes]
    if raw_labels:
        return labels
    if hash_labels:
        return [_hash_str(lbl, digest_size) for lbl in labels]
    return labels


def phi_edge_list(
    G: nx.Graph,
    *,
    canonicalize: bool = True,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> List[str]:
    feats: List[str] = []
    for u, v in G.edges():
        lu = get_node_label(G, u)
        lv = get_node_label(G, v)
        le = get_edge_label(G, u, v)
        a, b = (sorted([lu, lv]) if canonicalize else (lu, lv))
        triplet = f"{a}|{le}|{b}"
        feats.append(triplet if (raw_labels or not hash_labels) else _hash_str(triplet, digest_size))
    return feats


def _path_label(G: nx.Graph, path_nodes: List[Any], include_edge_labels: bool = True) -> str:
    tokens: List[str] = [get_node_label(G, path_nodes[0])]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if include_edge_labels:
            le = get_edge_label(G, u, v)
            if le:
                tokens.append(le)
        tokens.append(get_node_label(G, v))
    return "|".join(tokens)


def phi_shortest_path_list(
    G: nx.Graph,
    *,
    include_edge_labels: bool = True,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> List[str]:
    feats: List[str] = []
    all_paths: Dict[Any, Dict[Any, List[Any]]] = dict(nx.all_pairs_shortest_path(G))
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            pn: Optional[List[Any]] = None
            if v in all_paths.get(u, {}):
                pn = all_paths[u][v]
            elif u in all_paths.get(v, {}):
                pn = list(reversed(all_paths[v][u]))
            if not pn:
                continue
            fwd = _path_label(G, pn, include_edge_labels=include_edge_labels)
            rev = _path_label(G, list(reversed(pn)), include_edge_labels=include_edge_labels)
            lab = min(fwd, rev)
            feats.append(lab if (raw_labels or not hash_labels) else _hash_str(lab, digest_size))
    return feats


# ------------------------------------------------------------
# Multiset-symmetrische Differenz
# ------------------------------------------------------------

def multiset_symmetric_diff(a: Iterable[str], b: Iterable[str]) -> Counter[str]:
    ca, cb = Counter(a), Counter(b)
    out = Counter()
    for k in set(ca) | set(cb):
        diff = abs(ca[k] - cb[k])
        if diff:
            out[k] = diff
    return out


# ------------------------------------------------------------
# DRF: direkt (rsmi oder Graphpaare)
# ------------------------------------------------------------

def drf(
    *,
    rsmi: Optional[str] = None,
    educt_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    mode: str = "edge",                 # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> Counter[str]:
    """
    Berechnet DRF-Features. Entweder rsmi angeben ODER educt_graph + product_graph.
    """
    if rsmi is not None:
        educt_graph, product_graph = rsmi_to_educt_product(rsmi)
    if educt_graph is None or product_graph is None:
        raise ValueError("Either provide rsmi or both educt_graph and product_graph.")

    if mode == "vertex":
        fe = phi_vertex_list(educt_graph, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_vertex_list(product_graph, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    elif mode == "edge":
        fe = phi_edge_list(educt_graph, canonicalize=True, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_edge_list(product_graph, canonicalize=True, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    elif mode in ("sp", "shortest_path", "shortest-path"):
        fe = phi_shortest_path_list(educt_graph, include_edge_labels=include_edge_labels_in_sp, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_shortest_path_list(product_graph, include_edge_labels=include_edge_labels_in_sp, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    else:
        raise ValueError("mode must be one of: 'vertex', 'edge', 'sp'")

    return multiset_symmetric_diff(fe, fp)


# ------------------------------------------------------------
# WL-Iteration + DRF (flach, optional per-iteration)
# ------------------------------------------------------------

def wl_label_sequence(
    G: nx.Graph,
    h: int,
    *,
    hash_node_labels: bool = True,
    digest_size: int = 16,
) -> List[Dict[Any, str]]:
    L0: Dict[Any, str] = {n: str(get_node_label(G, n)) for n in G.nodes()}
    if hash_node_labels:
        L0 = {n: _hash_str(lbl, digest_size) for n, lbl in L0.items()}
    seq: List[Dict[Any, str]] = [L0]

    prev = L0
    for _ in range(h):
        new: Dict[Any, str] = {}
        for u in G.nodes():
            neigh = [prev[v] for v in G.neighbors(u)]
            neigh.sort()
            raw = prev[u] + "|" + "#".join(neigh)
            new[u] = _hash_str(raw, digest_size) if hash_node_labels else raw
        seq.append(new)
        prev = new
    return seq


def _phi_vertex_from_labels(G: nx.Graph, labels: Dict[Any, str], *, hash_features: bool = True, digest_size: int = 16) -> List[str]:
    feats = [labels[n] for n in G.nodes()]
    return [_hash_str(f"V|{x}", digest_size) for x in feats] if hash_features else feats


def _phi_edge_from_labels(G: nx.Graph, labels: Dict[Any, str], *, canonicalize: bool = True, hash_features: bool = True, digest_size: int = 16) -> List[str]:
    feats: List[str] = []
    for u, v in G.edges():
        lu = labels[u]; lv = labels[v]
        le = str(get_edge_label(G, u, v) or "")
        a, b = (sorted([lu, lv]) if canonicalize else (lu, lv))
        raw = f"{a}|{le}|{b}"
        feats.append(_hash_str(f"E|{raw}", digest_size) if hash_features else raw)
    return feats


def _path_label_from_labels(G: nx.Graph, path_nodes: List[Any], labels: Dict[Any, str], include_edge_labels: bool = True) -> str:
    tokens: List[str] = [labels[path_nodes[0]]]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if include_edge_labels:
            le = str(get_edge_label(G, u, v) or "")
            if le:
                tokens.append(le)
        tokens.append(labels[v])
    return "|".join(tokens)


def _phi_sp_from_labels(G: nx.Graph, labels: Dict[Any, str], *, include_edge_labels: bool = True, hash_features: bool = True, digest_size: int = 16) -> List[str]:
    feats: List[str] = []
    all_paths = dict(nx.all_pairs_shortest_path(G))
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if v not in all_paths.get(u, {}):
                continue
            pn = all_paths[u][v]
            fwd = _path_label_from_labels(G, pn, labels, include_edge_labels)
            rev = _path_label_from_labels(G, list(reversed(pn)), labels, include_edge_labels)
            lab = min(fwd, rev)
            feats.append(_hash_str(f"SP|{lab}", digest_size) if hash_features else lab)
    return feats


def wl_feature_sets_per_iter(
    G: nx.Graph,
    h: int,
    *,
    mode: str = "edge",                # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    as_multiset: bool = False,         # True -> Counter statt Set
) -> List[Iterable[str]]:
    labels_seq = wl_label_sequence(G, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    out: List[Iterable[str]] = []
    for labels in labels_seq:
        if mode == "vertex":
            feats = _phi_vertex_from_labels(G, labels, hash_features=hash_features, digest_size=digest_size)
        elif mode == "edge":
            feats = _phi_edge_from_labels(G, labels, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
        elif mode in ("sp", "shortest_path", "shortest-path"):
            feats = _phi_sp_from_labels(G, labels, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be 'vertex', 'edge', or 'sp'")
        out.append(Counter(feats) if as_multiset else set(feats))
    return out


def drf_wl(
    *,
    rsmi: Optional[str] = None,
    educt_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    return_per_iter: bool = False,
) -> Counter[str] | Tuple[List[Counter[str]], Counter[str]]:
    """
    DRF mit WL über h Iterationen.
    - return_per_iter=False -> nur Summe (Counter)
    - return_per_iter=True  -> (Liste pro i, Summe)
    """
    if rsmi is not None:
        educt_graph, product_graph = rsmi_to_educt_product(rsmi)
    if educt_graph is None or product_graph is None:
        raise ValueError("Either provide rsmi or both educt_graph and product_graph.")

    ed_seq = wl_label_sequence(educt_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    pr_seq = wl_label_sequence(product_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    per_iter: List[Counter[str]] = []
    total = Counter()

    for i in range(h + 1):
        L_ed = ed_seq[i]
        L_pr = pr_seq[i]

        if mode == "vertex":
            fe = _phi_vertex_from_labels(educt_graph, L_ed, hash_features=hash_features, digest_size=digest_size)
            fp = _phi_vertex_from_labels(product_graph, L_pr, hash_features=hash_features, digest_size=digest_size)
        elif mode == "edge":
            fe = _phi_edge_from_labels(educt_graph, L_ed, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
            fp = _phi_edge_from_labels(product_graph, L_pr, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
        elif mode in ("sp", "shortest_path", "shortest-path"):
            fe = _phi_sp_from_labels(educt_graph, L_ed, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
            fp = _phi_sp_from_labels(product_graph, L_pr, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be one of: 'vertex', 'edge', 'sp'")

        diff_i = multiset_symmetric_diff(fe, fp)
        per_iter.append(diff_i)
        total += diff_i

    return (per_iter, total) if return_per_iter else total


# ------------------------------------------------------------
# ITS – WL Feature Sets (einfacher Kern)
# ------------------------------------------------------------

def its_wl_feature_sets(
    *,
    rsmi: Optional[str] = None,
    its_graph: Optional[nx.Graph] = None,
    h: int = 2,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Tuple[List[Counter[str]], Counter[str]]:
    """
    Liefert (per_iter, total) für ITS–WL.
    Entweder rsmi angeben ODER its_graph.
    """
    if its_graph is None:
        if rsmi is None:
            raise ValueError("Provide rsmi or its_graph")
        from synkit.IO import rsmi_to_its
        its_graph = rsmi_to_its(rsmi, drop_non_aam=False, use_index_as_atom_map=False)

    labels_seq = wl_label_sequence(its_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    per_iter: List[Counter[str]] = []
    total = Counter()

    for i in range(h + 1):
        L = labels_seq[i]
        if mode == "vertex":
            feats = _phi_vertex_from_labels(its_graph, L, hash_features=hash_features, digest_size=digest_size)
        elif mode == "edge":
            feats = _phi_edge_from_labels(its_graph, L, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
        elif mode in ("sp", "shortest_path", "shortest-path"):
            feats = _phi_sp_from_labels(its_graph, L, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be 'vertex', 'edge', or 'sp'")
        c = Counter(feats)
        per_iter.append(c)
        total += c

    return per_iter, total