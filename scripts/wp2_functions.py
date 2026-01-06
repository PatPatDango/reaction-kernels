from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Optional, Dict
from collections import Counter
import hashlib
import networkx as nx


# ------------------------------------------------------------
# Robuste Label-Getter (gleichen unterschiedliche Attributnamen ab)
# Passe die Kandidaten-Keys bei Bedarf an deine Graph-Attribute an.
# ------------------------------------------------------------

NODE_LABEL_KEYS = ("label", "atom", "symbol", "element", "name")
EDGE_LABEL_KEYS = ("label", "bond", "type", "order", "change")

def get_node_label(G: nx.Graph, n: Any) -> str:
    d = G.nodes[n]
    for k in NODE_LABEL_KEYS:
        if k in d and d[k] is not None:
            return str(d[k])
    return str(n)  # Fallback auf Node-ID

def get_edge_label(G: nx.Graph, u: Any, v: Any) -> str:
    d = G.edges[u, v]
    for k in EDGE_LABEL_KEYS:
        if k in d and d[k] is not None:
            return str(d[k])
    return ""  # Fallback: kein Bond-Label gefunden


# ------------------------------------------------------------
# Hashing (blake2b). Wir geben Strings zurück, können aber Counters auf ihnen bilden.
# ------------------------------------------------------------

def _hash_str(x: str, digest_size: int = 16) -> str:
    return hashlib.blake2b(x.encode("utf-8"), digest_size=digest_size).hexdigest()


# ------------------------------------------------------------
# Φ_V: Vertex-Labels als Liste (mit Duplikaten)
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


# ------------------------------------------------------------
# Φ_E: Edge-Labels als Tripel l(u)-l(uv)-l(v), kanonisiert, als Liste
# ------------------------------------------------------------

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
        le = get_edge_label(G, u, v)  # kann "" sein, falls kein Bond-Label vorhanden

        if canonicalize:
            a, b = sorted([lu, lv])
        else:
            a, b = lu, lv

        triplet = f"{a}|{le}|{b}"
        if raw_labels:
            feats.append(triplet)
        else:
            feats.append(_hash_str(triplet, digest_size) if hash_labels else triplet)
    return feats


# ------------------------------------------------------------
# Φ_SP: Shortest-Path-Labels als Sequenz (mit Duplikaten), Liste
# - ungewichtete kürzeste Pfade
# - Richtungskanonisierung: min(vorwärts, rückwärts)
# ------------------------------------------------------------

def _path_label(
    G: nx.Graph,
    path_nodes: List[Any],
    include_edge_labels: bool = True,
) -> str:
    tokens: List[str] = []
    tokens.append(get_node_label(G, path_nodes[0]))
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
    # Alle kürzesten Pfade (unweighted)
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
            lab = min(fwd, rev)  # Richtungskanonisierung

            if raw_labels:
                feats.append(lab)
            else:
                feats.append(_hash_str(lab, digest_size) if hash_labels else lab)
    return feats


# ------------------------------------------------------------
# DRF als Multiset-Symmetrische-Differenz
# Wir geben einen Counter zurück (Feature -> Count-Differenz)
# ------------------------------------------------------------

def multiset_symmetric_diff(a: Iterable[str], b: Iterable[str]) -> Counter[str]:
    ca, cb = Counter(a), Counter(b)
    out = Counter()
    for k in set(ca) | set(cb):
        diff = abs(ca[k] - cb[k])
        if diff:
            out[k] = diff
    return out

def drf_features_from_graphs(
    educt_graph: nx.Graph,
    product_graph: nx.Graph,
    *,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> Counter[str]:
    if mode == "vertex":
        fe = phi_vertex_list(educt_graph, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_vertex_list(product_graph, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    elif mode == "edge":
        fe = phi_edge_list(educt_graph, canonicalize=True, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_edge_list(product_graph, canonicalize=True, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    elif mode in ("sp", "shortest_path", "shortest-path"):
        fe = phi_shortest_path_list(educt_graph, include_edge_labels=include_edge_labels_in_sp,
                                    hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        fp = phi_shortest_path_list(product_graph, include_edge_labels=include_edge_labels_in_sp,
                                    hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    else:
        raise ValueError("mode must be one of: 'vertex', 'edge', 'sp'")

    return multiset_symmetric_diff(fe, fp)

def drf_features_from_rsmi(
    rsmi: str,
    *,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_labels: bool = True,
    digest_size: int = 16,
    raw_labels: bool = False,
) -> Counter[str]:
    from synkit.IO import rsmi_to_graph
    ed, pr = rsmi_to_graph(rsmi)
    return drf_features_from_graphs(
        ed, pr,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_labels=hash_labels,
        digest_size=digest_size,
        raw_labels=raw_labels,
    )


# ------------------------------------------------------------
# Debug-Helfer: prüft, welche Attribute vorhanden sind
# ------------------------------------------------------------

def debug_print_graph_attrs(G: nx.Graph, title: str = "Graph", max_items: int = 5) -> None:
    print(f"== {title} ==")
    print(f"nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")
    for n in list(G.nodes)[:max_items]:
        print(" node", n, "attrs:", dict(G.nodes[n]))
    for u, v in list(G.edges)[:max_items]:
        print(" edge", (u, v), "attrs:", dict(G.edges[u, v]))