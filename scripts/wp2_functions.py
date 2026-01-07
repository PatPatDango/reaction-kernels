from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Optional, Dict
from collections import Counter
import hashlib
import networkx as nx
import numpy as np
from pathlib import Path


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


# ====================================================================
# WL-ITERATIONEN (einfaches 1-WL) + DRF über Iterationen (so simpel wie möglich)
# ====================================================================

def wl_label_sequence(
    G: nx.Graph,
    h: int,
    *,
    hash_node_labels: bool = True,
    digest_size: int = 16,
) -> List[Dict[Any, str]]:
    """
    Erzeuge die Sequenz L^0..L^h der WL-Knotenlabels.
    L^0(u) kommt aus get_node_label(G,u) (optional gehasht).
    L^{t+1}(u) = hash( L^t(u) | sort([L^t(v) for v in N(u)]) ).
    """
    # L0
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


def _phi_vertex_from_labels(
    G: nx.Graph,
    labels: Dict[Any, str],
    *,
    hash_features: bool = True,
    digest_size: int = 16,
) -> List[str]:
    feats = [labels[n] for n in G.nodes()]
    return [_hash_str(f"V|{x}", digest_size) for x in feats] if hash_features else feats


def _phi_edge_from_labels(
    G: nx.Graph,
    labels: Dict[Any, str],
    *,
    canonicalize: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> List[str]:
    feats: List[str] = []
    for u, v in G.edges():
        lu = labels[u]
        lv = labels[v]
        le = str(get_edge_label(G, u, v) or "")
        a, b = (sorted([lu, lv]) if canonicalize else (lu, lv))
        raw = f"{a}|{le}|{b}"
        feats.append(_hash_str(f"E|{raw}", digest_size) if hash_features else raw)
    return feats


def _path_label_from_labels(
    G: nx.Graph,
    path_nodes: List[Any],
    labels: Dict[Any, str],
    include_edge_labels: bool = True,
) -> str:
    tokens: List[str] = [labels[path_nodes[0]]]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if include_edge_labels:
            le = str(get_edge_label(G, u, v) or "")
            if le:
                tokens.append(le)
        tokens.append(labels[v])
    return "|".join(tokens)


def _phi_sp_from_labels(
    G: nx.Graph,
    labels: Dict[Any, str],
    *,
    include_edge_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> List[str]:
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


def drf_wl_features_from_graphs(
    educt_graph: nx.Graph,
    product_graph: nx.Graph,
    *,
    h: int = 2,
    mode: str = "edge",                  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,       # WL-intern: Labels hashen?
    hash_features: bool = True,          # Φ-Features hashen?
    digest_size: int = 16,
) -> Counter[str]:
    """
    DRF mit WL über h Iterationen:
      - L^0..L^h für Edukt & Produkt
      - pro i: Φ_i(E) und Φ_i(P)
      - pro i: symmetrische Multiset-Differenz
      - aufsummieren über i=0..h
    """
    ed_seq = wl_label_sequence(educt_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    pr_seq = wl_label_sequence(product_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

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
            fe = _phi_sp_from_labels(educt_graph, L_ed, include_edge_labels=include_edge_labels_in_sp,
                                     hash_features=hash_features, digest_size=digest_size)
            fp = _phi_sp_from_labels(product_graph, L_pr, include_edge_labels=include_edge_labels_in_sp,
                                     hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be one of: 'vertex', 'edge', 'sp'")

        total += multiset_symmetric_diff(fe, fp)

    return total


def drf_wl_features_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Counter[str]:
    from synkit.IO import rsmi_to_graph
    ed, pr = rsmi_to_graph(rsmi)
    return drf_wl_features_from_graphs(
        ed, pr,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )



def wl_feature_sets_per_iter_graph(
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
    """
    Liefert pro Iteration i=0..h die gehashten Feature-Sets Φ_i(G) (bzw. Multisets).
    Erfüllt: "hashed feature sets are returned at every iteration".
    """
    labels_seq = wl_label_sequence(G, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    out: List[Iterable[str]] = []
    for labels in labels_seq:
        if mode == "vertex":
            feats = _phi_vertex_from_labels(G, labels, hash_features=hash_features, digest_size=digest_size)
        elif mode == "edge":
            feats = _phi_edge_from_labels(G, labels, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
        elif mode in ("sp", "shortest_path", "shortest-path"):
            feats = _phi_sp_from_labels(G, labels, include_edge_labels=include_edge_labels_in_sp,
                                        hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be 'vertex', 'edge', or 'sp'")
        out.append(Counter(feats) if as_multiset else set(feats))
    return out


# ====================================================================
# DRF + WL: pro Iteration Differenz + optional Summe
# ====================================================================

def drf_wl_features_per_iter_from_graphs(
    educt_graph: nx.Graph,
    product_graph: nx.Graph,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Tuple[List[Counter[str]], Counter[str]]:
    """
    Pro Iteration i=0..h: DRF_i = symmetrische Multiset-Differenz von Φ_i(E) und Φ_i(P).
    Rückgabe: (Liste der DRF-Counter pro Iteration, Summe über alle Iterationen).
    """
    ed_labels = wl_label_sequence(educt_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    pr_labels = wl_label_sequence(product_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    per_iter: List[Counter[str]] = []
    total = Counter()

    for i in range(h + 1):
        L_ed, L_pr = ed_labels[i], pr_labels[i]
        if mode == "vertex":
            fe = _phi_vertex_from_labels(educt_graph, L_ed, hash_features=hash_features, digest_size=digest_size)
            fp = _phi_vertex_from_labels(product_graph, L_pr, hash_features=hash_features, digest_size=digest_size)
        elif mode == "edge":
            fe = _phi_edge_from_labels(educt_graph, L_ed, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
            fp = _phi_edge_from_labels(product_graph, L_pr, canonicalize=True, hash_features=hash_features, digest_size=digest_size)
        elif mode in ("sp", "shortest_path", "shortest-path"):
            fe = _phi_sp_from_labels(educt_graph, L_ed, include_edge_labels=include_edge_labels_in_sp,
                                     hash_features=hash_features, digest_size=digest_size)
            fp = _phi_sp_from_labels(product_graph, L_pr, include_edge_labels=include_edge_labels_in_sp,
                                     hash_features=hash_features, digest_size=digest_size)
        else:
            raise ValueError("mode must be 'vertex', 'edge', or 'sp'")

        diff_i = multiset_symmetric_diff(fe, fp)
        per_iter.append(diff_i)
        total += diff_i

    return per_iter, total


def drf_wl_features_per_iter_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Tuple[List[Counter[str]], Counter[str]]:
    from synkit.IO import rsmi_to_graph
    ed, pr = rsmi_to_graph(rsmi)
    return drf_wl_features_per_iter_from_graphs(
        ed, pr,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )


# ====================================================================
# Visualisierung der WL-Iterationen (DRF-basiert)
# ====================================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _split_direction_counts(ed_features: Iterable[str], pr_features: Iterable[str]) -> Tuple[Counter[str], Counter[str]]:
    ce, cp = Counter(ed_features), Counter(pr_features)
    more_e, more_p = Counter(), Counter()
    for k in set(ce) | set(cp):
        d = ce[k] - cp[k]
        if d > 0:
            more_e[k] = d
        elif d < 0:
            more_p[k] = -d
    return more_e, more_p

def _occ_vertex_from_labels(G: nx.Graph, labels: Dict[Any, str], *, hash_features: bool, digest_size: int) -> List[Tuple[Any, str]]:
    feats = []
    for n in G.nodes():
        f = labels[n]
        f = _hash_str(f"V|{f}", digest_size) if hash_features else f
        feats.append((n, f))
    return feats

def _occ_edge_from_labels(G: nx.Graph, labels: Dict[Any, str], *, hash_features: bool, digest_size: int, canonicalize: bool = True) -> List[Tuple[Tuple[Any, Any], str]]:
    occ = []
    for u, v in G.edges():
        lu = labels[u]; lv = labels[v]
        le = str(get_edge_label(G, u, v) or "")
        a, b = (sorted([lu, lv]) if canonicalize else (lu, lv))
        raw = f"{a}|{le}|{b}"
        f = _hash_str(f"E|{raw}", digest_size) if hash_features else raw
        occ.append(((u, v), f))
    return occ

def _path_label_from_labels_tokens(G: nx.Graph, path_nodes: List[Any], labels: Dict[Any, str], include_edge_labels: bool) -> str:
    tokens: List[str] = [labels[path_nodes[0]]]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if include_edge_labels:
            le = str(get_edge_label(G, u, v) or "")
            if le:
                tokens.append(le)
        tokens.append(labels[v])
    return "|".join(tokens)

def _occ_sp_from_labels(G: nx.Graph, labels: Dict[Any, str], *, include_edge_labels: bool, hash_features: bool, digest_size: int) -> List[Tuple[List[Tuple[Any, Any]], str]]:
    occ: List[Tuple[List[Tuple[Any, Any]], str]] = []
    all_paths = dict(nx.all_pairs_shortest_path(G))
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if v not in all_paths.get(u, {}):
                continue
            pn = all_paths[u][v]
            fwd = _path_label_from_labels_tokens(G, pn, labels, include_edge_labels)
            rev = _path_label_from_labels_tokens(G, list(reversed(pn)), labels, include_edge_labels)
            lab = min(fwd, rev)
            f = _hash_str(f"SP|{lab}", digest_size) if hash_features else lab
            edges = [(pn[k], pn[k+1]) for k in range(len(pn)-1)]
            occ.append((edges, f))
    return occ

def _mark_nodes(G: nx.Graph, occ: List[Tuple[Any, str]], more: Counter[str]) -> Dict[Any, bool]:
    flags = {n: False for n in G.nodes()}
    rem = more.copy()
    for n, feat in occ:
        if rem.get(feat, 0) > 0:
            flags[n] = True
            rem[feat] -= 1
    return flags

def _mark_edges(G: nx.Graph, occ: List[Tuple[Tuple[Any, Any], str]], more: Counter[str]) -> Dict[Tuple[Any, Any], bool]:
    flags = {e: False for e in G.edges()}
    rem = more.copy()
    for e, feat in occ:
        if rem.get(feat, 0) > 0:
            flags[e] = True
            rem[feat] -= 1
    return flags

def _mark_edges_by_paths(G: nx.Graph, occ: List[Tuple[List[Tuple[Any, Any]], str]], more: Counter[str]) -> Dict[Tuple[Any, Any], bool]:
    flags = {e: False for e in G.edges()}
    rem = more.copy()
    for edges, feat in occ:
        if rem.get(feat, 0) > 0:
            for e in edges:
                if e in flags:
                    flags[e] = True
                elif (e[1], e[0]) in flags:
                    flags[(e[1], e[0])] = True
            rem[feat] -= 1
    return flags

def _nx_traces_edges(G: nx.Graph, pos: Dict[Any, Tuple[float, float]], edge_flags: Dict[Tuple[Any, Any], bool], show_edge_labels: bool) -> List[go.Scatter]:
    # Nodes
    x_nodes, y_nodes, text_nodes = [], [], []
    for n, (x, y) in pos.items():
        x_nodes.append(x); y_nodes.append(y); text_nodes.append(get_node_label(G, n))
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text", text=text_nodes, textposition="top center",
        hovertext=text_nodes, hoverinfo="text", marker=dict(size=14), showlegend=False
    )
    # Edges
    x_c, y_c, x_s, y_s = [], [], [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if edge_flags.get((u, v), False):
            x_c += [x0, x1, None]; y_c += [y0, y1, None]
        else:
            x_s += [x0, x1, None]; y_s += [y0, y1, None]
    edge_same = go.Scatter(x=x_s, y=y_s, mode="lines", line=dict(color="#999", width=2), hoverinfo="none", showlegend=False)
    edge_changed = go.Scatter(x=x_c, y=y_c, mode="lines", line=dict(color="#d62728", width=3), hoverinfo="none", showlegend=False)

    traces = [edge_same, edge_changed, node_trace]

    if show_edge_labels and G.number_of_edges() > 0:
        x_lbl, y_lbl, t_lbl = [], [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            x_lbl.append((x0 + x1) / 2); y_lbl.append((y0 + y1) / 2)
            t_lbl.append(get_edge_label(G, u, v))
        traces.append(go.Scatter(x=x_lbl, y=y_lbl, mode="text", text=t_lbl, hoverinfo="none", showlegend=False))
    return traces

def _nx_traces_nodes(G: nx.Graph, pos: Dict[Any, Tuple[float, float]], node_flags: Dict[Any, bool]) -> List[go.Scatter]:
    # Edge layer (grau)
    x_e, y_e = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        x_e += [x0, x1, None]; y_e += [y0, y1, None]
    edge_trace = go.Scatter(x=x_e, y=y_e, mode="lines", line=dict(color="#bbb", width=2), hoverinfo="none", showlegend=False)
    # Nodes farbig
    x_nodes, y_nodes, text_nodes, colors = [], [], [], []
    for n, (x, y) in pos.items():
        x_nodes.append(x); y_nodes.append(y); text_nodes.append(get_node_label(G, n))
        colors.append("#d62728" if node_flags.get(n, False) else "#1f77b4")
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text", text=text_nodes, textposition="top center",
        hovertext=text_nodes, hoverinfo="text",
        marker=dict(size=16, color=colors),
        showlegend=False
    )
    return [edge_trace, node_trace]

def plot_wl_drf_iterations_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",                   # "edge" | "vertex" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    show_edge_labels: bool = True,
    seed: int = 42,
    title: Optional[str] = None,
    # Neu: Layout-Parameter für bessere Sichtbarkeit
    width: int = 1200,
    per_row_height: int = 320,            # Höhe pro Iteration/Reihe
    horizontal_spacing: float = 0.06,     # Abstand zwischen den beiden Spalten
    vertical_spacing: float = 0.06,       # Abstand zwischen Reihen
) -> go.Figure:
    """
    Visualisiert WL-Iterationen i=0..h: je Reihe i Edukt (links) und Produkt (rechts),
    markiert Elemente, die gemäß DRF_i 'mehr' auf der jeweiligen Seite sind.

    Layout-Tipp:
    - per_row_height steuert die Höhe pro Reihe; bei vielen Iterationen entsprechend erhöhen.
    - width steuert die Gesamtbreite; für mehr Platz je Spalte erhöhen.
    """
    from synkit.IO import rsmi_to_graph
    ed, pr = rsmi_to_graph(rsmi)

    # Positionen konstant pro Graph (für alle Iterationen)
    pos_ed = nx.spring_layout(ed, seed=seed)
    pos_pr = nx.spring_layout(pr, seed=seed)

    # Labels pro Iteration
    ed_seq = wl_label_sequence(ed, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    pr_seq = wl_label_sequence(pr, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    # Flache Liste von Subplot-Titeln
    subplot_titles = [t for i in range(h + 1) for t in (f"i={i} Educt", f"i={i} Product")]
    # Gesamthöhe abhängig von Iterationen
    height = max(500, (h + 1) * per_row_height)

    fig = make_subplots(
        rows=h + 1,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        column_widths=[0.5, 0.5],
        row_heights=[1] * (h + 1),
    )

    for i in range(h + 1):
        L_ed, L_pr = ed_seq[i], pr_seq[i]

        if mode == "edge":
            ed_occ = _occ_edge_from_labels(ed, L_ed, hash_features=hash_features, digest_size=digest_size, canonicalize=True)
            pr_occ = _occ_edge_from_labels(pr, L_pr, hash_features=hash_features, digest_size=digest_size, canonicalize=True)
            ed_feats = [f for _, f in ed_occ]; pr_feats = [f for _, f in pr_occ]
            more_ed, more_pr = _split_direction_counts(ed_feats, pr_feats)
            ed_flags = _mark_edges(ed, ed_occ, more_ed)
            pr_flags = _mark_edges(pr, pr_occ, more_pr)

            for tr in _nx_traces_edges(ed, pos_ed, ed_flags, show_edge_labels):
                fig.add_trace(tr, row=i + 1, col=1)
            for tr in _nx_traces_edges(pr, pos_pr, pr_flags, show_edge_labels):
                fig.add_trace(tr, row=i + 1, col=2)

        elif mode == "vertex":
            ed_occ = _occ_vertex_from_labels(ed, L_ed, hash_features=hash_features, digest_size=digest_size)
            pr_occ = _occ_vertex_from_labels(pr, L_pr, hash_features=hash_features, digest_size=digest_size)
            ed_feats = [f for _, f in ed_occ]; pr_feats = [f for _, f in pr_occ]
            more_ed, more_pr = _split_direction_counts(ed_feats, pr_feats)
            ed_node_flags = _mark_nodes(ed, ed_occ, more_ed)
            pr_node_flags = _mark_nodes(pr, pr_occ, more_pr)

            for tr in _nx_traces_nodes(ed, pos_ed, ed_node_flags):
                fig.add_trace(tr, row=i + 1, col=1)
            for tr in _nx_traces_nodes(pr, pos_pr, pr_node_flags):
                fig.add_trace(tr, row=i + 1, col=2)

        elif mode in ("sp", "shortest_path", "shortest-path"):
            ed_occ = _occ_sp_from_labels(ed, L_ed, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
            pr_occ = _occ_sp_from_labels(pr, L_pr, include_edge_labels=include_edge_labels_in_sp, hash_features=hash_features, digest_size=digest_size)
            ed_feats = [f for _, f in ed_occ]; pr_feats = [f for _, f in pr_occ]
            more_ed, more_pr = _split_direction_counts(ed_feats, pr_feats)
            ed_flags = _mark_edges_by_paths(ed, ed_occ, more_ed)
            pr_flags = _mark_edges_by_paths(pr, pr_occ, more_pr)

            for tr in _nx_traces_edges(ed, pos_ed, ed_flags, show_edge_labels=False):
                fig.add_trace(tr, row=i + 1, col=1)
            for tr in _nx_traces_edges(pr, pos_pr, pr_flags, show_edge_labels=False):
                fig.add_trace(tr, row=i + 1, col=2)
        else:
            raise ValueError("mode must be 'edge', 'vertex', or 'sp'")

    fig.update_layout(
        title=title or f"WL iterations DRF visualization (mode={mode}, h={h})",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        width=width,
        height=height,
    )

    # Achsen überall unsichtbar
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=False, zeroline=False, showticklabels=False)

    return fig

#----- kleiner Plot test 
def plot_wl_drf_feature_growth_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    title: Optional[str] = None,
    show_cumulative: bool = True,
) -> go.Figure:
    """
    Option 3 visualization:
    Plot how DRF feature complexity grows across WL iterations i=0..h.

    Metrics per iteration i:
    - unique_i = number of unique DRF features (len(Counter))
    - mass_i   = total DRF count (sum of Counter values)

    If show_cumulative=True:
    - cum_unique_i = #unique features in the union (sum) of DRF_0..DRF_i
    - cum_mass_i   = total count in DRF_0..DRF_i
    """
    # compute DRF per iteration
    per_iter, _total = drf_wl_features_per_iter_from_rsmi(
        rsmi,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )

    iters = list(range(h + 1))
    unique_per = [len(c) for c in per_iter]
    mass_per = [sum(c.values()) for c in per_iter]

    # cumulative stats
    cum_unique = []
    cum_mass = []
    if show_cumulative:
        acc = Counter()
        for c in per_iter:
            acc += c
            cum_unique.append(len(acc))
            cum_mass.append(sum(acc.values()))

    fig = go.Figure()

    # Per-iteration lines
    fig.add_trace(go.Scatter(
        x=iters, y=unique_per,
        mode="lines+markers",
        name="unique DRF features (per iter)"
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=mass_per,
        mode="lines+markers",
        name="total DRF count (per iter)"
    ))

    # Cumulative lines (optional)
    if show_cumulative:
        fig.add_trace(go.Scatter(
            x=iters, y=cum_unique,
            mode="lines+markers",
            name="unique DRF features (cumulative)",
            line=dict(dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=iters, y=cum_mass,
            mode="lines+markers",
            name="total DRF count (cumulative)",
            line=dict(dash="dash")
        ))

    fig.update_layout(
        title=title or f"WL-DRF feature growth (mode={mode}, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="count",
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
    )

    return fig


# ====================================================================
# Visualisierung der WL-Iterationen (ITS-basiert)
# ====================================================================

def its_wl_feature_sets_per_iter_from_graph(
    its_graph: nx.Graph,
    *,
    h: int = 2,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Tuple[List[Counter[str]], Counter[str]]:
    """
    Computes WL-enhanced feature representations for a single ITS graph.

    Returns:
      - per_iter: list of Counters (features -> counts) for i=0..h
      - total:    Counter = sum over all iterations (Multiset-union style via addition)

    If you need a true *set union* S_G, use: set(total.keys()).
    """
    # WL node-labels per iteration (dict[node] -> labelhash)
    labels_seq = wl_label_sequence(its_graph, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    per_iter: List[Counter[str]] = []
    total = Counter()

    for i in range(h + 1):
        L = labels_seq[i]

        # Apply Phi on the relabeled graph for this iteration
        if mode == "vertex":
            feats = _phi_vertex_from_labels(
                its_graph, L,
                hash_features=hash_features,
                digest_size=digest_size
            )
        elif mode == "edge":
            feats = _phi_edge_from_labels(
                its_graph, L,
                canonicalize=True,
                hash_features=hash_features,
                digest_size=digest_size
            )
        elif mode in ("sp", "shortest_path", "shortest-path"):
            feats = _phi_sp_from_labels(
                its_graph, L,
                include_edge_labels=include_edge_labels_in_sp,
                hash_features=hash_features,
                digest_size=digest_size
            )
        else:
            raise ValueError("mode must be 'vertex', 'edge', or 'sp'")

        # feats is a list[str] (can contain duplicates) -> Counter
        c = Counter(feats)
        per_iter.append(c)
        total += c

    return per_iter, total


def its_wl_feature_sets_per_iter_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> Tuple[List[Counter[str]], Counter[str]]:
    """
    Convenience wrapper: build ITS from reaction SMILES and compute WL feature sets.
    """
    from synkit.IO import rsmi_to_its
    its = rsmi_to_its(rsmi)
    return its_wl_feature_sets_per_iter_from_graph(
        its,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )


def its_final_hashset_SG_from_rsmi(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
) -> set[str]:
    """
    Returns the final hashset S_G = union of feature sets over all WL generations.
    (Set of feature hashes, no counts.)
    """
    per_iter, total = its_wl_feature_sets_per_iter_from_rsmi(
        rsmi,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )
    return set(total.keys())


def plot_its_wl_feature_growth_from_rsmi(
    rsmi: str,
    *,
    h: int = 3,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    show_cumulative: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Feature growth across WL iterations for ITS.

    Per iteration i:
      - unique_i = number of unique ITS-WL features in iteration i
      - mass_i   = total feature count (duplicates included) in iteration i

    If show_cumulative=True:
      - cum_unique_i = number of unique features across iterations 0..i (union size)
      - cum_mass_i   = total feature count across iterations 0..i
    """

    per_iter, total = its_wl_feature_sets_per_iter_from_rsmi(
        rsmi,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
    )

    iters = list(range(h + 1))
    unique_per = [len(c) for c in per_iter]
    mass_per = [sum(c.values()) for c in per_iter]

    cum_unique, cum_mass = [], []
    if show_cumulative:
        acc = Counter()
        for c in per_iter:
            acc += c
            cum_unique.append(len(acc))          # union size (unique)
            cum_mass.append(sum(acc.values()))   # total count

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iters, y=unique_per,
        mode="lines+markers",
        name="unique ITS-WL features (per iter)",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=mass_per,
        mode="lines+markers",
        name="total ITS-WL count (per iter)",
    ))

    if show_cumulative:
        fig.add_trace(go.Scatter(
            x=iters, y=cum_unique,
            mode="lines+markers",
            name="unique ITS-WL features (cumulative)",
            line=dict(dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=iters, y=cum_mass,
            mode="lines+markers",
            name="total ITS-WL count (cumulative)",
            line=dict(dash="dash"),
        ))

    fig.update_layout(
        title=title or f"ITS–WL Feature Growth Across Iterations (mode={mode}, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="count",
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
    )

    return fig

def plot_its_wl_feature_growth_subset(
    df,
    *,
    rxn_col: str = "clean_rxn",
    h: int = 3,
    mode: str = "edge",  # "vertex" | "edge" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plots mean ITS–WL feature growth across WL iterations, averaged over a dataset subset.
    """
    from wp2_functions import its_wl_feature_sets_per_iter_from_rsmi

    per_iter_counts = [[] for _ in range(h + 1)]

    for rsmi in df[rxn_col]:
        per_iter, _ = its_wl_feature_sets_per_iter_from_rsmi(
            rsmi,
            h=h,
            mode=mode,
            include_edge_labels_in_sp=include_edge_labels_in_sp,
            hash_node_labels=hash_node_labels,
            hash_features=hash_features,
            digest_size=digest_size,
        )
        for i, c in enumerate(per_iter):
            per_iter_counts[i].append(len(c))

    means = [np.mean(v) for v in per_iter_counts]
    stds = [np.std(v) for v in per_iter_counts]
    iters = list(range(h + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iters,
        y=means,
        mode="lines+markers",
        name="mean #ITS–WL features",
        error_y=dict(type="data", array=stds, visible=True),
    ))

    fig.update_layout(
        title=title or f"ITS–WL Feature Growth Across Iterations (subset mean, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="mean number of features",
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=True,
    )

    return fig

def plot_feature_growth_subset_its_vs_drf(
    df,
    *,
    rxn_col: str = "clean_rxn",
    h: int = 3,
    mode: str = "edge",  # use same mode for fair comparison
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    title: Optional[str] = None,
    show_errorbars: bool = True,
) -> go.Figure:
    """
    Compare feature growth across WL iterations for ITS–WL vs DRF–WL in ONE plot,
    averaged over all reactions in df.
    """
    from wp2_functions import (
        its_wl_feature_sets_per_iter_from_rsmi,
        drf_wl_features_per_iter_from_rsmi,
    )

    # collect per-iteration counts for each reaction
    its_counts = [[] for _ in range(h + 1)]
    drf_counts = [[] for _ in range(h + 1)]

    for rsmi in df[rxn_col].astype(str):
        # ITS–WL: per_iter is list[Counter]
        its_per_iter, _ = its_wl_feature_sets_per_iter_from_rsmi(
            rsmi,
            h=h,
            mode=mode,
            include_edge_labels_in_sp=include_edge_labels_in_sp,
            hash_node_labels=hash_node_labels,
            hash_features=hash_features,
            digest_size=digest_size,
        )
        for i, c in enumerate(its_per_iter):
            its_counts[i].append(len(c))  # unique features in iteration i

        # DRF–WL: per_iter is list[Counter]
        drf_per_iter, _total = drf_wl_features_per_iter_from_rsmi(
            rsmi,
            h=h,
            mode=mode,
            include_edge_labels_in_sp=include_edge_labels_in_sp,
            hash_node_labels=hash_node_labels,
            hash_features=hash_features,
            digest_size=digest_size,
        )
        for i, c in enumerate(drf_per_iter):
            drf_counts[i].append(len(c))  # unique DRF features in iteration i

    iters = list(range(h + 1))

    its_mean = [float(np.mean(v)) for v in its_counts]
    its_std  = [float(np.std(v))  for v in its_counts]
    drf_mean = [float(np.mean(v)) for v in drf_counts]
    drf_std  = [float(np.std(v))  for v in drf_counts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iters, y=its_mean,
        mode="lines+markers",
        name="ITS–WL (mean unique features / iter)",
        error_y=dict(type="data", array=its_std, visible=show_errorbars),
    ))

    fig.add_trace(go.Scatter(
        x=iters, y=drf_mean,
        mode="lines+markers",
        name="DRF–WL (mean unique features / iter)",
        error_y=dict(type="data", array=drf_std, visible=show_errorbars),
    ))

    fig.update_layout(
        title=title or f"ITS–WL vs DRF–WL Feature Growth (subset mean, mode={mode}, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="mean number of unique features",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
    )

    return fig
