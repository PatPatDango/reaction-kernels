from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Optional, Dict
from collections import Counter

import numpy as np
import networkx as nx

import plotly.io as pio
pio.renderers.default = "vscode"
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# synkit turns a reaction SMILES string into NetworkX graphs
from synkit.IO import rsmi_to_its, rsmi_to_graph

#local imports
from .wp0.chem_graph_handling import visualize_graph

from .wp2.wp2_functions import (
    get_node_label, get_edge_label, _hash_str,
    _path_label_from_labels, wl_label_sequence, rsmi_to_educt_product,
    drf_wl, its_wl_feature_sets, phi_vertex_list, phi_edge_list, phi_shortest_path_list
)



# ----------------------- Plot-Helpers -----------------------
# ------------------------WP1 Helpers-------------------------

def _get_node_label(G: nx.Graph, n: Any) -> str:
    """
    NetworkX nodes have attribute dicts: G.nodes[n] -> dict.

    Depending on the graph builder, the atom label may be stored under
    different keys (e.g., 'element', 'atom', 'label', ...).

    We try common keys and return the first that exists.
    If none exists, we return the node id itself.
    """
    d = G.nodes[n]
    for key in ("label", "atom", "symbol", "element", "name"):
        if key in d and d[key] is not None:
            return str(d[key])
    return str(n)


def _get_edge_label(G: nx.Graph, u: Any, v: Any) -> str:
    """
    NetworkX edges also have attribute dicts: G.edges[u, v] -> dict.

    For ITS graphs, an edge label often encodes reaction changes.
    We try common keys and return the first match.
    If none exists, return empty string.
    """
    d = G.edges[u, v]
    for key in ("label", "bond", "type", "order", "change"):
        if key in d and d[key] is not None:
            return str(d[key])
    return ""


def _safe_spring_layout(G: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    """
    Graphs have no positions by default. We compute positions with a
    spring layout (force-directed).

    seed=42 makes plots reproducible (same graph -> similar layout).
    """
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, seed=42)

# ----------------------- WP2 Plot Helpers -----------------------
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

def _build_occurrences_vertex(G: nx.Graph, *, hash_labels: bool, digest_size: int, raw_labels: bool) -> List[Tuple[Any, str]]:
    feats = phi_vertex_list(G, hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
    # Wir müssen die Zuordnung Node -> Feature auflösen: wir berechnen pro Node das Feature einzeln
    occ: List[Tuple[Any, str]] = []
    for n in G.nodes:
        single = phi_vertex_list(G.subgraph([n]).copy(), hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
        # single enthält genau 1 Feature
        occ.append((n, single[0]))
    # Fallback: wenn Subgraph-Ansatz nicht passt, direkt Node-Label hashen
    if len(occ) != G.number_of_nodes():
        occ = []
        for n in G.nodes:
            lbls = phi_vertex_list(G.subgraph([n]).copy(), hash_labels=hash_labels, digest_size=digest_size, raw_labels=raw_labels)
            occ.append((n, lbls[0] if lbls else ""))
    return occ


def _build_occurrences_edge(G: nx.Graph, *, hash_labels: bool, digest_size: int, raw_labels: bool) -> List[Tuple[Tuple[Any, Any], str]]:
    # Roh-Triplet pro Kante
    raw = phi_edge_list(G, canonicalize=True, hash_labels=False, raw_labels=True)
    # Hash ggf. anwenden (genau wie im DRF)
    if raw_labels:
        features = raw
    elif hash_labels:
        import hashlib
        def H(x: str) -> str:
            return hashlib.blake2b(x.encode("utf-8"), digest_size=digest_size).hexdigest()
        features = [H(x) for x in raw]
    else:
        features = raw

    occ: List[Tuple[Tuple[Any, Any], str]] = []
    for (u, v), feat in zip(G.edges(), features):
        occ.append(((u, v), feat))
    return occ


def _build_occurrences_sp(G: nx.Graph, *, include_edge_labels: bool, hash_labels: bool, digest_size: int, raw_labels: bool) -> List[Tuple[List[Tuple[Any, Any]], str]]:
    # Alle kürzesten Pfade (unweighted)
    all_paths = dict(nx.all_pairs_shortest_path(G))
    nodes = list(G.nodes)
    occ: List[Tuple[List[Tuple[Any, Any]], str]] = []

    import hashlib
    def H(x: str) -> str:
        return hashlib.blake2b(x.encode("utf-8"), digest_size=digest_size).hexdigest()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if v not in all_paths.get(u, {}):
                continue
            pn = all_paths[u][v]
            fwd = _path_label(G, pn, include_edge_labels)
            rev = _path_label(G, list(reversed(pn)), include_edge_labels)
            lab = min(fwd, rev)  # Richtungskanonisierung

            if raw_labels:
                feat = lab
            elif hash_labels:
                feat = H(lab)
            else:
                feat = lab

            # Kantenliste des Pfades
            edges = [(pn[k], pn[k+1]) for k in range(len(pn)-1)]
            occ.append((edges, feat))
    return occ


def _split_direction(
    ed_features: Iterable[str],
    pr_features: Iterable[str],
) -> Tuple[Counter[str], Counter[str]]:
    """Ermittle, welche Features 'mehr' im Edukt bzw. Produkt sind."""
    ce = Counter(ed_features)
    cp = Counter(pr_features)
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
            fwd = _path_label_from_labels(G, pn, labels, include_edge_labels)
            rev = _path_label_from_labels(G, list(reversed(pn)), labels, include_edge_labels)
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



def _nx_to_traces_edges(G: nx.Graph, pos: Dict[Any, Tuple[float, float]], edge_flags: Dict[Tuple[Any, Any], bool], title: str, show_edge_labels: bool) -> List[go.Scatter]:
    # Nodes
    x_nodes, y_nodes, text_nodes = [], [], []
    for n, (x, y) in pos.items():
        x_nodes.append(x); y_nodes.append(y); text_nodes.append(get_node_label(G, n))
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text", text=text_nodes, textposition="top center",
        hovertext=text_nodes, hoverinfo="text", marker=dict(size=14), name=f"{title} nodes", showlegend=False
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


def _nx_to_traces_nodes(G: nx.Graph, pos: Dict[Any, Tuple[float, float]], node_flags: Dict[Any, bool], title: str) -> List[go.Scatter]:
    x_nodes, y_nodes, text_nodes, colors = [], [], [], []
    for n, (x, y) in pos.items():
        x_nodes.append(x); y_nodes.append(y); text_nodes.append(get_node_label(G, n))
        colors.append("#d62728" if node_flags.get(n, False) else "#1f77b4")
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text", text=text_nodes, textposition="top center",
        hovertext=text_nodes, hoverinfo="text",
        marker=dict(size=16, color=colors),
        name=f"{title} nodes", showlegend=False
    )
    # edges grau
    x_e, y_e = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        x_e += [x0, x1, None]; y_e += [y0, y1, None]
    edge_trace = go.Scatter(x=x_e, y=y_e, mode="lines", line=dict(color="#bbb", width=2), hoverinfo="none", showlegend=False)
    return [edge_trace, node_trace]


# ----------------------- Plots -----------------------

# ----------------------- WP1 Plots -----------------------

def visualize_reaction(
    rsmi: str,
    which: str = "its",
):
    """
    Unified visualization entry point.

    - ITS: uses chem_graph_handling.visualize_graph (Synkit ITS visualizer)
    - educt/product: uses Plotly NetworkX visualization
    """

    # ---------- ITS ----------
    if which == "its":
        # expects a STRING, not a graph
        visualize_graph(rsmi)
        return

    # ---------- EDUCT / PRODUCT ----------
    educt_graph, product_graph = rsmi_to_graph(rsmi)

    if which == "educt":
        fig = plot_nx_graph(educt_graph, title="Educt graph")
        fig.show()
        return

    if which == "product":
        fig = plot_nx_graph(product_graph, title="Product graph")
        fig.show()
        return

    raise ValueError("which must be 'its', 'educt', or 'product'")

def plot_nx_graph(
    G: nx.Graph,
    title: str = "Graph",
    show_edge_labels: bool = True,
    node_size: int = 16,
) -> go.Figure:
    """
    Plot a NetworkX graph with Plotly:
    - nodes as markers with text labels
    - edges as line segments
    - optional edge labels at edge midpoints

    Returns a Plotly Figure (you can call fig.show()).
    """
    pos = _safe_spring_layout(G)

    # ---- nodes ----
    x_nodes, y_nodes, text_nodes = [], [], []
    for n, (x, y) in pos.items():
        x_nodes.append(x)
        y_nodes.append(y)
        text_nodes.append(_get_node_label(G, n))

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=text_nodes,
        textposition="top center",
        hovertext=text_nodes,
        hoverinfo="text",
        marker=dict(size=node_size),
    )

    # ---- edges ----
    x_edges, y_edges = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        x_edges += [x0, x1, None]
        y_edges += [y0, y1, None]

    edge_trace = go.Scatter(
        x=x_edges,
        y=y_edges,
        mode="lines",
        hoverinfo="none",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # ---- edge labels (optional) ----
    if show_edge_labels and G.number_of_edges() > 0:
        x_lbl, y_lbl, t_lbl = [], [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            x_lbl.append((x0 + x1) / 2)
            y_lbl.append((y0 + y1) / 2)
            t_lbl.append(_get_edge_label(G, u, v))

        lbl_trace = go.Scatter(
            x=x_lbl,
            y=y_lbl,
            mode="text",
            text=t_lbl,
            hoverinfo="none",
        )
        fig.add_trace(lbl_trace)

    return fig

#---------------------WP2 Plots -----------------------
def plot_drf_from_counters_rsmi(
    rsmi: str,
    drf_counter: Counter[str],
    *,
    mode: str = "edge",                   # "edge" | "vertex" | "sp"
    include_edge_labels_in_sp: bool = True,
    hash_labels: bool = True,             # so hast du drf_features_from_rsmi aufgerufen (default True)
    digest_size: int = 16,                # muss mit deinem DRF-Aufruf übereinstimmen
    show_edge_labels: bool = True,
    seed: int = 42,
    title: Optional[str] = None,
) -> go.Figure:

    from synkit.IO import rsmi_to_graph
    ed, pr = rsmi_to_graph(rsmi)

    # Vorkommen pro Seite erzeugen (genau wie im DRF)
    if mode == "edge":
        ed_occ = _build_occurrences_edge(ed, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        pr_occ = _build_occurrences_edge(pr, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        ed_features = [f for _, f in ed_occ]
        pr_features = [f for _, f in pr_occ]
        more_ed, more_pr = _split_direction(ed_features, pr_features)
        # Wir beschränken die Markierung auf die in drf_counter geforderten Features:
        # d.h. nur Features mit diff>0 markieren (Sicherheit: Schnittmenge nehmen)
        more_ed &= drf_counter
        more_pr &= drf_counter
        ed_flags = _mark_edges(ed, ed_occ, more_ed)
        pr_flags = _mark_edges(pr, pr_occ, more_pr)

        pos_ed = nx.spring_layout(ed, seed=seed)
        pos_pr = nx.spring_layout(pr, seed=seed)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Educt", "Product"))
        for tr in _nx_traces_edges(ed, pos_ed, ed_flags, "Educt", show_edge_labels):
            fig.add_trace(tr, row=1, col=1)
        for tr in _nx_traces_edges(pr, pos_pr, pr_flags, "Product", show_edge_labels):
            fig.add_trace(tr, row=1, col=2)

    elif mode == "vertex":
        # Node-Vorkommen
        # Wir hashen nicht Node-für-Node in phi_vertex_list, daher erzeugen wir Feature pro Node direkt
        # mittels der gleichen Hashlogik wie in phi_vertex_list (hier über Subgraph-Trick).
        ed_occ = _build_occurrences_vertex(ed, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        pr_occ = _build_occurrences_vertex(pr, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        ed_features = [f for _, f in ed_occ]
        pr_features = [f for _, f in pr_occ]
        more_ed, more_pr = _split_direction(ed_features, pr_features)
        more_ed &= drf_counter
        more_pr &= drf_counter
        ed_node_flags = _mark_nodes(ed, ed_occ, more_ed)
        pr_node_flags = _mark_nodes(pr, pr_occ, more_pr)

        pos_ed = nx.spring_layout(ed, seed=seed)
        pos_pr = nx.spring_layout(pr, seed=seed)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Educt", "Product"))
        for tr in _nx_to_traces_nodes(ed, pos_ed, ed_node_flags, "Educt"):
            fig.add_trace(tr, row=1, col=1)
        for tr in _nx_to_traces_nodes(pr, pos_pr, pr_node_flags, "Product"):
            fig.add_trace(tr, row=1, col=2)

    elif mode in ("sp", "shortest_path", "shortest-path"):
        ed_occ = _build_occurrences_sp(ed, include_edge_labels=include_edge_labels_in_sp, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        pr_occ = _build_occurrences_sp(pr, include_edge_labels=include_edge_labels_in_sp, hash_labels=hash_labels, digest_size=digest_size, raw_labels=not hash_labels)
        ed_features = [f for _, f in ed_occ]
        pr_features = [f for _, f in pr_occ]
        more_ed, more_pr = _split_direction(ed_features, pr_features)
        more_ed &= drf_counter
        more_pr &= drf_counter
        ed_flags = _mark_edges_by_paths(ed, ed_occ, more_ed)
        pr_flags = _mark_edges_by_paths(pr, pr_occ, more_pr)

        pos_ed = nx.spring_layout(ed, seed=seed)
        pos_pr = nx.spring_layout(pr, seed=seed)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Educt", "Product"))
        for tr in _nx_to_traces_edges(ed, pos_ed, ed_flags, "Educt", show_edge_labels=False):
            fig.add_trace(tr, row=1, col=1)
        for tr in _nx_to_traces_edges(pr, pos_pr, pr_flags, "Product", show_edge_labels=False):
            fig.add_trace(tr, row=1, col=2)
    else:
        raise ValueError("mode must be one of: 'edge', 'vertex', 'sp'")

    fig.update_layout(
        title=title or f"DRF visualization ({mode})",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

def plot_wl_drf_iterations(
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
    width: int = 1200,
    per_row_height: int = 320,
    horizontal_spacing: float = 0.06,
    vertical_spacing: float = 0.06,
) -> go.Figure:
    ed, pr = rsmi_to_educt_product(rsmi)

    pos_ed = nx.spring_layout(ed, seed=seed)
    pos_pr = nx.spring_layout(pr, seed=seed)

    ed_seq = wl_label_sequence(ed, h, hash_node_labels=hash_node_labels, digest_size=digest_size)
    pr_seq = wl_label_sequence(pr, h, hash_node_labels=hash_node_labels, digest_size=digest_size)

    subplot_titles = [t for i in range(h + 1) for t in (f"i={i} Educt", f"i={i} Product")]
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

    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=False, zeroline=False, showticklabels=False)

    return fig


def plot_wl_drf_feature_growth(
    rsmi: str,
    *,
    h: int = 2,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    title: Optional[str] = None,
    show_cumulative: bool = True,
) -> go.Figure:
    per_iter, total = drf_wl(
        rsmi=rsmi,
        h=h,
        mode=mode,
        include_edge_labels_in_sp=include_edge_labels_in_sp,
        hash_node_labels=hash_node_labels,
        hash_features=hash_features,
        digest_size=digest_size,
        return_per_iter=True,
    )

    iters = list(range(h + 1))
    unique_per = [len(c) for c in per_iter]
    mass_per = [sum(c.values()) for c in per_iter]

    cum_unique = []
    cum_mass = []
    if show_cumulative:
        acc = Counter()
        for c in per_iter:
            acc += c
            cum_unique.append(len(acc))
            cum_mass.append(sum(acc.values()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=unique_per, mode="lines+markers", name="unique DRF features (per iter)"))
    fig.add_trace(go.Scatter(x=iters, y=mass_per, mode="lines+markers", name="total DRF count (per iter)"))
    if show_cumulative:
        fig.add_trace(go.Scatter(x=iters, y=cum_unique, mode="lines+markers", name="unique DRF features (cumulative)", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=iters, y=cum_mass, mode="lines+markers", name="total DRF count (cumulative)", line=dict(dash="dash")))
    fig.update_layout(
        title=title or f"WL-DRF feature growth (mode={mode}, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="count",
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
    )
    return fig


def plot_its_wl_feature_growth(
    rsmi: str,
    *,
    h: int = 3,
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    show_cumulative: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    per_iter, total = its_wl_feature_sets(
        rsmi=rsmi,
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
            cum_unique.append(len(acc))
            cum_mass.append(sum(acc.values()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=unique_per, mode="lines+markers", name="unique ITS–WL features (per iter)"))
    fig.add_trace(go.Scatter(x=iters, y=mass_per, mode="lines+markers", name="total ITS–WL count (per iter)"))
    if show_cumulative:
        fig.add_trace(go.Scatter(x=iters, y=cum_unique, mode="lines+markers", name="unique ITS–WL features (cumulative)", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=iters, y=cum_mass, mode="lines+markers", name="total ITS–WL count (cumulative)", line=dict(dash="dash")))
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
    mode: str = "edge",
    include_edge_labels_in_sp: bool = True,
    hash_node_labels: bool = True,
    hash_features: bool = True,
    digest_size: int = 16,
    title: Optional[str] = None,
) -> go.Figure:
    per_iter_counts = [[] for _ in range(h + 1)]

    for rsmi in df[rxn_col]:
        per_iter, _ = its_wl_feature_sets(
            rsmi=rsmi,
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
    its_counts = [[] for _ in range(h + 1)]
    drf_counts = [[] for _ in range(h + 1)]

    for rsmi in df[rxn_col].astype(str):
        its_per_iter, _ = its_wl_feature_sets(
            rsmi=rsmi,
            h=h,
            mode=mode,
            include_edge_labels_in_sp=include_edge_labels_in_sp,
            hash_node_labels=hash_node_labels,
            hash_features=hash_features,
            digest_size=digest_size,
        )
        for i, c in enumerate(its_per_iter):
            its_counts[i].append(len(c))

        drf_per_iter, _ = drf_wl(
            rsmi=rsmi,
            h=h,
            mode=mode,
            include_edge_labels_in_sp=include_edge_labels_in_sp,
            hash_node_labels=hash_node_labels,
            hash_features=hash_features,
            digest_size=digest_size,
            return_per_iter=True,
        )
        for i, c in enumerate(drf_per_iter):
            drf_counts[i].append(len(c))

    iters = list(range(h + 1))
    its_mean = [float(np.mean(v)) for v in its_counts]
    its_std  = [float(np.std(v))  for v in its_counts]
    drf_mean = [float(np.mean(v)) for v in drf_counts]
    drf_std  = [float(np.std(v))  for v in drf_counts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=its_mean, mode="lines+markers", name="ITS–WL (mean unique features / iter)", error_y=dict(type="data", array=its_std, visible=show_errorbars)))
    fig.add_trace(go.Scatter(x=iters, y=drf_mean, mode="lines+markers", name="DRF–WL (mean unique features / iter)", error_y=dict(type="data", array=drf_std, visible=show_errorbars)))
    fig.update_layout(
        title=title or f"ITS–WL vs DRF–WL Feature Growth (subset mean, mode={mode}, h={h})",
        xaxis_title="WL iteration i",
        yaxis_title="mean number of unique features",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
    )
    return fig