# src/vis_utils.py
# ============================================================
# This file provides small helper functions to:
# 1) build graphs from reaction SMILES (rsmi) using synkit
# 2) print debug info about graphs (nodes/edges + attributes)
# 3) plot graphs interactively with Plotly for manual checks
# 4) build and inspect WP1 subsets (3–5 classes, 20–200 samples)
#
# Typical usage in your notebook:
#   from src.vis_utils import visualize_graph, make_random_subset
#   subset = make_random_subset(data, class_col="rxn_class", rxn_col="clean_rxn",
#                               n_classes=3, n_per_class=20, seed=42)
#   rsmi = subset["clean_rxn"].iloc[9]
#   visualize_graph(rsmi)              # shows ITS
#   visualize_graph(rsmi, which="educt")
#   visualize_graph(rsmi, which="product")
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from chem_graph_handling import visualize_graph

# synkit turns a reaction SMILES string into NetworkX graphs
from synkit.IO import rsmi_to_its, rsmi_to_graph




# ============================================================
# 1) Robust label getters (because attribute names can differ)
# ============================================================

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


# ============================================================
# 2) Build graphs from reaction SMILES
# ============================================================

@dataclass
class ReactionGraphs:
    """
    Container holding up to three graphs from the same reaction SMILES:

    - its:        one ITS graph representing the reaction as a single graph
    - educt_graph: reactant-side graph
    - product_graph: product-side graph

    NOTE:
    synkit.IO.rsmi_to_graph(rsmi) returns TWO graphs (educt, product).
    """
    rsmi: str
    its: Optional[nx.Graph] = None
    educt_graph: Optional[nx.Graph] = None
    product_graph: Optional[nx.Graph] = None

def build_graphs(
    rsmi: str,
    build_its: bool = True,
    build_educt_product: bool = True,
) -> ReactionGraphs:
    """
    Build NetworkX graphs from a reaction SMILES string.

    - rsmi_to_its(rsmi) returns a single ITS graph
    - rsmi_to_graph(rsmi) returns (educt_graph, product_graph)
    """
    out = ReactionGraphs(rsmi=rsmi)

    if build_its:
        out.its = rsmi_to_its(rsmi)

    if build_educt_product:
        ed, pr = rsmi_to_graph(rsmi)
        out.educt_graph = ed
        out.product_graph = pr

    return out

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

# ============================================================
# 3) Printing / debug helpers
# ============================================================

def print_reaction(rsmi: str) -> None:
    """Print the raw reaction SMILES string."""
    print("Reaction SMILES:")
    print(rsmi)


def print_graph_info(G: nx.Graph, title: str = "Graph") -> None:
    """
    Print basic graph stats and show a few node/edge attributes.
    This is very useful before implementing WL features, because you need
    to know which attributes exist (node labels, edge labels, etc.).
    """
    print(f"== {title} ==")
    print(f"nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

    # show only a small sample (otherwise too much output)
    nodes = list(G.nodes())[:8]
    edges = list(G.edges())[:8]

    print("sample nodes:")
    for n in nodes:
        print(f"  {n}: {_get_node_label(G, n)} | attrs={dict(G.nodes[n])}")

    print("sample edges:")
    for u, v in edges:
        lbl = _get_edge_label(G, u, v)
        attrs = dict(G.edges[u, v])
        print(f"  ({u}, {v}): {lbl} | attrs={attrs}")


def quick_sanity(rsmi: str) -> ReactionGraphs:
    """
    Quick manual test:
    - print rsmi
    - build ITS + (educt, product) graphs
    - print small summaries for each
    """
    print_reaction(rsmi)
    gs = build_graphs(rsmi, build_its=True, build_educt_product=True)

    if gs.its is not None:
        print_graph_info(gs.its, title="ITS graph")
    if gs.educt_graph is not None:
        print_graph_info(gs.educt_graph, title="Educt graph")
    if gs.product_graph is not None:
        print_graph_info(gs.product_graph, title="Product graph")

    return gs


# ============================================================
# 4) Plotting: NetworkX -> Plotly
# ============================================================

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



# ============================================================
# 5) WP1 subset builders
# ============================================================

def make_random_subset(
    df: pd.DataFrame,
    class_col: str = "rxn_class",
    rxn_col: str = "clean_rxn",
    n_classes: int = 3,
    n_per_class: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a WP1 subset:
    - choose n_classes classes that have at least n_per_class samples
    - sample exactly n_per_class rows from each chosen class

    Returns a tidy dataframe with only two columns:
        [rxn_col, class_col]

    This matches the WP1 requirement: choose 3–5 classes and 20–200 samples each.

    Example:
        subset = make_random_subset(data, class_col="rxn_class", rxn_col="clean_rxn",
                                    n_classes=3, n_per_class=20, seed=42)
    """
    if class_col not in df.columns:
        raise KeyError(f"class_col '{class_col}' not in df.columns: {list(df.columns)}")
    if rxn_col not in df.columns:
        raise KeyError(f"rxn_col '{rxn_col}' not in df.columns: {list(df.columns)}")

    counts = df[class_col].value_counts()
    eligible = counts[counts >= n_per_class].index.tolist()

    if len(eligible) < n_classes:
        raise ValueError(
            f"Only {len(eligible)} classes have >= {n_per_class} samples; need {n_classes}."
        )

    chosen_classes = pd.Series(eligible).sample(n=n_classes, random_state=seed).tolist()

    parts = []
    for c in chosen_classes:
        sub = df[df[class_col] == c].sample(n=n_per_class, random_state=seed)
        parts.append(sub)

    subset = pd.concat(parts, axis=0).reset_index(drop=True)
    return subset[[rxn_col, class_col]]


def make_subset_by_classes(
    df: pd.DataFrame,
    classes: List[Any],
    class_col: str = "rxn_class",
    rxn_col: str = "clean_rxn",
    n_per_class: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Same as make_random_subset, but you specify the classes explicitly.

    Example:
        subset = make_subset_by_classes(data, classes=[1, 5, 9],
                                        class_col="rxn_class", rxn_col="clean_rxn",
                                        n_per_class=20, seed=42)
    """
    if class_col not in df.columns:
        raise KeyError(f"class_col '{class_col}' not in df.columns: {list(df.columns)}")
    if rxn_col not in df.columns:
        raise KeyError(f"rxn_col '{rxn_col}' not in df.columns: {list(df.columns)}")

    parts = []
    for c in classes:
        sub = df[df[class_col] == c]
        if len(sub) < n_per_class:
            raise ValueError(f"Class {c} has only {len(sub)} rows; need {n_per_class}.")
        parts.append(sub.sample(n=n_per_class, random_state=seed))

    subset = pd.concat(parts, axis=0).reset_index(drop=True)
    return subset[[rxn_col, class_col]]


# ============================================================
# 6) Convenience: inspect rows (works on full df or subset df)
# ============================================================

def inspect_by_index(
    df: pd.DataFrame,
    idx: int,
    rxn_col: str = "clean_rxn",
    which: str = "its",
    show_edge_labels: bool = True,
) -> None:
    """
    Inspect one row by integer position:
    - print small summaries (quick_sanity)
    - then visualize
    """
    rsmi = df[rxn_col].iloc[idx]
    quick_sanity(rsmi)
    visualize_graph(rsmi, which=which, show_edge_labels=show_edge_labels)


def inspect_random(
    df: pd.DataFrame,
    n: int = 5,
    rxn_col: str = "clean_rxn",
    which: str = "its",
    seed: int = 42,
) -> None:
    """
    Inspect n random reactions from the given dataframe.
    Works on the full dataset OR on your WP1 subset.
    """
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    for i in range(len(sample)):
        print("\n" + "=" * 80)
        print(f"Random example #{i}")
        rsmi = sample[rxn_col].iloc[i]
        quick_sanity(rsmi)
        visualize_graph(rsmi, which=which)