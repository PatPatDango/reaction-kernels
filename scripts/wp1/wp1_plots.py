from synkit.IO import rsmi_to_graph
import networkx as nx
import plotly.graph_objects as go
from typing import Any, Iterable, List, Tuple, Optional, Dict

#local imports
from scripts.wp0.chem_graph_handling import visualize_graph


#-- helper functions ----
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