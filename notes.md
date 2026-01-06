# Reaction Kernels Lab (Graph Theory) Notes 
generelle grobe tendenzen -- offentsichtliche UNterschiede, Classifier, IT /Symmetrc difference

def visualize_graph(
    rsmi: str,
    which: str = "its",
    show_edge_labels: bool = True,
    node_size: int = 16,
) -> None:
    """
    One-liner visualization, like you requested:

        rsmi = subset["clean_rxn"].iloc[9]
        visualize_graph(rsmi)               # default: ITS

    Parameters:
      which:
        - "its"     : ITS graph (reaction as one graph)
        - "educt"   : reactants-only graph
        - "product" : products-only graph
    """
    if which == "its":
        G = rsmi_to_its(rsmi)
        fig = plot_nx_graph(
            G,
            title="ITS (reaction as one graph)",
            show_edge_labels=show_edge_labels,
            node_size=node_size,
        )
        fig.show()
        return

    # rsmi_to_graph returns two graphs
    ed, pr = rsmi_to_graph(rsmi)

    if which == "educt":
        fig = plot_nx_graph(
            ed,
            title="Educt graph",
            show_edge_labels=show_edge_labels,
            node_size=node_size,
        )
        fig.show()
        return

    if which == "product":
        fig = plot_nx_graph(
            pr,
            title="Product graph",
            show_edge_labels=show_edge_labels,
            node_size=node_size,
        )
        fig.show()
        return

    raise ValueError("which must be 'its', 'educt', or 'product'")