from synkit.IO import rsmi_to_its, rsmi_to_graph
from synkit.Vis import GraphVisualizer
import matplotlib.pyplot as plt
import networkx as nx

#rsmi = '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'


# Parse Individual Chemical Graphs as NetworkX Objects
def visualize_graph(rsmi:str):
    educt_graph, product_graph = rsmi_to_graph(rsmi, drop_non_aam=False, use_index_as_atom_map=False)

    print("Eductgraph")
    for n, d in educt_graph.nodes(data=True):
        print(n, d)
    for u,v, d in educt_graph.edges(data=True):
        print(u,v, d)

    print("Productgraph")
    for n, d in product_graph.nodes(data=True):
        print(n, d)
    for u,v, d in product_graph.edges(data=True):
        print(u,v, d)

    # Full ITS graph as NetworkX Object
    full_graph = rsmi_to_its(rsmi, core=False,  drop_non_aam=False, use_index_as_atom_map=False)

    print("ITS graph")
    for n, d in full_graph.nodes(data=True):
        print(n, d)
    for u,v, d in full_graph.edges(data=True):
        print(u,v, d)

    # Visualize ITS
    viz = GraphVisualizer()
    fig = viz.visualize_its(full_graph, use_edge_color=True)
    plt.show()

