from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch

# TODO: plotGraph
def plotGraph (graph, pos=None, color_map=None, edge_weights=False, MUTAG=False):
    if edge_weights: nxGraph = to_networkx(graph, edge_attrs=["edge_attr"], to_undirected="upper")
    else: nxGraph = to_networkx(graph, to_undirected=True)
    
    if pos is None: pos = nx.spring_layout(nxGraph)
    if color_map is None and MUTAG is False:
        color_map = [
            '#c41f1c' if node in [20,21] else 
            '#d3eb0d' if node in [22,23] else 
            '#28a8ec' if node == 24 else
            '#eb790d' 
            for node in nxGraph.nodes()
        ]
        
    if color_map is None and MUTAG:
        indices = torch.argmax(graph.x, axis=1)
        colors = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        color_map = [
            colors[idx]
            for idx in indices
        ]
        
    plt.figure(1)
    nx.draw(nxGraph, pos=pos, node_color=color_map)

    nx.draw_networkx_labels(nxGraph, pos, labels={i: f"{i}" for i in range(graph.x.shape[0])})

    if edge_weights:
        labels = nx.get_edge_attributes(nxGraph,'edge_attr')
        labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(nxGraph, 'edge_attr').items()}
        nx.draw_networkx_edge_labels(nxGraph, pos, edge_labels=labels, font_size=6)
 
    plt.show()
    return pos

# Edge weights seem to work, at least on graphs. graph needs attribute edge_attr containing weights. TODO: Test on NodeClass
def plotGraphAll (graph, pos=None, color_map=None, edge_weights=False, graph_task=False, MUTAG=False, number_nodes=False):
    if edge_weights: nxGraph = to_networkx(graph, edge_attrs=["edge_attr"], to_undirected="upper")
    else: nxGraph = to_networkx(graph, to_undirected=True)
    
    if pos is None: pos = nx.spring_layout(nxGraph, seed=42)
        
    if color_map is None:
        if graph_task:
            # TODO: Find a clean way to differentiate between MUTAG and BA2Motif to color motif nodes in latter
            if MUTAG:
                indices = torch.argmax(graph.x, axis=1)
                colors = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
                color_map = [
                colors[idx]
                for idx in indices
                ]
            else:
               color_map = [
                '#c41f1c' if node in [20,21] else 
                '#d3eb0d' if node in [22,23] else 
                '#28a8ec' if node == 24 else
                '#eb790d' 
                for node in nxGraph.nodes()
        ] 
        else: 
            color_map = []
            for i, j in enumerate(graph.y):
                color_map.append([j.item()])
        
    
    node_size = 300 if graph_task else 80
    plt.figure(1, figsize=(10, 10))
    nx.draw(nxGraph, pos=pos, node_size=node_size, node_color=color_map, font_size=8)

    if number_nodes: nx.draw_networkx_labels(nxGraph, pos, labels={i: f"{i}" for i in range(graph.x.shape[0])})

    if edge_weights:
        labels = nx.get_edge_attributes(nxGraph,'edge_attr')
        labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(nxGraph, 'edge_attr').items()}
        nx.draw_networkx_edge_labels(nxGraph, pos, edge_labels=labels, font_size=6)
 
    plt.show()
    return pos


# TODO: Not quite clean, maybe do k_hop_subgraph outside
def plotKhopGraph (startNode, x, edge_index, num_hops=3, pos=None, color_map=None):
    subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(startNode, num_hops, edge_index, relabel_nodes=False)

    G_hop = Data(x=x, edge_index=edge_index_hop)
    
    nxGraph = to_networkx(G_hop, to_undirected=True)
    
    if pos is None: pos = nx.spring_layout(nxGraph)
    
    if color_map is None: color_map = [
        'red' if node == startNode else 
        'green' if node in subset else 
        'yellow' 
        for node in nxGraph.nodes()
    ]
    
    plt.figure(1)
    nx.draw(nxGraph, pos=pos, node_color=color_map)

    nx.draw_networkx_labels(nxGraph, pos, labels={i: f"{i}" for i in range(G_hop.x.shape[0])})
    plt.show()
    
    return pos, G_hop


def plotMotif ():
    return


def plotTreeCycles (graph, pos=None, color_map=None, edge_weights=None):
    if color_map is None:
        color_map = []
        for i, j in enumerate(graph.y):
            color_map.append([j.item()])
            
    plt.figure(figsize=(10, 10))  # You can adjust the size as needed
    
    if edge_weights is not None: 
        graphSampled = Data(x=graph.x,edge_index=graph.edge_index,edge_attr=edge_weights)
        g = to_networkx(graphSampled, edge_attrs=["edge_attr"], to_undirected="lower")
        labels = nx.get_edge_attributes(g,'edge_attr')
        labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(g, 'edge_attr').items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_size=8)
    else:
        g = to_networkx(graph, to_undirected=True)

    if pos is None: pos = nx.spring_layout(g, seed=42)          # try kamada_kawai_layout
    
    nx.draw(g, pos, node_size=40,font_size=8, node_color = color_map)    
    
    plt.show()
    
    return pos
    
    
# TODO: This shit does not work
def combineEdgeWeights (edge_index, edge_weights):
    # Example: Edge index (2, num_edges) and edge weights
    edge_index = edge_index
    w_ij = edge_weights  # Assume this is a 1D tensor of shape (num_edges,)

    # Create a dictionary to store reversed edge indices for quick lookup
    edge_dict = {}
    processed = set()  # To avoid processing the same pair twice
    
    for i in range(edge_index.shape[1]):
        edge_dict[(edge_index[0, i].item(), edge_index[1, i].item())] = i  # Store index position

    # Create a new tensor to store combined weights
    combined_w_ij = edge_weights.clone()

    # Iterate and combine opposite edge weights
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Check if we already processed this edge (or its reverse)
        if (v, u) in edge_dict and (u, v) in processed:
            continue  # Skip duplicate processing
        
        if (v, u) in edge_dict:  # Check if the reverse edge exists
            j = edge_dict[(v, u)]  # Get the index of the reverse edge
            combined_weight = w_ij[i] + w_ij[j]
            
            # Assign the same combined weight to both edges
            combined_w_ij[i] = combined_weight / 2
            combined_w_ij[j] = combined_weight / 2

            # Mark as processed
            processed.add((u, v))
            processed.add((v, u))
            
    # Update w_ij with the new combined weights
    return combined_w_ij