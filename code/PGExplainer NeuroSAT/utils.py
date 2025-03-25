from torch_geometric.utils import to_networkx
from torch_geometric.utils import k_hop_subgraph
import torch
import json
from pathlib import Path
import random
from pyvis.network import Network
import networkx as nx
import numpy as np


def get_batch_mask(batch_edges, batch_idx, batch_size=40, n_variables=1440):
    """
    Get a mask for selecting edges belonging to a given batch index.

    Args:
        batch_edges (torch.Tensor): Tensor of shape (num_edges, 2), where first column is literals.
        batch_idx (int): The batch index to filter (0-based).
        batch_size (int): Number of positive and negative literals per batch.
        n_variables (int): Offset where negative literals start.

    Returns:
        torch.Tensor: Boolean mask for filtering edges belonging to the given batch.
    """
    # Compute the range of literals for the given batch
    start_pos = batch_idx * batch_size  # Positive literals start index
    start_neg = n_variables + start_pos  # Negative literals start index

    # Create the set of literals for this batch
    batch_literals = torch.cat([
        torch.arange(start_pos, start_pos + batch_size),       # Positive literals
        torch.arange(start_neg, start_neg + batch_size)        # Negative literals
    ])

    # Create and return the mask
    return torch.isin(batch_edges[:, 0], batch_literals)
    

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


def loadConfig(dataset):
    datasetType = ['BA-2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid', 'NeuroSAT']
    if dataset not in datasetType:
        print("Dataset argument must be one of the following: BA-2Motif, MUTAG, BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid")
        return -1
    
    # Load JSON file "dataset"
    script_dir = Path(__file__).resolve().parent
    config_dir = f"configs/{dataset}.json"

    with open(script_dir.parent.parent / config_dir) as f:
        config = json.load(f)

    return config



def visualize_cnf_interactive(clauses):
    """
    Creates an interactive CNF visualization with PyVis.
    - Allows zooming, dragging, and tooltips.
    - Clause nodes are red, literal nodes are blue.
    """
    net = Network(notebook=True, width="1000px", height="800px", directed=False)

    G = nx.Graph()
    clauses_nodes = []  # Store clause nodes

    # Add nodes and edges
    for i, clause in enumerate(clauses):
        clause_name = f"C{i+1}"  # Clause identifier
        clauses_nodes.append(clause_name)
        net.add_node(clause_name, label=clause_name, color="red", shape="circle")

        for literal in clause:
            if literal not in G.nodes:
                net.add_node(literal, label=str(literal), color="magenta", shape="circle")
            net.add_edge(clause_name, literal, color="gray")

    # Save and display
    net.show("cnf_visualization.html")
    print("Visualization saved as cnf_visualization.html. Open it in your browser.")
    
    
    

def visualize_edge_index_interactive(edge_index, edge_weights=None, name="edge_index_visualization", pos=None, topK=None):
    """
    Creates an interactive CNF visualization with PyVis.
    - edge_index: Tensor of shape [E, 2], where each row represents an edge (literal, clause)
    - edge_weights: Optional tensor of shape [E] containing weights for each edge.
    - Clause nodes are red squares, literal nodes are blue circles.
    """
    net = Network(notebook=True, width="1000px", height="800px", directed=False)
    
    #pos = nx.spring_layout(G ,seed=42) if pos is None else pos
    # Extract unique literals and clauses
    literals = set(edge_index[:, 0].tolist())
    clauses = set(edge_index[:, 1].tolist())
    
    if pos is None:
        G = nx.Graph()
        G.add_nodes_from(literals | {f"C{c}" for c in clauses})
        pos = nx.spring_layout(G, seed=42)  # Generate positions with consistent layout
        
    # Scale positions for PyVis (PyVis uses larger coordinate space)
    scale_factor = 1000  
    scaled_pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
    
    # Define a function to map weight to a color
    def weight_to_color(weight, highlight=False):
        #if highlight:
        #    return "rgb(255, 255, 0)"  # Bright yellow for highlighting top-K edges
        weight = max(0, min(weight, 1))     # Normalize between 0 and 1
        r = int(173  * (1 - weight))         # Reduce red component for darker blue
        g = int(216  * (1 - weight))         # Reduce green component for deeper blue
        b = int(230 - (91 * weight))                        # Keep blue at maximum intensity
        return f"rgb({r},{g},{b})"          # Light blue to dark blue gradient
    
        """r = int(100 * (1 - weight))         # Reduce red component for darker blue
        g = int(150 * (1 - weight))         # Reduce green component for deeper blue
        b = int(255)                        # Keep blue at maximum intensity
        return f"rgb({r},{g},{b})"          # Light blue to dark blue gradient"""
    
    # If numpy array is passed, convert to torch tensor first
    if edge_weights is not None and isinstance(edge_weights, np.ndarray):
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    
    top_k_weights = set()
    if topK is not None and edge_weights is not None:
        sorted_weights, indices = torch.topk(edge_weights, min(topK, len(edge_weights)))
        top_k_weights = set(sorted_weights.tolist())  # Convert to set for easy lookup
        
    
    # Add nodes
    for literal in literals:
        x, y = scaled_pos.get(literal, (0, 0))
        net.add_node(literal, label=str(literal), color="magenta", shape="circle", x=x, y=y)
    
    for clause in clauses:
        clause_name = f"C{clause}"  # Clause identifier
        x, y = scaled_pos.get(clause_name, (0, 0))
        net.add_node(clause_name, label=clause_name, color="red", shape="circle", x=x, y=y)
    
    # Add edges
    for i, (literal, clause) in enumerate(edge_index.tolist()):
        clause_name = f"C{clause}"
        weight = edge_weights[i].item() if edge_weights is not None else 1.0
        
        is_top_k = weight in top_k_weights  # Check if this edge is in the top-K
        #edge_color = weight_to_color(weight, highlight=is_top_k)
        edge_color = weight_to_color(weight)
        if topK is not None:
            edge_width = 2 if is_top_k else 1  # Increase thickness for top-K edges
        else:
            edge_width = weight
        
        # label=str(weight) to display edge weight on edge
        net.add_edge(literal, clause_name, color=edge_color, title=f"Weight: {weight}", value=edge_width)
    
    net.toggle_physics(False)
    
    # Save and display
    net.show(f"{name}.html")
    print(f"Visualization saved as {name}.html. Open it in your browser.")
    
    return pos


"""def showExplanation(mlp, downstreamTask, data, num_explanation_edges, motifNodes, graphTask=True, MUTAG=False):
    mlp.eval()
    downstreamTask.eval()
    
    randomAUCNode = 1

    if graphTask:
        w_ij = mlp.forward(downstreamTask, data.x, data.edge_index)

        edge_ij = mlp.sampleGraph(w_ij)

        _, top_k_indices = torch.topk(edge_ij, k=num_explanation_edges*2, largest=True)

        mask = torch.zeros_like(edge_ij, dtype=torch.bool)
        mask[top_k_indices] = True

        sortedTopK, indices = torch.sort(top_k_indices)

        edge_index_masked = data.edge_index[:,mask]
        weights_masked = edge_ij[sortedTopK]                # This is ordered by size

        # edge_index_masked and weights_masked to display top 5 edges; data1.edge_index and w_ij to display original graph
        G_weights = Data(x=data.x, edge_index=edge_index_masked, edge_attr=weights_masked)
        Gs = Data(x=data.x, edge_index=data.edge_index, edge_attr=edge_ij)
        G_gt = Data(x=data.x, edge_index=data.edge_index[:,data.gt_mask])

        print("-----------------Original Graph-----------------")

        pos = plotGraphAll(data, number_nodes=True, graph_task=True, MUTAG=MUTAG)
        
        print("-----------Original Graph with edge weights-----------")
        
        pos1 = plotGraphAll(G_weights, pos=pos, number_nodes=True, graph_task=True, edge_weights=True, MUTAG=MUTAG)

        print("-----------------Explanation Graph-----------------")

        pos1 = plotGraphAll(Gs, pos=pos, number_nodes=True, graph_task=True, edge_weights=True, MUTAG=MUTAG)
        
        print("-----------------Ground truth Graph-----------------")
    
        pos2 = plotGraphAll(G_gt, pos=pos, number_nodes=True, graph_task=True, MUTAG=MUTAG)
    
    else:
        randomAUCNode = random.choice(motifNodes)
        
        subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=randomAUCNode, num_hops=3, edge_index=data.edge_index, relabel_nodes=True)

        indexNodeToPred = (subset == randomAUCNode).nonzero().item()

        G_hop = Data(x=data.x[subset], edge_index=edge_index_hop, y=data.y[subset])

        print("-----------------Original Computational Graph-----------------")
        pos = plotGraphAll(G_hop)

        w_ij = mlp.forward(downstreamTask, data.x[subset], edge_index_hop, indexNodeToPred)

        # Min-Max Normalization. This works pretty well
        weights_min = w_ij.min()
        weights_max = w_ij.max()
        weights_norm = (w_ij - weights_min) / (weights_max - weights_min)

        edge_ij = mlp.sampleGraph(w_ij)

        ## REMOVE IF SIGMOID WANTED
        edge_ij = weights_norm


        GraphSampled = Data(x=G_hop.x, edge_index=G_hop.edge_index, y=G_hop.y, edge_attr=edge_ij.detach())

        print("-----------Original Computational Graph with calculated Weights -----------")
        pos = plotGraphAll(GraphSampled, pos, edge_weights=True)


        # Print topK edges
        k = num_explanation_edges * 2 if len(w_ij) >= num_explanation_edges*2 else len(w_ij)
        _, top_k_indices = torch.topk(edge_ij, k=k, largest=True)

        mask = torch.zeros_like(edge_ij, dtype=torch.bool)
        mask[top_k_indices] = True

        sortedTopK, indices = torch.sort(top_k_indices)

        edge_index_masked = G_hop.edge_index[:,mask]
        weights_masked = edge_ij[sortedTopK]                # This is ordered by size

        GtopK = Data(x=G_hop.x, edge_index=edge_index_masked, y=G_hop.y, edge_attr=weights_masked)

        print("-----------------Top K Motif Graph-----------------")
        pos1 = plotGraphAll(GtopK, pos=pos, color_map=None, edge_weights=True)
        
        G_gt = Data(x=G_hop.x, edge_index=G_hop.edge_index[:,data.gt_mask[edge_mask]], y=G_hop.y)
        
        print("-----------------Ground Truth-----------------")
        pos1 = plotGraphAll(G_gt, pos=pos, color_map=None, edge_weights=False)
    
    return randomAUCNode"""