from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch
import json
from pathlib import Path
import random

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
        print("Dataset argument must be one of the following: BA-2Motif, MUTAG, BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid, NeuroSAT")
        return -1
    
    # Load JSON file "dataset"
    script_dir = Path(__file__).resolve().parent
    config_dir = f"configs/{dataset}.json"

    with open(script_dir.parent.parent / config_dir) as f:
        config = json.load(f)

    return config


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