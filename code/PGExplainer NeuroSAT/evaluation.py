import torch
import torch.nn as nn
import utils
import random
import datasetLoader
from torcheval.metrics.aggregation.auc import AUC
from torchmetrics.functional import roc
from torch_geometric.loader import DataLoader
from torcheval.metrics import BinaryAUROC
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import k_hop_subgraph
import numpy as np

def evaluateExplainerAUC (mlp, modelGraphGNN, dataset, num_explanation_edges=5):
    AUCLoader = DataLoader(dataset, 1, False)

    mlp.eval()
    modelGraphGNN.eval()
    metric2 = BinaryAUROC()
    
    reals = []
    preds = []

    for batch_index, data in enumerate(AUCLoader):
        w_ij = mlp.forward(modelGraphGNN, data.x, data.edge_index)

        edge_ij = mlp.sampleGraph(w_ij, 1).detach()
        
        # Instead of weights, take top k edges according to motif?
        """k = num_explanation_edges * 2 if len(w_ij) >= num_explanation_edges*2 else len(w_ij)
        _, top_k_indices = torch.topk(w_ij, k=k, largest=True)
        topEdgesMask = torch.zeros_like(w_ij, dtype=torch.float32)
        topEdgesMask[top_k_indices] = 1"""
        
        groundTruthMask = data.gt_mask
        
        # This prevents AUC being calculated if all nodes are from the same class, since only either positives or negatives
        """if len(torch.unique(topEdgesMask)) == 1 or len(torch.unique(groundTruthMask)) == 1:
            print("AUC not computable")
        else:"""
        metric2.update(edge_ij, groundTruthMask.float())
        
        reals.append(groundTruthMask.flatten().numpy())
        preds.append(edge_ij.cpu().flatten().numpy())
            
    # Convert the lists to numpy arrays and calculate the roc_auc_score
    reals = np.concatenate(reals)  # Flatten the list of arrays
    preds = np.concatenate(preds)  # Flatten the list of arrays
    roc_auc = roc_auc_score(reals, preds)
    
    print(f"BinaryAUROC: {metric2.compute().item()}")
    print(f"roc_auc_score: {roc_auc}")
    
    return roc_auc
    
    
def evaluateNodeExplainerAUC (mlp, modelNodeGNN, data, evalNodes, num_explanation_edges=6):
    mlp.eval()
    modelNodeGNN.eval()
    metric2 = BinaryAUROC()
    
    reals = []
    preds = []
    
    for currentNode in evalNodes:
        # Compute k hop graph for 1 node
        subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=currentNode, num_hops=3, edge_index=data.edge_index, relabel_nodes=False)
    
        w_ij = mlp.forward(modelNodeGNN, data.x, edge_index_hop, currentNode)
        edge_ij = mlp.sampleGraph(w_ij, 1).detach()
        
        #TODO: Sum up predictions of both directions

        # Instead of weights, take top k edges according to motif?
        """k = num_explanation_edges * 2 if len(w_ij) >= num_explanation_edges*2 else len(w_ij)
        _, top_k_indices = torch.topk(w_ij, k=k, largest=True)
        topEdgesMask = torch.zeros_like(w_ij, dtype=torch.float32)
        topEdgesMask[top_k_indices] = 1"""
    
        # TODO: Use edge_mask from k_hop_subgraph on gt of same length. This should work but gt (edge_index_mask) MUST match data.edge_index!
        subgraph_ground_truth = data.gt_mask[edge_mask]
            
        # This prevents AUC being calculated if all nodes are from the same class, since only either positives or negatives
        """if len(torch.unique(edge_ij)) == 1 or len(torch.unique(subgraph_ground_truth)) == 1:
            print("AUC not computable")
            return -1"""
    
        metric2.update(edge_ij, subgraph_ground_truth.float())
        
        reals.append(subgraph_ground_truth.flatten().numpy())
        preds.append(edge_ij.cpu().flatten().numpy())
        
    binaryAUROC = metric2.compute().item()
    
    # Convert the lists to numpy arrays and calculate the roc_auc_score
    reals = np.concatenate(reals)  # Flatten the list of arrays
    preds = np.concatenate(preds)  # Flatten the list of arrays
    roc_auc = roc_auc_score(reals, preds)
    
    print(f"BinaryAUROC: {binaryAUROC}")
    print(f"roc_auc_score: {roc_auc}")
    
    return roc_auc




def evaluate (datasetName, mlp, downstreamTask):
    config = utils.loadConfig(datasetName)
    if config == -1:
        return
    
    params = config['params']
    graph_task = params['graph_task']
    num_explanation_edges = params['num_explanation_edges']
    
    MUTAG = True if datasetName == "MUTAG" else False

    
    data, labels = datasetLoader.loadGraphDataset(datasetName) if graph_task else datasetLoader.loadOriginalNodeDataset(datasetName)
    
    if graph_task:
        # TODO: This loading of datasets needs to be generalized, maybe move back to data laoder
        if MUTAG:
            selected_data = []
            selectedIndices = []
            for i in range(0, len(data)):
                if data[i].y == 0 and torch.sum(data[i].gt_mask) > 0:
                    selectedIndices.append(i)
                    selected_data.append(data[i])
        
            data = selected_data
        
        meanAuc = evaluateExplainerAUC(mlp, downstreamTask, data, num_explanation_edges=num_explanation_edges)
        
        # For showExplanation, pass random graph from data
        data = data[random.choice(range(0,len(data)))]
        
        _ = utils.showExplanation(mlp, downstreamTask, data, num_explanation_edges, motifNodes=None, graphTask=graph_task, MUTAG=MUTAG)
    else:
        # TODO: move to config
        if datasetName == "BA-Community":
            single_label = data.y
            motifNodes = [i for i in range(single_label.shape[0]) if single_label[i] != 0 and single_label[i] != 4]
            
        else:
            motif_node_indices = params['motif_node_indices']
            motifNodes = [i for i in range(motif_node_indices[0], motif_node_indices[1], motif_node_indices[2])]
        
        meanAuc = evaluateNodeExplainerAUC(mlp, downstreamTask, data, motifNodes, num_explanation_edges=num_explanation_edges)
        
        randomAucNode = utils.showExplanation(mlp, downstreamTask, data, num_explanation_edges, motifNodes, graph_task)
        auc = evaluateNodeExplainerAUC(mlp, downstreamTask, data, [randomAucNode], num_explanation_edges=num_explanation_edges)
        print(f"AUC for random Node: {auc}")
        
    return meanAuc





def evaluateNeuroSATAUC (mlp, downstreamTask, dataset, num_explanation_edges=5):
    AUCLoader = DataLoader(dataset, 1, False)

    mlp.eval()
    downstreamTask.eval()
    metric2 = BinaryAUROC()
    
    reals = []
    preds = []

    for batch_index, data in enumerate(AUCLoader):
        w_ij = mlp.forward(downstreamTask, data)

        edge_ij = mlp.sampleGraph(w_ij, 1).detach()
        
        # TODO: GT mask for SAT problems!
        groundTruthMask = data.gt_mask
        
        # This prevents AUC being calculated if all nodes are from the same class, since only either positives or negatives
        """if len(torch.unique(topEdgesMask)) == 1 or len(torch.unique(groundTruthMask)) == 1:
            print("AUC not computable")
        else:"""
        metric2.update(edge_ij, groundTruthMask.float())
        
        reals.append(groundTruthMask.flatten().numpy())
        preds.append(edge_ij.cpu().flatten().numpy())
            
    # Convert the lists to numpy arrays and calculate the roc_auc_score
    reals = np.concatenate(reals)  # Flatten the list of arrays
    preds = np.concatenate(preds)  # Flatten the list of arrays
    roc_auc = roc_auc_score(reals, preds)
    
    print(f"BinaryAUROC: {metric2.compute().item()}")
    print(f"roc_auc_score: {roc_auc}")
    
    return roc_auc