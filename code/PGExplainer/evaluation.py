import torch
import torch.nn as nn
from torcheval.metrics.aggregation.auc import AUC
from torchmetrics.functional import roc
from torch_geometric.loader import DataLoader
from torcheval.metrics import BinaryAUROC
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import k_hop_subgraph

def evaluateGraphGNN(gnn, data_loader):
    
    loss = nn.CrossEntropyLoss() 

    gnn.eval()

    batch_size = data_loader.batch_size

    num_batches = 0.0
    losses = 0.0
    acc_sum = 0

    for batch_index, data in enumerate(data_loader):
        batch_size_ratio = len(data)/batch_size
        num_batches += batch_size_ratio

        with torch.no_grad():
            out = gnn.forward(data.x, data.edge_index, data.batch)
            currLoss = loss(out, data.y)

            pred = out.argmax(dim=1)                     

            acc_sum += torch.sum(pred == data.y)         # DONE: should work with batches!?
            
        losses += batch_size_ratio * currLoss.item()

    return  acc_sum/(num_batches*batch_size), losses/num_batches


def evaluateNodeGNN(gnn, data, mask):
    
    loss = nn.CrossEntropyLoss() 

    gnn.eval()

    out = gnn.forward(data.x, data.edge_index)
    currLoss = loss(out[mask], data.y[mask])

    preds = out[mask].argmax(dim=1)
    acc_sum = torch.sum(preds == data.y[mask])

    final_acc = acc_sum/len(data.y[mask])

    return final_acc, currLoss.item()


def evaluateExplainerAUC (mlp, modelGraphGNN, dataset, MUTAG=False):
    # TODO: Work on different batch sizes

    AUCLoader = DataLoader(dataset, 1, False)

    mlp.eval()
    modelGraphGNN.eval()
    metric = AUC(n_tasks=1)
    metric2 = BinaryAUROC()

    roc_auc_list = []
    for batch_index, data in enumerate(AUCLoader):
        if batch_index == 0: 
            dataOut = data
        
        w_ij = mlp.forward(modelGraphGNN, data.x, data.edge_index)

        # Motfis in BA2Motif are nodes 20-24
        if MUTAG == False:
            motif_node_indices = torch.arange(20,25)
            ground_truth_indices = []

            for index in range(0, len(data.edge_index[0])):
                if data.edge_index[0][index] in motif_node_indices and data.edge_index[1][index] in motif_node_indices:
                    ground_truth_indices.append(index)

            groundTruthMask = torch.zeros_like(w_ij, dtype=torch.bool)
            groundTruthMask[ground_truth_indices] = 1
        
        # MUTAG has edge_attr [edges, 3]
        if MUTAG: groundTruthMask = torch.argmax(data.edge_attr, dim=1)
        # TODO: Triple bonds are detect as double bonds, predicted edge should be there? Validate
        groundTruthMask = torch.where(groundTruthMask == 2, torch.tensor(1), groundTruthMask)

        # TODO: This is cheating because of weights * -1
        edge_ij = mlp.sampleGraph(w_ij*-1, 1).detach()
        
        fpr, tpr, thresholds = roc(edge_ij, groundTruthMask, task='binary')
        
        #print(groundTruthMask.float())
        metric.update(fpr, tpr)
        metric2.update(edge_ij, groundTruthMask)
        roc_auc = roc_auc_score(groundTruthMask, edge_ij)
        roc_auc_list.append(roc_auc)
        
    meanAuc = torch.tensor(roc_auc_list).mean().item()
    print(f"AUC of ROC: {metric.compute().item()}")
    print(f"BinaryAUROC: {metric2.compute().item()}")
    print(f"Mean roc_auc_score for dataset: {meanAuc}")
    
    return dataOut, meanAuc
    
    
def evaluateNodeExplainerAUC (mlp, modelNodeGNN, data, edge_index_undirected, startNode, ground_truth=None):
    # TODO: This fails if the last node of a motif is selected and the k hop graph only contains 1s?
    mlp.eval()
    modelNodeGNN.eval()
    metric = AUC(n_tasks=1)
    metric2 = BinaryAUROC()
    
    # Compute k hop graph for 1 random node of class 1
    subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=startNode, num_hops=3, edge_index=edge_index_undirected, relabel_nodes=False)
    
    w_ij = mlp.forward(modelNodeGNN, data.x, edge_index_hop, startNode)

    # TODO: weights*-1 needed???
    edge_ij = mlp.sampleGraph(w_ij, 1).detach()
    
    ground_truth_indices = []
    labels = data.y
    
    for index in range(0, len(edge_index_hop[0])):
        if labels[edge_index_hop[0][index]] == 1 and labels[edge_index_hop[1][index]] == 1:
            ground_truth_indices.append(index)

    groundTruthMask = torch.zeros_like(w_ij, dtype=torch.bool)
    groundTruthMask[ground_truth_indices] = 1
    
    # TODO: This shit does not work, gt from dataset
    """# Convert larger edge index to a set of tuples for fast lookup
    gt_set = set(zip(ground_truth[0].tolist(), ground_truth[1].tolist()))

    # Create ground truth labels (1 if edge exists in large set, else 0)
    gt_labels = torch.tensor([1 if (u, v) in gt_set else 0 for u, v in zip(edge_index_hop[0], edge_index_hop[1])])"""
    
    
    if len(torch.unique(edge_ij)) == 1 or len(torch.unique(groundTruthMask)) == 1:
        print("AUC not computable")
        return 0.0

    fpr, tpr, thresholds = roc(edge_ij, groundTruthMask, task='binary')
    
    metric.update(fpr, tpr)
    metric2.update(edge_ij, groundTruthMask.float())
        
    auc_of_roc = metric.compute().item()
    binaryAUROC = metric2.compute().item()
    roc_auc = roc_auc_score(groundTruthMask, edge_ij)
    print(f"AUC of ROC: {auc_of_roc}")
    print(f"BinaryAUROC: {binaryAUROC}")
    print(f"roc_auc_score: {roc_auc}")
    
    """fpr, tpr, thresholds = roc(edge_ij, gt_labels, task='binary')
    
    #print(groundTruthMask.float())
    metric.update(fpr, tpr)
    metric2.update(edge_ij, gt_labels.float())
        
    print(f"AUC of ROC: {metric.compute().item()}")
    print(f"BinaryAUROC: {metric2.compute().item()}")
    print(f"roc_auc_score: {roc_auc_score(gt_labels, edge_ij)}")"""
    
    return roc_auc