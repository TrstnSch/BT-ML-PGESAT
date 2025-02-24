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


def evaluateExplainerAUC (mlp, modelGraphGNN, dataset, MUTAG=False, num_explanation_edges=5):
    # TODO: Work on different batch sizes

    AUCLoader = DataLoader(dataset, 1, False)

    mlp.eval()
    modelGraphGNN.eval()
    metric = AUC(n_tasks=1)
    metric2 = BinaryAUROC()

    roc_auc_list = []
    for batch_index, data in enumerate(AUCLoader):
        w_ij = mlp.forward(modelGraphGNN, data.x, data.edge_index)

        """# Motfis in BA2Motif are nodes 20-24
        if MUTAG == False:
            k = 6 if data.y.item() == 1 else 5
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
        groundTruthMask = torch.where(groundTruthMask == 2, torch.tensor(1), groundTruthMask)"""

        # TODO: This is cheating because of weights * -1
        edge_ij = mlp.sampleGraph(w_ij, 1).detach()
        
        # Instead of weights, take top k edges according to motif?
        
        k = num_explanation_edges * 2 if len(w_ij) >= num_explanation_edges*2 else len(w_ij)
        _, top_k_indices = torch.topk(w_ij, k=k, largest=True)
        topEdgesMask = torch.zeros_like(w_ij, dtype=torch.float32)
        topEdgesMask[top_k_indices] = 1
        
        groundTruthMask = data.gt_mask
        
        topEdgesMask = edge_ij
        
        # This prevents AUC being calculated if all nodes are from the same class, since only either positives or negatives
        if len(torch.unique(topEdgesMask)) == 1 or len(torch.unique(groundTruthMask)) == 1:
            print("AUC not computable")
        else:
            fpr, tpr, thresholds = roc(topEdgesMask, groundTruthMask, task='binary')
            
            #print(groundTruthMask.float())
            metric.update(fpr, tpr)
            metric2.update(topEdgesMask, groundTruthMask.float())
            roc_auc = roc_auc_score(groundTruthMask, topEdgesMask)
            roc_auc_list.append(roc_auc)
        
    meanAuc = torch.tensor(roc_auc_list).mean().item()
    print(f"AUC of ROC: {metric.compute().item()}")
    print(f"BinaryAUROC: {metric2.compute().item()}")
    print(f"Mean roc_auc_score for dataset: {meanAuc}")
    
    return meanAuc
    
    
# TODO: Not necessary to pass edge_index_undirected and gt, just take from data!!!
def evaluateNodeExplainerAUC (mlp, modelNodeGNN, data, startNode, num_explanation_edges=6):
    mlp.eval()
    modelNodeGNN.eval()
    metric = AUC(n_tasks=1)
    metric2 = BinaryAUROC()
    
    # Compute k hop graph for 1 random node of class 1
    subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=startNode, num_hops=3, edge_index=data.edge_index, relabel_nodes=False)
    
    w_ij = mlp.forward(modelNodeGNN, data.x, edge_index_hop, startNode)

    # TODO: weights*-1 needed???
    # TODO: PROBLEM: edges start to all get rounded to the same weight, since weights increase/decrease too strongly!?
    edge_ij = mlp.sampleGraph(w_ij, 1).detach()
    
    #print(w_ij)
    
    # Instead of weights, take top k edges according to motif?
    k = num_explanation_edges * 2 if len(w_ij) >= num_explanation_edges*2 else len(w_ij)
    _, top_k_indices = torch.topk(w_ij, k=k, largest=True)
    topEdgesMask = torch.zeros_like(w_ij, dtype=torch.float32)
    topEdgesMask[top_k_indices] = 1
    
    
    """# This correctly generates ground truth for tree-grid/cycle
    ground_truth_indices = []
    labels = data.y
    
    for index in range(0, len(edge_index_hop[0])):
        if labels[edge_index_hop[0][index]] == 1 and labels[edge_index_hop[1][index]] == 1:
            ground_truth_indices.append(index)

    groundTruthMask = torch.zeros_like(w_ij, dtype=torch.bool)
    groundTruthMask[ground_truth_indices] = 1"""
    
    # IDEA: Take gt edge index from dataset, extract subset nodes! Problem: Can't use indices, have to search in edge_index to get nodes
    
    

    # TODO: Use edge_mask from k_hop_subgraph on gt of same length. This should work well but gt (edge_index_mask) MUST match data.edge_index!
    subgraph_ground_truth = data.gt_mask[edge_mask]

    """# TODO: INSTEAD take subgraph_indices from subgraph_edge_mask
    # Get the indices of edges that are part of the subgraph (where edge_mask is True)
    subgraph_indices = edge_mask.nonzero(as_tuple=True)[0]  # This gives the indices where edge_mask is True

    # Filter the ground truth mask to match the subgraph's edges
    filtered_gt = data.gt[subgraph_indices]"""

    
    """# TODO: This shit does not work, gt from dataset
    # Convert larger edge index to a set of tuples for fast lookup
    gt_set = set(zip(ground_truth[0].tolist(), ground_truth[1].tolist()))

    # Create ground truth labels (1 if edge exists in large set, else 0)
    gt_labels = torch.tensor([1 if (u, v) in gt_set else 0 for u, v in zip(edge_index_hop[0], edge_index_hop[1])])
    
    #for index in range(0, len(edge_index_hop[0])):
        # if ground_truth edge index contains edge_index_hop[0][index], edge_index_hop[1][index] at one index => 1"""
            
    
    # This prevents AUC being calculated if all nodes are from the same class, since only either positives or negatives
    if len(torch.unique(topEdgesMask)) == 1 or len(torch.unique(subgraph_ground_truth)) == 1:
        print("AUC not computable")
        return -1

    #print(topEdgesMask)
    #print(groundTruthMask)
    
    fpr, tpr, thresholds = roc(edge_ij, subgraph_ground_truth, task='binary')
    
    metric.update(fpr, tpr)
    metric2.update(edge_ij, subgraph_ground_truth.float())
        
    auc_of_roc = metric.compute().item()
    binaryAUROC = metric2.compute().item()
    roc_auc = roc_auc_score(subgraph_ground_truth, edge_ij)
    print(f"AUC of ROC: {auc_of_roc}")
    print(f"BinaryAUROC: {binaryAUROC}")
    print(f"roc_auc_score: {roc_auc}")
    
    """fpr, tpr, thresholds = roc(topEdgesMask, subgraph_ground_truth, task='binary')
    
    metric.update(fpr, tpr)
    metric2.update(topEdgesMask, subgraph_ground_truth.float())
        
    auc_of_roc = metric.compute().item()
    binaryAUROC = metric2.compute().item()
    roc_auc = roc_auc_score(subgraph_ground_truth, topEdgesMask)
    print(f"AUC of ROC: {auc_of_roc}")
    print(f"BinaryAUROC: {binaryAUROC}")
    print(f"roc_auc_score: {roc_auc}")"""
    
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
        
            # TODO: Pre process data here or before loading to only contain motif graphs
            data = selected_data
            
        graph_dataset_seed = 42
        generator1 = torch.Generator().manual_seed(graph_dataset_seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1], generator1)
        
        meanAuc = evaluateExplainerAUC(mlp, downstreamTask, test_dataset, MUTAG, num_explanation_edges=num_explanation_edges)
        
        # For showExplanation, pass first(TODO: random) graph from test_dataset
        data = test_dataset[random.choice(range(0,len(test_dataset)))]
        
        _ = utils.showExplanation(mlp, downstreamTask, data, num_explanation_edges, motifNodes=None, graphTask=graph_task, MUTAG=MUTAG)
    else:
        # TODO: move to config
        if datasetName == "BA-Community":
            single_label = data.y
            motifNodes = [i for i in range(single_label.shape[0]) if single_label[i] != 0 and single_label[i] != 4]
            
            #allNodes = [i for i in range(len(data.x))]
            #motifNodes = allNodes
        else:
            motif_node_indices = params['motif_node_indices']
            motifNodes = [i for i in range(motif_node_indices[0], motif_node_indices[1], motif_node_indices[2])]
        
        aucList = []
        for i in motifNodes:
            currAuc = evaluateNodeExplainerAUC(mlp, downstreamTask, data, i, num_explanation_edges=num_explanation_edges)
            if currAuc != -1: aucList.append(currAuc)
        meanAuc = torch.tensor(aucList).mean().item()
        
        randomAucNode = utils.showExplanation(mlp, downstreamTask, data, num_explanation_edges, motifNodes, graph_task)
        auc = evaluateNodeExplainerAUC(mlp, downstreamTask, data, randomAucNode, num_explanation_edges=num_explanation_edges)
        print(f"AUC for random Node: {auc}")
        
    return meanAuc