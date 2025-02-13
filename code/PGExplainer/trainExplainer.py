import datasetLoader
import evaluation
import explainer
import json
import networks
import sys
import torch
import torch.nn.functional as fn
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
import wandb


datasetType = ['BA-2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid']

def trainExplainer (dataset, save_model=False) :
    # Check valid dataset name
    if dataset not in datasetType:
        print("Dataset argument must be one of the following: BA-2Motif, MUTAG, BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid")
        return
    
    config = loadConfig(dataset)
    params = config['params']
    graph_task = params['graph_task']
    epochs = params['epochs']
    t0 = params['t0']
    tT = params['tT']
    sampled_graphs = params['sampled_graphs']
    coefficient_size_reg = params['coefficient_size_reg']
    coefficient_entropy_reg = params['coefficient_entropy_reg']
    coefficient_L2_reg = params['coefficient_L2_reg']

    MUTAG = True if dataset == "MUTAG" else False
    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    

    wandb.init(project="Explainer-Training", config=config)

    data, labels = datasetLoader.loadGraphDataset(dataset) if graph_task else datasetLoader.loadOriginalNodeDataset(dataset)

    if graph_task:
        graph_dataset_seed = 42
        generator1 = torch.Generator().manual_seed(graph_dataset_seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1], generator1)

        train_loader = DataLoader(train_dataset, params['batch_size'], True)
        val_loader = DataLoader(val_dataset, params['batch_size'], False)
        test_loader = DataLoader(test_dataset, params['batch_size'])
    else:
        # TODO
        motifNodes = [i for i in range(511,800,1)]


    downstreamTask = networks.GraphGNN(features = train_dataset[0].x.shape[1], labels=2) if graph_task else networks.NodeGNN(features = data.x.shape[1], labels=labels)
    downstreamTask.load_state_dict(torch.load(f"models/{dataset}", weights_only=True))

    mlp = explainer.MLP(GraphTask=graph_task, hidden_dim=hidden_dim)
    wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = params['lr_mlp'], maximize=False)

    downstreamTask.eval()
    for param in downstreamTask.parameters():
        param.requires_grad = False


    training_iterator = train_loader if graph_task else motifNodes

    for epoch in range(0, epochs) :
        mlp.train()
        mlp_optimizer.zero_grad()

        temperature = t0*((tT/t0) ** ((epoch+1)/epochs))

        current_data = data

        # If graph task: iterate over training loader with content = current graph. If node task: iterate over motifNodes with content = current node 
        for index, content in enumerate(training_iterator):
            node_to_predict = None
            if graph_task: current_data = content

            if not graph_task:
                subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=content, num_hops=3, edge_index=current_data.edge_index, relabel_nodes=False)
                node_to_predict = index

            current_edge_index = current_data.edge_index if graph_task else edge_index_hop

            # MLP forward
            w_ij = mlp.forward(downstreamTask, current_data.x, current_edge_index, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0])
            loss = torch.FloatTensor([0])
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, temperature)

                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                pOriginal = fn.softmax(downstreamTask.forward(current_data.x, current_edge_index, current_data.batch), dim=1)
                pSample = fn.softmax(downstreamTask.forward(current_data.x, current_edge_index, current_data.batch, edge_weights=edge_ij), dim=1)

                if graph_task:
                    # For graph
                    for graph_index in range(current_data.num_graphs):
                        node_mask = current_data.batch == graph_index
                        edge_mask = (node_mask[current_edge_index[0]] & node_mask[current_edge_index[1]])

                        # TODO: VALIDATE pOriginal and pSample pass both label predictions, not just correct one
                        currLoss = mlp.loss(pOriginal[graph_index], pSample[graph_index], edge_ij[edge_mask], coefficient_size_reg, coefficient_entropy_reg)
                        sampleLoss += currLoss
                else:
                    # For node
                    currLoss = mlp.loss(pOriginal[content], pSample[content], edge_ij, coefficient_size_reg, coefficient_entropy_reg, coefficient_L2_reg)
                    sampleLoss += currLoss

            loss += sampleLoss / sampled_graphs

        loss = loss / len(training_iterator)
        loss.backward()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)

        mlp_optimizer.step()

        mlp.eval()
        
        if graph_task:
            dataOut, meanAuc = evaluation.evaluateExplainerAUC(mlp, downstreamTask, val_dataset, MUTAG)
        else:
            aucList = []
            for i in motifNodes:
                aucList.append(evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, data.edge_index, i, data.gt))
            meanAuc = torch.tensor(aucList).mean().item()
    
        wandb.log({"train/Loss": loss, "val/mean_AUC": meanAuc})

    wandb.finish()

    if save_model:
        torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{meanAuc}_{wandb.run.name}")

    return mlp


def loadConfig(dataset):
    # Load JSON file "dataset"
    script_dir = Path(__file__).resolve().parent
    config_dir = f"configs/{dataset}.json"

    with open(script_dir.parent.parent / config_dir) as f:
        config = json.load(f)

    return config


trainExplainer(dataset=sys.argv[1], save_model=sys.argv[2])

