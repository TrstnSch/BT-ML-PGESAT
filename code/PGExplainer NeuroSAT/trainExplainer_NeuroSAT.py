import datasetLoader
import evaluation
import explainer_NeuroSAT
import NeuroSAT
import utils
import sys
import torch
import torch.nn.functional as fn
from torch_geometric.loader import DataLoader
from torch_geometric import seed
from torch_geometric.utils import k_hop_subgraph
import wandb



datasetType = ['BA-2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid']


def loadExplainer(dataset):
    # Check valid dataset name
    config = utils.loadConfig(dataset)
    if config == -1:
        return
    
    params = config['params']
    graph_task = params['graph_task']
    
    data, labels = datasetLoader.loadGraphDataset(dataset) if graph_task else datasetLoader.loadOriginalNodeDataset(dataset)
    
    mlp = explainer.MLP(GraphTask=graph_task, hidden_dim=64)     # Adjust according to data and task
    mlp.load_state_dict(torch.load(f"models/explainer{dataset}", weights_only=True))

    downstreamTask = networks.GraphGNN(features = data[0].x.shape[1], labels=labels) if graph_task else networks.NodeGNN(features = data.x.shape[1], labels=labels)
    downstreamTask.load_state_dict(torch.load(f"models/{dataset}", weights_only=True))
    
    return mlp, downstreamTask




def trainExplainer (dataset, save_model=False, wandb_project="Experiment-Replication",runSeed=None) :
    if runSeed is not None: seed.seed_everything(runSeed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check valid dataset name
    configOG = utils.loadConfig(dataset)
    if configOG == -1:
        return
    

    params = configOG['params']
    graph_task = params['graph_task']
    epochs = params['epochs']
    t0 = params['t0']
    tT = params['tT']
    sampled_graphs = params['sampled_graphs']
    coefficient_size_reg = params['coefficient_size_reg']
    coefficient_entropy_reg = params['coefficient_entropy_reg']
    coefficient_L2_reg = params['coefficient_L2_reg']
    num_explanation_edges = params['num_explanation_edges']
    lr_mlp = params['lr_mlp']

    wandb.init(project=wandb_project, config=params)

    # Config for sweep
    # This works, BUT cannot pass arguments. Dataset therefore has to be hardcoded or passed otherwise?!
    """wandb.init(project="Explainer-Tree-Cycles-Sweep", config=wandb.config)
    
    params = configOG['params']
    graph_task = params['graph_task']
    epochs = params['epochs']
    t0 = params['t0']
    tT = wandb.config.tT
    sampled_graphs = params['sampled_graphs']
    coefficient_size_reg = wandb.config.size_reg
    coefficient_entropy_reg = wandb.config.entropy_reg
    coefficient_L2_reg = params['coefficient_L2_reg']
    num_explanation_edges = params['num_explanation_edges']
    lr_mlp = wandb.config.lr_mlp"""


    MUTAG = True if dataset == "MUTAG" else False
    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    min_clip_value = -2
    
    data, labels = datasetLoader.loadGraphDataset(dataset) if graph_task else datasetLoader.loadOriginalNodeDataset(dataset)
    
    if graph_task:
        # FOR MUTAG: SELECT GRAPHS WITH GROUND TRUTH
        if MUTAG:
            selected_data = []
            selectedIndices = []
            for i in range(0, len(data)):
                if data[i].y == 0 and torch.sum(data[i].gt_mask) > 0:
                    selectedIndices.append(i)
                    selected_data.append(data[i])
        
            data = selected_data

        #graph_dataset_seed = 43
        #generator1 = torch.Generator().manual_seed(graph_dataset_seed)
        #train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1], generator1)
        
        train_loader = DataLoader(data, params['batch_size'], True)
        #val_loader = DataLoader(val_dataset, params['batch_size'], False)
        #test_loader = DataLoader(test_dataset, params['batch_size'])
        

    downstreamTask = NeuroSAT.NeuroSAT(opts=,device=device)
    downstreamTask.load_state_dict(torch.load(f"models/neurosat_sr10to40_ep1024_nr26_d128_last.pth.tar", weights_only=True))

    mlp = explainer_NeuroSAT.MLP(GraphTask=graph_task, hidden_dim=hidden_dim).to(device)
    wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = lr_mlp, maximize=False)

    downstreamTask.eval()
    for param in downstreamTask.parameters():
        param.requires_grad = False

    # TODO: Adapt for problem loader!
    training_iterator = train_loader
    
    for epoch in range(0, epochs) :
        mlp.train()
        mlp_optimizer.zero_grad()

        temperature = t0*((tT/t0) ** ((epoch+1)/epochs))
        
        sampledEdges = 0.0
        sumSampledEdges = 0.0
        
        samplePredSum = 0

        # If graph task: iterate over training loader with content = current graph. If node task: iterate over motifNodes with content = current node 
        # TODO: For NeuroSAT iterate over Problems
        for index, content in enumerate(training_iterator):
            node_to_predict = None
            if graph_task: 
                current_problem = content.to(device)

            # MLP forward
            w_ij = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, temperature)
                
                sampledEdges += torch.sum(edge_ij)
            
                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                pOriginal, _, _, _ = downstreamTask.forward(current_problem)
                pSample, _, _, _ = downstreamTask.forward(current_problem, edge_weights=edge_ij)

                samplePredSum += torch.sum(torch.argmax(pSample, dim=1))
                
                if graph_task:
                    currLoss = mlp.loss(pOriginal, pSample, edge_ij, coefficient_size_reg, coefficient_entropy_reg)
                    sampleLoss += currLoss

            loss += sampleLoss / sampled_graphs
            
            sumSampledEdges += sampledEdges / sampled_graphs

        print(samplePredSum)
        
        loss = loss / len(training_iterator)
        loss.backward()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)
        
        # TODO: Try if this does something, clip grad for -2. Does not seem to do anything
        """for param in mlp.parameters():
            if param.grad is not None:
                param.grad.data = torch.max(param.grad.data, min_clip_value * torch.ones_like(param.grad.data))"""

        mlp_optimizer.step()

        mlp.eval()
        
        if graph_task:
            meanAuc = evaluation.evaluateNeuroSATAUC(mlp, downstreamTask, data)
    
        sumSampledEdges = sumSampledEdges / len(training_iterator)
        wandb.log({"train/Loss": loss, "val/mean_AUC": meanAuc, "val/sum_sampledEdges": sumSampledEdges, "val/temperature": temperature})

        """for name, param in mlp.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad}")"""
        
    if save_model == "True":
        torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{meanAuc}_{wandb.run.name}")

    wandb.finish()
    
    return mlp, downstreamTask
