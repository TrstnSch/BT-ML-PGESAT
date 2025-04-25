import datasetLoader
import evaluation
import explainer
import networks
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
    
    torch.use_deterministic_algorithms(True)
    
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
    sample_bias = params['sample_bias']
    num_training_instances = params['num_training_instances']

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
    
    generator_seed = 43
    generator1 = torch.Generator().manual_seed(generator_seed)
    
    
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
            
        total_size = len(data)
        remaining = total_size - num_training_instances
        # To prevent an impossible absolute selection of training instance, default to 0.8,0.1,0.1
        if remaining < 2 or remaining == total_size:
            num_training_instances = 0.8
            num_val = 0.1
            num_test = 0.1
        else:
            num_val = remaining // 2
            num_test = remaining - num_val  # In case of odd number
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [num_training_instances, num_val, num_test], generator1)
        train_loader = DataLoader(train_dataset, params['batch_size'], True)
    else:
        if dataset == "BA-Community":
            single_label = data.y
            motifNodes = [i for i in range(single_label.shape[0]) if single_label[i] != 0 and single_label[i] != 4]
        else:
            motif_node_indices = params['motif_node_indices']
            motifNodes = [i for i in range(motif_node_indices[0], motif_node_indices[1], motif_node_indices[2])]
        
        total_size = len(motifNodes)
        remaining = total_size - num_training_instances
        # To prevent an impossible absolute selection of training instance, default to 0.8,0.1,0.1
        if remaining < 2 or remaining == total_size:
            num_training_instances = 0.8
            num_val = 0.1
            num_test = 0.1
        else:
            num_val = remaining // 2
            num_test = remaining - num_val  # In case of odd number
            
        train_nodes, val_nodes, test_nodes = torch.utils.data.random_split(motifNodes, [0.08, 0.46, 0.46], generator1)
    

    # TODO: Instead of loading static one, pass model as argument?
    downstreamTask = networks.GraphGNN(features = data[0].x.shape[1], labels=labels) if graph_task else networks.NodeGNN(features = 10, labels=labels)
    downstreamTask.load_state_dict(torch.load(f"models/{dataset}", weights_only=True))
    downstreamTask.to(device)

    mlp = explainer.MLP(GraphTask=graph_task, hidden_dim=hidden_dim).to(device)
    wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = lr_mlp, maximize=False)

    downstreamTask.eval()
    for param in downstreamTask.parameters():
        param.requires_grad = False

    training_iterator = train_loader if graph_task else train_nodes
    
    for epoch in range(0, epochs) :
        mlp.train()
        mlp_optimizer.zero_grad()

        temperature = t0*((tT/t0) ** ((epoch+1)/epochs))
        
        current_data = data
        
        sampledEdges = 0.0
        sumSampledEdges = 0.0
        
        samplePredSum = 0

        # If graph task: iterate over training loader with content = current graph. If node task: iterate over motifNodes with content = current node 
        for index, content in enumerate(training_iterator):
            node_to_predict = None
            if graph_task: 
                current_data = content.to(device)

            if not graph_task:
                node_to_predict = content
                subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=node_to_predict, num_hops=3, edge_index=current_data.edge_index, relabel_nodes=False)
                edge_index_hop = edge_index_hop.to(device)

            current_edge_index = current_data.edge_index if graph_task else edge_index_hop
            current_edge_index = current_edge_index.to(device)

            # MLP forward
            # TODO: CHECK IF THE unique_pairs, inverse_indices WORK AS EXPECTED!!!
            w_ij, unique_pairs, inverse_indices = mlp.forward(downstreamTask, current_data.x.to(device), current_edge_index, nodeToPred=node_to_predict)
            
            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, unique_pairs, inverse_indices, temperature, sample_bias)
                
                sampledEdges += torch.sum(edge_ij)
            
                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                # TODO: batch.to(device) not possible for nodes since batch is None. Check if batch.to(device) necessary if current_data is moved to device
                pOriginal = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, current_data.batch), dim=1)
                pSample = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, batch=current_data.batch, edge_weights=edge_ij), dim=1)

                samplePredSum += torch.sum(torch.argmax(pSample, dim=1))
                
                if graph_task:
                    # For graph
                    """for graph_index in range(current_data.num_graphs):
                        node_mask = current_data.batch == graph_index
                        edge_mask = (node_mask[current_edge_index[0]] & node_mask[current_edge_index[1]])

                        # TODO: VALIDATE pOriginal and pSample pass both label predictions, not just correct one
                        currLoss = mlp.loss(pOriginal[graph_index], pSample[graph_index], edge_ij[edge_mask], coefficient_size_reg, coefficient_entropy_reg)
                        sampleLoss += currLoss"""
                    currLoss = mlp.loss(pOriginal, pSample, edge_ij, coefficient_size_reg, coefficient_entropy_reg)
                    sampleLoss += currLoss
                else:
                    # For node
                    currLoss = mlp.loss(pOriginal[node_to_predict], pSample[node_to_predict], edge_ij, coefficient_size_reg, coefficient_entropy_reg, coefficient_L2_reg)
                    sampleLoss += currLoss

            loss += sampleLoss / sampled_graphs
            
            sumSampledEdges += sampledEdges / sampled_graphs

        print(samplePredSum)
        
        loss = loss / len(training_iterator)
        loss.backward()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)

        mlp_optimizer.step()

        mlp.eval()
        
        print("---------------- TRAIN AUC ----------------")
        if graph_task:
            trainAUC, individual_aurocs_train, trainInfTime = evaluation.evaluateExplainerAUC(mlp, downstreamTask, train_dataset, num_explanation_edges)
        else:
            trainAUC, individual_aurocs_train, trainInfTime = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, train_nodes, num_explanation_edges)
        
        #Evaluation on validation set
        print("---------------- VAL AUC ----------------")
        if graph_task:
            valAUC, individual_aurocs_val, valInfTime = evaluation.evaluateExplainerAUC(mlp, downstreamTask, val_dataset, num_explanation_edges)
        else:
            valAUC, individual_aurocs_val, valInfTime = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, val_nodes, num_explanation_edges)
    
        sumSampledEdges = sumSampledEdges / len(training_iterator)
        wandb.log({"train/Loss": loss, "train/AUC": trainAUC, "train/mean_ind_AUC": torch.tensor(individual_aurocs_train).mean(), "val/AUC": valAUC, "val/mean_ind_AUC": torch.tensor(individual_aurocs_val).mean(), "val/sum_sampledEdges": sumSampledEdges, "temperature": temperature})
        
        #print(edge_ij)
    print("---------------- TEST AUC ----------------")
    # Evaluation on test set
    if graph_task:
        testAUC, individual_aurocs_test, testInfTime = evaluation.evaluateExplainerAUC(mlp, downstreamTask, test_dataset, num_explanation_edges)
    else:
        testAUC, individual_aurocs_test, testInfTime = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, test_nodes, num_explanation_edges)
        
    wandb.log({"test/AUC": testAUC, "test/mean_ind_AUC": torch.tensor(individual_aurocs_test).mean(), "test/mean_infTime": testInfTime})
    
    if save_model is True:
        torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{testAUC}_{wandb.run.name}")

    wandb.finish()
    
    return mlp, downstreamTask
