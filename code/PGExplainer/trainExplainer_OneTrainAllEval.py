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
import pandas as pd



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


def trainExplainer (dataset, save_model=False, wandb_project="Experiment-Replication",runSeed=None, collective=False):
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
        
        # COLLECTIVE
        if collective:
            train_dataset = data
        else:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [num_training_instances, num_val, num_test], generator1)
            val_loader = DataLoader(val_dataset, params['batch_size'], False)
            
        train_loader = DataLoader(train_dataset, params['batch_size'], True)
        
    else:
        if dataset == "BA-Community":
            single_label = data.y
            allMotifNodes = [i for i in range(single_label.shape[0]) if single_label[i] != 0 and single_label[i] != 4]
            
            # USED TO EVAL One of the two middele house nodes!
            # Calc all two middle nodes of each motif
            middleCommunityNodes = [i for i in range(single_label.shape[0]) if single_label[i] == 1 or single_label[i] == 5]
            # Since middle nodes are next to each other, select every second node
            oneMotifNodes = [_ for i,_ in enumerate(middleCommunityNodes) if i%2 == 0]
        else:
            motif_node_indices = params['motif_node_indices']
            allMotifNodes = [i for i in range(motif_node_indices[0], motif_node_indices[1], 1)]
            if dataset == "Tree-Grid":
                oneMotifNodes = [i for i in range(512,800,9)]
            else:
                oneMotifNodes = [i for i in range(motif_node_indices[0], motif_node_indices[1], motif_node_indices[2])]
        
        total_size = len(oneMotifNodes)
        remaining = total_size - num_training_instances
        # To prevent an impossible absolute selection of training instance, default to 0.8,0.1,0.1
        if remaining < 2 or remaining == total_size:
            num_training_instances = 0.8
            num_val = 0.1
            num_test = 0.1
        else:
            num_val = remaining // 2
            num_test = remaining - num_val  # In case of odd number
            
        if collective:
            train_nodes = oneMotifNodes
        else:
            train_nodes, val_nodes, test_nodes = torch.utils.data.random_split(oneMotifNodes, [num_training_instances, num_val, num_test], generator1)
            
            # REMOVE train_nodes FROM allMotifNodes and split in half for validation and test set
            remainingNodes = [i for i in allMotifNodes if i not in train_nodes]
            val_nodes, test_nodes = torch.utils.data.random_split(remainingNodes, [0.5, 0.5], generator1)

    

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
    
    # Prepare predictions for both and computation graphs for node task
    # THIS LEADS TO WORSE RESULTS FOR E.G. MUTAG!!!
    # PROBABLY A PROBLEM WITH SHAPES AND INDEXING!!!
    """current_data = data
    train_subgraphs = []
    original_preds = []
    for index, content in enumerate(training_iterator):
        if graph_task: 
            current_data = content.to(device)
        else:
            node_to_predict = content
            subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=node_to_predict, num_hops=3, edge_index=current_data.edge_index, relabel_nodes=False)
            edge_index_hop = edge_index_hop.to(device)
            
            train_subgraphs.append(edge_index_hop)
            
        current_edge_index = current_data.edge_index if graph_task else edge_index_hop
        current_edge_index = current_edge_index.to(device)
                
        pOriginal = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, current_data.batch), dim=1)
        if graph_task:
            original_preds.append(pOriginal)
        else:
            original_preds.append(pOriginal[node_to_predict])"""
        
    
    results = []
    
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
            
            pOriginal = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, current_data.batch), dim=1)
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, unique_pairs, inverse_indices, temperature, sample_bias)
                
                sampledEdges += torch.sum(edge_ij)
            
                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                # TODO: batch.to(device) not possible for nodes since batch is None. Check if batch.to(device) necessary if current_data is moved to device
                pSample = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, batch=current_data.batch, edge_weights=edge_ij), dim=1)

                samplePredSum += torch.sum(torch.argmax(pSample, dim=1))
                
                if graph_task:
                    # For graph
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

        if not collective:
            #Evaluation on validation set
            print("---------------- VAL AUC ----------------")
            if graph_task:
                valAUC, individual_aurocs_val, valInfTime = evaluation.evaluateExplainerAUC(mlp, downstreamTask, val_dataset, num_explanation_edges)
            else:
                valAUC, individual_aurocs_val, valInfTime = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, val_nodes, num_explanation_edges)
        
            sumSampledEdges = sumSampledEdges / len(training_iterator)
            
            # VAL loss
            valLoss = torch.FloatTensor([0]).to(device)
            
            validation_iterator = val_loader if graph_task else val_nodes
            for index, content in enumerate(validation_iterator):
                node_to_predict = None
                if graph_task: 
                    current_data = content.to(device)

                if not graph_task:
                    node_to_predict = content
                    subset, edge_index_hop, mapping, edge_mask = k_hop_subgraph(node_idx=node_to_predict, num_hops=3, edge_index=current_data.edge_index, relabel_nodes=False)
                    edge_index_hop = edge_index_hop.to(device)

                current_edge_index = current_data.edge_index if graph_task else edge_index_hop
                current_edge_index = current_edge_index.to(device)
                
                w_ij, unique_pairs, inverse_indices = mlp.forward(downstreamTask, current_data.x.to(device), current_edge_index, nodeToPred=node_to_predict)
                
                pOriginal = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, current_data.batch), dim=1)

                edge_ij = mlp.sampleGraph(w_ij, unique_pairs, inverse_indices, temperature, sample_bias)
                pSample = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, batch=current_data.batch, edge_weights=edge_ij), dim=1)

                if graph_task:
                    valLoss += mlp.loss(pOriginal, pSample, edge_ij, coefficient_size_reg, coefficient_entropy_reg)
                else:
                    valLoss += mlp.loss(pOriginal[node_to_predict], pSample[node_to_predict], edge_ij, coefficient_size_reg, coefficient_entropy_reg, coefficient_L2_reg)
                    
            
                    
        if collective:
            wandb.log({"train/Loss": loss, "train/AUC": trainAUC, "train/mean_ind_AUC": torch.tensor(individual_aurocs_train).mean(), "temperature": temperature})
        else:    
            valLoss = valLoss / len(validation_iterator)
            
            results.append({
                "epoch": epoch,
                "val_loss": valLoss.item(),
                "val_auroc": torch.tensor(individual_aurocs_val).mean().item(),
                "run": f"{dataset}_{runSeed}"
            })
            
            wandb.log({"train/Loss": loss, "train/AUC": trainAUC, "train/mean_ind_AUC": torch.tensor(individual_aurocs_train).mean(), "val/AUC": valAUC, "val/mean_ind_AUC": torch.tensor(individual_aurocs_val).mean(), "val/sum_sampledEdges": sumSampledEdges, "val/loss": valLoss,"temperature": temperature})
            
    if not collective:
        print("---------------- TEST AUC ----------------")
        # Evaluation on test set
        if graph_task:
            testAUC, individual_aurocs_test, testInfTime = evaluation.evaluateExplainerAUC(mlp, downstreamTask, test_dataset, num_explanation_edges)
        else:
            testAUC, individual_aurocs_test, testInfTime = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, test_nodes, num_explanation_edges)
            
        if graph_task:
            test_data = test_dataset[-1]
            w_ij_test, unique_pairs_test, inverse_indices_test = mlp.forward(downstreamTask, test_data.x, test_data.edge_index)
            edge_ij_test = mlp.sampleGraph(w_ij_test, unique_pairs_test, inverse_indices_test).detach()
        else:
            test_node = test_nodes[-1]
            subset, edge_index_hop_test, mapping, edge_mask = k_hop_subgraph(node_idx=test_node, num_hops=3, edge_index=data.edge_index, relabel_nodes=False)

            w_ij_test, unique_pairs_test, inverse_indices_test = mlp.forward(downstreamTask, data.x, edge_index_hop_test, test_node)
            edge_ij_test = mlp.sampleGraph(w_ij_test, unique_pairs_test, inverse_indices_test).detach()
            
        wandb.log({"test/AUC": testAUC, "test/mean_ind_AUC": torch.tensor(individual_aurocs_test).mean(), 
                "test/mean_infTime": testInfTime, "edge_importance/min": edge_ij_test.min().item(), "edge_importance/max": edge_ij_test.max().item(),
                "edge_importance/mean": edge_ij_test.mean().item()})
        
    #,"test/edge_importance_histogram": wandb.Histogram(edge_ij_test.cpu().numpy())
    
    if save_model is True:
        torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{testAUC}_{wandb.run.name}")

    wandb.finish()
    
    if collective:
        return mlp, downstreamTask, trainAUC, individual_aurocs_train, trainInfTime
    else:
        df = pd.DataFrame(results)
        df.to_csv(f"{dataset}_{runSeed}.csv", index=False)
        
        return mlp, downstreamTask, testAUC, individual_aurocs_test, testInfTime
    
