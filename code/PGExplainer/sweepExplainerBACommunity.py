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


def trainExplainer () :
    dataset="BA-Community"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check valid dataset name
    configOG = utils.loadConfig(dataset)
    if configOG == -1:
        return

    # Config for sweep
    # This works, BUT cannot pass arguments. Dataset therefore has to be hardcoded or passed otherwise?!
    wandb.init(project="Explainer-BA-Community-Sweep-Fin", config=wandb.config)
    seed.seed_everything(wandb.config.seed)
    
    params = configOG['params']
    graph_task = params['graph_task']
    epochs = wandb.config.epochs
    t0 = wandb.config.t0
    tT = wandb.config.tT
    sampled_graphs = wandb.config.sampled_graphs
    coefficient_size_reg = wandb.config.size_reg
    coefficient_entropy_reg = wandb.config.entropy_reg
    coefficient_L2_reg = wandb.config.L2_reg
    num_explanation_edges = params['num_explanation_edges']
    lr_mlp = wandb.config.lr_mlp
    sample_bias = wandb.config.sample_bias
    num_training_instances = wandb.config.num_training_instances


    MUTAG = True if dataset == "MUTAG" else False
    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    min_clip_value = -2 # Not utilized
    
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
    #wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

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
            w_ij, unique_pairs, inverse_indices = mlp.forward(downstreamTask, current_data.x.to(device), current_edge_index, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, unique_pairs, inverse_indices, temperature, sample_bias)
            
                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                pOriginal = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, current_data.batch), dim=1)
                pSample = fn.softmax(downstreamTask.forward(current_data.x.to(device), current_edge_index, batch=current_data.batch, edge_weights=edge_ij), dim=1)
                
                if graph_task:
                    # For graph
                    """for graph_index in range(current_data.num_graphs):
                        node_mask = current_data.batch == graph_index
                        edge_mask = (node_mask[current_edge_index[0]] & node_mask[current_edge_index[1]])

                        # TODO: VALIDATE pOriginal and pSample pass both label predictions, not just correct one
                        currLoss = mlp.loss(pOriginal[graph_index], pSample[graph_index], edge_ij[edge_mask], coefficient_size_reg, coefficient_entropy_reg)
                        sampleLoss += currLoss"""
                    currLoss = mlp.loss(pOriginal, pSample, edge_ij, coefficient_size_reg, coefficient_entropy_reg, paperLoss=bool(wandb.config.paper_loss))
                    sampleLoss += currLoss
                else:
                    # For node
                    currLoss = mlp.loss(pOriginal[node_to_predict], pSample[node_to_predict], edge_ij, coefficient_size_reg, coefficient_entropy_reg, coefficient_L2_reg, paperLoss=bool(wandb.config.paper_loss))
                    sampleLoss += currLoss

            loss += sampleLoss / sampled_graphs
        
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
            
        wandb.log({"train/Loss": loss, "train/AUC": trainAUC, "train/mean_ind_AUC": torch.tensor(individual_aurocs_train).mean(), "val/AUC": valAUC, "val/mean_ind_AUC": torch.tensor(individual_aurocs_val).mean()})
        
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

    wandb.finish()
    
    return mlp, downstreamTask