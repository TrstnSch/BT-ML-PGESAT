import datasetLoader
import evaluation
import explainer
import networks
import utils
import sys
import torch
import torch.nn.functional as fn
from torch_geometric.loader import DataLoader
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

#dataset, save_model=False
def trainExplainer (dataset, save_model=False) :
    
    
    
    
    
    # CAREFUL FOR WANDB SWEEP dataset = "MUTAG"
    #dataset = "Tree-Cycles"
    
    
    
    
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

    wandb.init(project="Explainer-Training", config=params)

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
    

    #wandb.init(project="Explainer-Training", config=params)
    #wandb.init(project="Explainer-Training", config=sweepConfig)
    
    data, labels = datasetLoader.loadGraphDataset(dataset) if graph_task else datasetLoader.loadOriginalNodeDataset(dataset)
    
    if graph_task:
        #TODO: FOR MUTAG: SELECT GRAPHS WITH GROUND TRUTH. HOW TO EVALUATE ONLY ON THESE? CHANGE TRAINING AGAIN AND REMOVE DATALOADERS?!
        #if np.argmax(original_labels[gid]) == 0 and np.sum(edge_label_lists[gid]) > 0:
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

        train_loader = DataLoader(train_dataset, params['batch_size'], True)
        #val_loader = DataLoader(val_dataset, params['batch_size'], False)
        #test_loader = DataLoader(test_dataset, params['batch_size'])
    else:
        # TODO
        if dataset == "BA-Shapes":
            motifNodesOriginal = [i for i in range(400,700,5)]
            allNodes = [i for i in range(len(data.x))]
            
            motifNodes = motifNodesOriginal
        
        if dataset == "BA-Community":
            # TODO: Validate this
            single_label = data.y
            #print(all_label)
            #single_label = torch.argmax(all_label,axis=-1)
            #print(single_label)
            motifNodesOriginal = [i for i in range(single_label.shape[0]) if single_label[i] != 0 and single_label[i] != 4]
            
            allNodes = [i for i in range(len(data.x))]
            
            motifNodes = motifNodesOriginal
            
        if dataset == "Tree-Cycles":
            motifNodesOriginal = [i for i in range(511,871,6)]          # It LOOKS like this takes only the first node of each motif
            motifNodesModified = [i for i in range(len(data.x)) if data.y[i] == 1 ]
            allNodes = [i for i in range(len(data.x))]
        
            motifNodes = motifNodesOriginal
            
        if dataset == "Tree-Grid":
            motifNodesOriginal = [i for i in range(511,800,9)]              #in(512,514,515,516,518) out(511,513, 517,519)                   range(511,800,1)
            motifNodesNew = [i for i in range(len(data.x)) if data.y[i] == 1 ]
            allNodes = [i for i in range(len(data.x))]

            # motifNodesModified is supposed to only contain the "middle" elements of the motif, for which the 3-hop graph contains the complete motif 
            motifNodesModified = [i for i in range(512,800,9)]
            trainableNodes = [514, 515, 516, 518]

            motifNodesMinimal = [i for i in range(512,800,9)]
            motifNodesMinimal.extend([i for i in range(514,800,9)])
            motifNodesMinimal = sorted(set(motifNodesMinimal))

            def add_multiples_of_9(start, end=800, step=9):
                return [i for i in range(start, end, step)]

            for element in trainableNodes:
                motifNodesModified.extend(add_multiples_of_9(element))
                
            motifNodesModified = sorted(set(motifNodesModified))
            
            motifNodes = motifNodesOriginal
        


    # TODO: Instead of loading static one, pass model as argument?
    downstreamTask = networks.GraphGNN(features = train_dataset[0].x.shape[1], labels=labels) if graph_task else networks.NodeGNN(features = 10, labels=labels)
    downstreamTask.load_state_dict(torch.load(f"models/{dataset}", weights_only=True))

    mlp = explainer.MLP(GraphTask=graph_task, hidden_dim=hidden_dim)
    #wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = lr_mlp, maximize=False)

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
                # TODO: CHANGE TO content instead of index. VALIDATE THIS!!!! For index AUC seems to decrease
                node_to_predict = content

            current_edge_index = current_data.edge_index if graph_task else edge_index_hop

            # MLP forward
            # TODO: Why node_to_predict = index for node task instead of content? Should probably be content!
            w_ij = mlp.forward(downstreamTask, current_data.x, current_edge_index, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0])
            loss = torch.FloatTensor([0])
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, temperature)
            
                # TODO: Check if current_data.batch works with nodes! Add batch support for nodes? Batch has to contain map for edge_index?
                pOriginal = fn.softmax(downstreamTask.forward(current_data.x, current_edge_index, current_data.batch), dim=1)
                pSample = fn.softmax(downstreamTask.forward(current_data.x, current_edge_index, batch=current_data.batch, edge_weights=edge_ij), dim=1)


                #TODO: Potential problem: Does not correctly take into account if pOriginal is a different class than pSample????
                
                
                
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
        
        # TODO: Try if this does something, clip grad for -2. Does not seem to do anything
        for param in mlp.parameters():
            if param.grad is not None:
                param.grad.data = torch.max(param.grad.data, min_clip_value * torch.ones_like(param.grad.data))

        mlp_optimizer.step()

        mlp.eval()
        
        if graph_task:
            # TODO: IF MUTAG val_dataset should be changed to dataset with only motifs/pass indices
            meanAuc = evaluation.evaluateExplainerAUC(mlp, downstreamTask, val_dataset, num_explanation_edges)
        else:
            # TODO: CHECK IF THIS IS RESET CORRECTLY?!?!?!?! Should be though
            aucList = []
            for i in motifNodes:
                currAuc = evaluation.evaluateNodeExplainerAUC(mlp, downstreamTask, data, i, num_explanation_edges)
                if currAuc != -1: aucList.append(currAuc)
            meanAuc = torch.tensor(aucList).mean().item()
            print(f"Mean auc: {meanAuc}")
    
        wandb.log({"train/Loss": loss, "val/mean_AUC": meanAuc})

    #if save_model:
    #    torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{meanAuc}_{wandb.run.name}")

    wandb.finish()
    
    return mlp, downstreamTask


#trainExplainer(dataset=sys.argv[1], save_model=sys.argv[2])

