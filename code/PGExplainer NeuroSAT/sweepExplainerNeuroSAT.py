import explainer_NeuroSAT
import utils
import torch
import torch.nn.functional as fn
from torch_geometric import seed
import wandb
import pickle
import NeuroSAT
from pysat.formula import CNF
from pysat.examples.musx import MUSX
from torcheval.metrics.functional import binary_auroc


datasetType = ['BA-2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid', 'NeuroSAT']


def trainExplainer () :
    # HARDCODED STUFF FOR SWEEPING
    dataset="NeuroSAT"
    
    opts = {
        'out_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000',
        'logging': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/log/dataset_train_8_size4000.log',
        'reals_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_reals',
        'gt_edges_per_problem': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_8_size4000_gt_edges_per_problem',
        'n_pairs': 100,  # Anzahl der zu generierenden Paare
        'min_n': 8,
        'max_n': 8,
        'p_k_2': 0.3,
        'p_geo': 0.4,
        'max_nodes_per_batch': 4000,
        'one_pair': False,
        'emb_dim': 128,
        'iterations': 26,
    }
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check valid dataset name
    configOG = utils.loadConfig(dataset)
    if configOG == -1:
        return

    # Config for sweep
    # This works, BUT cannot pass arguments. Dataset therefore has to be hardcoded or passed otherwise?!
    wandb.init(project="Explainer-NeuroSAT-SWEEP", config=wandb.config)
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
    coefficient_consistency = wandb.config.consistency_reg
    bce = bool(wandb.config.bce_loss)
    #num_explanation_edges = params['num_explanation_edges']
    lr_mlp = wandb.config.lr_mlp
    #num_training_instances = wandb.config.num_training_instances
    complex_architecture = bool(wandb.config.complex_architecture)
    three_embs = bool(wandb.config.three_embs)
    adamW = bool(wandb.config.adamW)

    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    min_clip_value = -2
    
    with open(opts['out_dir'], 'rb') as file:
        data = pickle.load(file)
    
    # TODO: Split data into train and test
    # TODO: !!! each data consist of multiple problems !!! -> Extract singular problems for calculating the loss!
    dataset = data
    
    eval_problem = data[-2]
    test_problem = data[-1]
    
    with open(opts['reals_dir'], 'rb') as file:
        reals = pickle.load(file)
        
    """with open(opts['gt_edges_per_problem'], 'rb') as file:
        gt_edges_per_problem = pickle.load(file)"""
    

    downstreamTask = NeuroSAT.NeuroSAT(opts=opts,device=device)
    checkpoint = torch.load(f"models/neurosat_sr10to40_ep1024_nr26_d128_last.pth.tar", weights_only=True, map_location=device)
    downstreamTask.load_state_dict(checkpoint['state_dict'])

    #mlp = explainer_NeuroSAT.MLP_SAT(GraphTask=graph_task, complex_architecture=complex_architecture, three_embs=three_embs).to(device)
    mlp = explainer_NeuroSAT.MLP(GraphTask=graph_task, three_embs=three_embs).to(device)
    wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    if adamW:
        mlp_optimizer = torch.optim.AdamW(params = mlp.parameters(), lr = lr_mlp)
    else:
        mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = lr_mlp)

    downstreamTask.eval()
    for param in downstreamTask.parameters():
        param.requires_grad = False


    training_iterator = dataset
    
    for epoch in range(0, epochs) :
        mlp.train()
        mlp_optimizer.zero_grad()

        temperature = t0*((tT/t0) ** ((epoch+1)/epochs))

        for index, content in enumerate(training_iterator):
            # stop training before second to last batch, used for evaluation
            if index == len(training_iterator)-2: break
            node_to_predict = None
            if graph_task: 
                # !! current_problem is really a batch of problems !!
                current_problem = content

            # MLP forward
            #w_ij, unique_clauses, inverse_indices = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)
            w_ij = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            pOriginal, _, _, _ = downstreamTask.forward(current_problem)
            pOriginal = fn.softmax(pOriginal, dim=0)
            
            #pOriginal = torch.tensor([1 - pOriginal, pOriginal])
            
            for k in range(0, sampled_graphs):
                #edge_ij = mlp.sampleGraph(w_ij, unique_clauses, inverse_indices, temperature)
                edge_ij = mlp.sampleGraph(w_ij, temperature)
                
                #sampledEdges += torch.sum(edge_ij)
            
                # TODO: softmax needed for loss, beacuse negative values do not work witg log! Need for normalization?
                pSample, _, _, _ = downstreamTask.forward(current_problem, edge_weights=edge_ij)
                pSample = fn.softmax(pSample, dim=0)
                #pSample = torch.tensor([1 - pSample, pSample])

                #samplePredSum += torch.sum(torch.argmax(pSample, dim=1))
                
                if graph_task:
                    currLoss = mlp.loss(pOriginal, pSample, edge_ij, current_problem.batch_edges, coefficient_size_reg, coefficient_entropy_reg, coefficientConsistency=coefficient_consistency)
                    sampleLoss.add_(currLoss)
                    

            loss += sampleLoss / sampled_graphs
        
        loss = loss / ((len(training_iterator)-2)*len(data[0].is_sat))
        loss.backward()
        
        mlp_optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)
        
        """for param in mlp.parameters():
            if param.grad is not None:
                param.grad.data = torch.max(param.grad.data, min_clip_value * torch.ones_like(param.grad.data))"""

        mlp.eval()
        
        # Calculate weights and prediction for all sub_problems in eval_problem
        #w_ij_eval, unique_clauses_eval, inverse_indices_eval = mlp.forward(downstreamTask, eval_problem, nodeToPred=node_to_predict)
        w_ij_eval = mlp.forward(downstreamTask, eval_problem, nodeToPred=node_to_predict)
        #edge_ij_eval = mlp.sampleGraph(w_ij_eval, unique_clauses_eval, inverse_indices_eval, temperature).detach()
        edge_ij_eval = mlp.sampleGraph(w_ij_eval).detach()
        #pSample_eval, _, _, _ = downstreamTask.forward(eval_problem, edge_weights=edge_ij_eval)
        #pOriginal_eval, _, _, _ = downstreamTask.forward(eval_problem)
        
        auroc_list = []
        
        valLoss = torch.FloatTensor([0]).to(device)
        
        for current_batch_num in range(len(eval_problem.is_sat)):
            # This can be repeated for each sub_problem
            #clauses_current_problem = eval_problem.get_clauses_for_problem(current_batch_num)

            #print(f"Length of sub_problem 0: {len(clauses_current_problem)}")
            # Sub problem 1 Contains literals 11-20, ...
            #print(f"Sub_problem 0 clauses: {clauses_current_problem}")
            
            # Calculate mask for current sub_problem in eval_problem
            eval_batch_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=current_batch_num, batch_size=opts['min_n'], n_variables=eval_problem.n_variables)
            
            # Edge probabilites for current sub problem in eval_problems
            edge_ij_eval_masked = edge_ij_eval[eval_batch_mask]
            
            #sub_problem_edges = eval_problem.batch_edges[eval_batch_mask]
            
            curr_auroc = binary_auroc(edge_ij_eval_masked, reals[current_batch_num])
            auroc_list.append(curr_auroc.item())
            
            
        pOriginal, _, _, _ = downstreamTask.forward(eval_problem)
        pOriginal = fn.softmax(pOriginal, dim=0)
        
        pSample, _, _, _ = downstreamTask.forward(eval_problem, edge_weights=edge_ij_eval)
        pSample = fn.softmax(pSample, dim=0)
        
        valLoss = mlp.loss(pOriginal, pSample, edge_ij_eval, eval_problem.batch_edges, coefficient_size_reg, coefficient_entropy_reg, coefficientConsistency=coefficient_consistency)
            
        valLoss = valLoss / len(eval_problem.is_sat)
        
        auroc_tensor = torch.tensor(auroc_list)
        mean_auroc = auroc_tensor.mean()
        if epoch == epochs-1:
            highest_auroc_index = torch.argmax(auroc_tensor)
            #highest_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=highest_auroc_index, batch_size=opts['min_n'], n_variables=eval_problem.n_variables)
            highest_auroc = auroc_tensor[highest_auroc_index]
            wandb.log({"val/highest_individual_auroc": highest_auroc})
            """highest_edge_ij = edge_ij_eval[highest_mask]
            highest_gt = reals[highest_auroc_index]
            sub_problem_edges_highest = eval_problem.batch_edges[highest_mask]"""
            
            """lowest_auroc_index = torch.argmin(auroc_tensor)
            lowest_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=lowest_auroc_index, batch_size=opts['min_n'], n_variables=eval_problem.n_variables)
            lowest_auroc = auroc_tensor[lowest_auroc_index]
            lowest_edge_ij = edge_ij_eval[lowest_mask]
            lowest_gt = reals[lowest_auroc_index]
            sub_problem_edges_lowest = eval_problem.batch_edges[lowest_mask]"""
                
        #weights_eval_masked = w_ij_eval[eval_batch_mask]
        #print(f"Edge weights for last sub_problem in last problem batch: {weights_eval_masked}")
        
        #print(f"Edge probabilites for last sub_problem in last problem batch: {edge_ij_eval_masked}")
        
        #print(f"Prediction for sub_problem: {pOriginal_eval[0]}")
        #print(f"Prediction for sampled sub_problem: {pSample_eval[0]}")
        
        print(f"mean auroc score: {mean_auroc}")
        
        wandb.log({"train/Loss": loss, "val/Loss": valLoss, "val/auroc": mean_auroc, "edge_importance/min": edge_ij_eval.min().item(), "edge_importance/max": edge_ij_eval.max().item(),
                "edge_importance/mean": edge_ij_eval.mean().item()})
                
    wandb.finish()
    
    return mlp, downstreamTask