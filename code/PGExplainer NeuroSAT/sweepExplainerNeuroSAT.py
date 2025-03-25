import datasetLoader
import evaluation
import explainer_NeuroSAT
import networks
import utils
import sys
import torch
import torch.nn.functional as fn
from torch_geometric.loader import DataLoader
from torch_geometric import seed
from torch_geometric.utils import k_hop_subgraph
import wandb
import pickle
from pysat.solvers import Solver
import numpy as np
import NeuroSAT
from sklearn.metrics import roc_auc_score


datasetType = ['BA-2Motif','MUTAG', 'BA-Shapes', 'BA-Community', 'Tree-Cycles', 'Tree-Grid', 'NeuroSAT']


def trainExplainer () :
    # HRADCODED STUFF FOR SWEEPING
    dataset="NeuroSAT"
    
    opts = {
        'out_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_train_10',
        'logging': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/log/dataset_train_10.log',
        'n_pairs': 100,  # Anzahl der zu generierenden Paare
        'min_n': 10,
        'max_n': 10,
        'p_k_2': 0.3,
        'p_geo': 0.4,
        'max_nodes_per_batch': 12000,
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
    wandb.init(project="Explainer-NeuroSAT-Sweep", config=wandb.config)
    seed.seed_everything(wandb.config.seed)
    
    params = configOG['params']
    graph_task = params['graph_task']
    epochs = wandb.config.epochs
    t0 = params['t0']
    tT = wandb.config.tT
    sampled_graphs = params['sampled_graphs']
    coefficient_size_reg = wandb.config.size_reg
    coefficient_entropy_reg = wandb.config.entropy_reg
    coefficient_L2_reg = params['coefficient_L2_reg']
    num_explanation_edges = params['num_explanation_edges']
    lr_mlp = wandb.config.lr_mlp

    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    min_clip_value = -2
    
    with open(opts['out_dir'], 'rb') as file:
        data = pickle.load(file)
    
    # TODO: Split data into train and test
    # TODO: !!! each data consist of multiple problems !!! -> Extract singular problems for calculating the loss!
    dataset = data
    
    eval_problem = data[-1]
    
    # gt only needs to be calced once, not per epoch!
    countClauses = 0
    reals = []
    for j in range(len(eval_problem.is_sat)):
            # This can be repeated for each sub_problem
            current_batch_num = j
            
            clauses_current_problem = eval_problem.get_clauses_for_problem(current_batch_num)

            #print(f"Length of sub_problem 0: {len(clauses_current_problem)}")
            # Sub problem 1 Contains literals 11-20, ...
            #print(f"Sub_problem 0 clauses: {clauses_current_problem}")
            
            # Next part is only for calculating unsat_core -> Move to creation of data and save?
            solver = Solver(name='m22')

            offset = eval_problem.n_literals + 1

            # Assumptions must be unique, therefore eval_problem.n_literals + 1
            assumptions = [i + offset for i in range(len(clauses_current_problem))]

            # Add the clauses with selector literals
            for i, clause in enumerate(clauses_current_problem):
                solver.add_clause(clause + [-assumptions[i]])  # Each clause gets a unique assumption
                
            # is_sat not needed right now as all problems should be unsat
            is_sat = solver.solve(assumptions=assumptions)
            
            unsat_core = solver.get_core()

            # Core contains clauses in reverse order, so we reverse it back
            reversed_core = torch.tensor(unsat_core[::-1])
            # Subtract offset from clauses in core to get original clauses
            unsat_core_clauses = reversed_core - offset
                
            # Map back to original clauses
            core_clause_literals = [clauses_current_problem[i] for i in range(len(clauses_current_problem)) if assumptions[i] in unsat_core]

            solver.delete()
            
            # Calculate mask for current sub_problem in eval_problem
            eval_batch_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=current_batch_num, batch_size=opts['min_n'], n_variables=eval_problem.n_variables)
            
            
            gt_mask = []
            for i, idx in enumerate(unsat_core_clauses):
                literals = core_clause_literals[i]
                
                # TODO: This only works if all sub_problems have the same amount of clauses!! WRONG!
                # clause = len(clauses_current_problem)*current_batch_num + unsat_core_clauses[i]
                clause = countClauses + unsat_core_clauses[i]
                
                # Sum of problemBatch.n_clauses_per_batch[] before current_batch_num? Or count while calculating gt for data?
                #clause = problemBatch.n_clauses_per_batch[current_batch_num] + unsat_core_clauses[i]
                
                for value in literals:
                    value = value -1 if value >= 1 else eval_problem.n_variables - (value + 1)
                    gt_mask.append([value, clause])
            
            countClauses = countClauses + len(clauses_current_problem)
                    
            sub_problem_edges = eval_problem.batch_edges[eval_batch_mask]
            
            motif_size = len(gt_mask)
            
            # TODO: VALIDATE THIS!!!
            gt = torch.isin(torch.tensor(sub_problem_edges), torch.tensor(gt_mask))
            # We only need the right column of the isin tensor
            gt = gt[:,1].int()
            
            reals.append(gt.flatten().numpy())
    
    reals = np.concatenate(reals)  # Flatten the list of arrays
    
        

    downstreamTask = NeuroSAT.NeuroSAT(opts=opts,device=device)
    checkpoint = torch.load(f"models/neurosat_sr10to40_ep1024_nr26_d128_last.pth.tar", weights_only=True, map_location=device)
    downstreamTask.load_state_dict(checkpoint['state_dict'])

    mlp = explainer_NeuroSAT.MLP(GraphTask=graph_task).to(device)
    wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

    mlp_optimizer = torch.optim.Adam(params = mlp.parameters(), lr = lr_mlp, maximize=False)

    downstreamTask.eval()
    for param in downstreamTask.parameters():
        param.requires_grad = False


    training_iterator = dataset
    
    for epoch in range(0, epochs) :
        mlp.train()
        mlp_optimizer.zero_grad()

        temperature = t0*((tT/t0) ** ((epoch+1)/epochs))
        
        #sampledEdges = 0.0
        #sumSampledEdges = 0.0
        
        #samplePredSum = 0

        for index, content in enumerate(training_iterator):
            if index > 0: break
            node_to_predict = None
            if graph_task: 
                # !! current_problem is really a batch of problems !!
                current_problem = content

            # MLP forward
            # TODO: Implement embeddingCalculation for SAT
            w_ij = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            pOriginal, _, _, _ = downstreamTask.forward(current_problem)
            pOriginal = fn.softmax(pOriginal.detach(), dim=0)
            
            for k in range(0, sampled_graphs):
                edge_ij = mlp.sampleGraph(w_ij, temperature)
                
                #sampledEdges += torch.sum(edge_ij)
            
                # TODO: softmax needed for loss, beacuse negative values do not work witg log! Need for normalization?
                pSample, _, _, _ = downstreamTask.forward(current_problem, edge_weights=edge_ij)
                pSample = fn.softmax(pSample.detach(), dim=0)

                #samplePredSum += torch.sum(torch.argmax(pSample, dim=1))
                
                if graph_task:
                    for sub_problem_idx in range(len(current_problem.is_sat)):
                        # batch_mask needed to differentiate sub_problems in batch of problems for loss
                        # batch_edges cannot simply be divided since sub_problems have different number of edges/clauses
                        # IMPORTANT: batch_size and n variables dependant on data!!
                        batch_mask = utils.get_batch_mask(torch.tensor(current_problem.batch_edges), sub_problem_idx, opts['min_n'], current_problem.n_variables)
                        currLoss = mlp.loss(pOriginal[sub_problem_idx], pSample[sub_problem_idx], edge_ij[batch_mask], coefficient_size_reg, coefficient_entropy_reg)
                        sampleLoss.add_(currLoss)
                    

            loss += sampleLoss / sampled_graphs
            
            #sumSampledEdges += sampledEdges / sampled_graphs

        #print(samplePredSum)
        
        loss = loss / len(training_iterator)
        loss.backward()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)
        
        """for param in mlp.parameters():
            if param.grad is not None:
                param.grad.data = torch.max(param.grad.data, min_clip_value * torch.ones_like(param.grad.data))"""

        mlp_optimizer.step()

        mlp.eval()
        
        """if graph_task:
            #TODO: Evaluation for SAT! Needs gt
            meanAuc = evaluation.evaluateNeuroSATAUC(mlp, downstreamTask, data)"""
            
        # Get one sub problem for evaluation:
        preds = []
        
        # Calculate weights and prediction for all sub_problems in eval_problem
        w_ij_eval = mlp.forward(downstreamTask, eval_problem, nodeToPred=node_to_predict)
        edge_ij_eval = mlp.sampleGraph(w_ij_eval, temperature).detach()
        #pSample_eval, _, _, _ = downstreamTask.forward(eval_problem, edge_weights=edge_ij_eval)
        #pOriginal_eval, _, _, _ = downstreamTask.forward(eval_problem)
        
        for j in range(len(eval_problem.is_sat)):
            # This can be repeated for each sub_problem
            current_batch_num = j
            
            clauses_current_problem = eval_problem.get_clauses_for_problem(current_batch_num)

            #print(f"Length of sub_problem 0: {len(clauses_current_problem)}")
            # Sub problem 1 Contains literals 11-20, ...
            #print(f"Sub_problem 0 clauses: {clauses_current_problem}")
            
            # Calculate mask for current sub_problem in eval_problem
            eval_batch_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=current_batch_num, batch_size=opts['min_n'], n_variables=eval_problem.n_variables)
            
            # Edge probabilites for current sub problem in eval_problems
            edge_ij_eval_masked = edge_ij_eval[eval_batch_mask]
                    
            sub_problem_edges = eval_problem.batch_edges[eval_batch_mask]
            
            motif_size = len(gt_mask)
            
            preds.append(edge_ij_eval_masked.cpu().flatten().numpy())
        
        
        preds = np.concatenate(preds)  # Flatten the list of arrays
    
        roc_auc = roc_auc_score(reals, preds)
        
        #print(f"Edge probabilites for first sub_problem in last problem: {edge_ij_eval_masked}")
        
        #print(f"Prediction for sub_problem: {pOriginal_eval[0]}")
        #print(f"Prediction for sampled sub_problem: {pSample_eval[0]}")
        
        print(f"roc_auc score: {roc_auc}")
    
        # print sub_problem 0 with calculated edge weights
        """if (epoch+1) % 5 == 0:
            pos = utils.visualize_edge_index_interactive(sub_problem_edges, edge_ij_eval_masked, f"results/{wandb.run.name}_vis_edge_ij_{epoch+1}", topK=motif_size)"""
        
        
        #sumSampledEdges = sumSampledEdges / len(training_iterator)
        #, "val/mean_AUC": meanAuc
        wandb.log({"train/Loss": loss, "val/temperature": temperature, "val/roc_auc": roc_auc})

        """for name, param in mlp.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad}")"""
        
    # print sub_problem 0 with calculated edge weights
    """pos = utils.visualize_edge_index_interactive(sub_problem_edges, edge_ij_eval_masked, f"results/{wandb.run.name}_vis_edge_ij_{epoch+1}", topK=motif_size)"""
    # print sub_problem 0 with gt
    """pos = utils.visualize_edge_index_interactive(sub_problem_edges, gt, f"results/{wandb.run.name}_vis_gt", pos)"""
        
    #if save_model == "True":
    #    torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{meanAuc}_{wandb.run.name}")

    wandb.finish()
    
    return mlp, downstreamTask