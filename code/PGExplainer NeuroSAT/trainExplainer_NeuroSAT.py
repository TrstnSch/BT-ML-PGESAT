import datasetLoader
import evaluation
import explainer_NeuroSAT
import NeuroSAT
import utils
import sys
import pickle
import torch
import torch.nn.functional as fn
from torch_geometric.loader import DataLoader
from torch_geometric import seed
from torch_geometric.utils import k_hop_subgraph
from torcheval.metrics.functional import binary_auroc
import wandb
from collections import defaultdict



datasetType = ['NeuroSAT']


"""def loadExplainer(dataset):
    # Check valid dataset name
    config = utils.loadConfig(dataset)
    if config == -1:
        return
    
    params = config['params']
    graph_task = params['graph_task']
    
    data, labels = datasetLoader.loadGraphDataset(dataset) if graph_task else datasetLoader.loadOriginalNodeDataset(dataset)
    
    mlp = explainer_NeuroSAT.MLP(GraphTask=graph_task, hidden_dim=64)     # Adjust according to data and task
    mlp.load_state_dict(torch.load(f"models/explainer{dataset}", weights_only=True))

    downstreamTask = networks.GraphGNN(features = data[0].x.shape[1], labels=labels) if graph_task else networks.NodeGNN(features = data.x.shape[1], labels=labels)
    downstreamTask.load_state_dict(torch.load(f"models/{dataset}", weights_only=True))
    
    return mlp, downstreamTask"""




def trainExplainer (dataset, save_model=False, wandb_project="Experiment",runSeed=None, opts=None) :
    if runSeed is not None: seed.seed_everything(runSeed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check valid dataset name
    if dataset == "NeuroSAT-hard":
        hard_constraint = True
    else:
        hard_constraint = False
    
    
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
    coefficient_consistency = params['coefficient_consistency']
    num_explanation_edges = params['num_explanation_edges']
    bce = bool(params['bce_loss'])
    lr_mlp = params['lr_mlp']
    complex_architecture = bool(params['complex_architecture'])
    three_embs = bool(params['three_embs'])
    adamW = bool(params['adamW'])

    wandb.init(project=wandb_project, config=params)

    hidden_dim = 64 # Make loading possible
    clip_grad_norm = 2 # Make loading possible
    min_clip_value = -2
    
    with open(opts['out_dir'], 'rb') as file:
        data = pickle.load(file)
        
    dataset = data
    
    val_problem = data[-2]
    test_problem = data[-1]
    
    with open(opts['val_reals_dir'], 'rb') as file:
        val_reals = pickle.load(file)
        
    with open(opts['val_gt_edges_per_problem'], 'rb') as file:
        val_gt_edges_per_problem = pickle.load(file)
        
    with open(opts['test_reals_dir'], 'rb') as file:
        test_reals = pickle.load(file)
        
    with open(opts['test_gt_edges_per_problem'], 'rb') as file:
        test_gt_edges_per_problem = pickle.load(file)

    downstreamTask = NeuroSAT.NeuroSAT(opts=opts,device=device)
    checkpoint = torch.load(f"models/neurosat_sr10to40_ep1024_nr26_d128_last.pth.tar", weights_only=True, map_location=device)
    downstreamTask.load_state_dict(checkpoint['state_dict'])
    
    if hard_constraint:
        mlp = explainer_NeuroSAT.MLP_SAT(GraphTask=graph_task, complex_architecture=complex_architecture, three_embs=three_embs).to(device)
    else:
        mlp = explainer_NeuroSAT.MLP(GraphTask=graph_task, three_embs=three_embs).to(device)
    #wandb.watch(mlp, log= "all", log_freq=2, log_graph=False)

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
            if hard_constraint:
                w_ij, unique_clauses, inverse_indices = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)
            else:
                w_ij = mlp.forward(downstreamTask, current_problem, nodeToPred=node_to_predict)

            sampleLoss = torch.FloatTensor([0]).to(device)
            loss = torch.FloatTensor([0]).to(device)
            
            pOriginal, _, _, _ = downstreamTask.forward(current_problem)
            pOriginal = fn.softmax(pOriginal, dim=0)
            
            #pOriginal = torch.tensor([1 - pOriginal, pOriginal])
            
            for k in range(0, sampled_graphs):
                if hard_constraint:
                    edge_ij = mlp.sampleGraph(w_ij, unique_clauses, inverse_indices, temperature)
                else:
                    edge_ij = mlp.sampleGraph(w_ij, temperature)
            
                # TODO: softmax needed for loss, beacuse negative values do not work witg log! Need for normalization?
                pSample, _, _, _ = downstreamTask.forward(current_problem, edge_weights=edge_ij)
                pSample = fn.softmax(pSample, dim=0)
                #pSample = torch.tensor([1 - pSample, pSample])
                
                if graph_task:
                    currLoss = mlp.loss(pOriginal, pSample, edge_ij, current_problem.batch_edges, coefficient_size_reg, coefficient_entropy_reg, coefficientConsistency=coefficient_consistency)
                    sampleLoss.add_(currLoss)
                    

            loss += sampleLoss / sampled_graphs
        
        loss = loss / ((len(training_iterator)-2)*len(data[0].is_sat))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)
        
        mlp_optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        mlp.eval()
        
        # TODO: CHANGE FOR HARD CONSTRAINT!
        if hard_constraint:
            w_ij_eval, unique_clauses_eval, inverse_indices_eval = mlp.forward(downstreamTask, val_problem, nodeToPred=node_to_predict)
            edge_ij_eval = mlp.sampleGraph(w_ij_eval, unique_clauses_eval, inverse_indices_eval, temperature).detach()
        else:
            w_ij_eval = mlp.forward(downstreamTask, val_problem, nodeToPred=node_to_predict)
            edge_ij_eval = mlp.sampleGraph(w_ij_eval).detach()
        
        pOriginal, _, _, _ = downstreamTask.forward(val_problem)
        pOriginal = fn.softmax(pOriginal, dim=0)
        
        pSample, _, _, _ = downstreamTask.forward(val_problem, edge_weights=edge_ij_eval)
        pSample = fn.softmax(pSample, dim=0)
        
        auroc_list = []
        
        valLoss = torch.FloatTensor([0]).to(device)
        
        for current_batch_num in range(len(val_problem.is_sat)):
            # This can be repeated for each sub_problem
            #clauses_current_problem = eval_problem.get_clauses_for_problem(current_batch_num)

            #print(f"Length of sub_problem 0: {len(clauses_current_problem)}")
            # Sub problem 1 Contains literals 11-20, ...
            #print(f"Sub_problem 0 clauses: {clauses_current_problem}")
            
            # Calculate mask for current sub_problem in eval_problem
            eval_batch_mask = utils.get_batch_mask(torch.tensor(val_problem.batch_edges), batch_idx=current_batch_num, batch_size=opts['min_n'], n_variables=val_problem.n_variables)
            
            # Edge probabilites for current sub problem in eval_problems
            edge_ij_eval_masked = edge_ij_eval[eval_batch_mask]
            
            sub_problem_edges = val_problem.batch_edges[eval_batch_mask]
            
            curr_auroc = binary_auroc(edge_ij_eval_masked, val_reals[current_batch_num])
            auroc_list.append(curr_auroc.item())
            
        valLoss = mlp.loss(pOriginal, pSample, edge_ij_eval, val_problem.batch_edges, coefficient_size_reg, coefficient_entropy_reg, coefficientConsistency=coefficient_consistency)
            
        valLoss = valLoss / len(val_problem.is_sat)
            
        auroc_tensor = torch.tensor(auroc_list)
        mean_auroc = auroc_tensor.mean()
        if epoch == epochs-1:
            highest_auroc_index = torch.argmax(auroc_tensor)
            highest_mask = utils.get_batch_mask(torch.tensor(val_problem.batch_edges), batch_idx=highest_auroc_index, batch_size=opts['min_n'], n_variables=val_problem.n_variables)
            highest_auroc = auroc_tensor[highest_auroc_index]
            wandb.log({"val/highest_individual_auroc": highest_auroc})
            highest_edge_ij = edge_ij_eval[highest_mask]
            highest_gt = val_reals[highest_auroc_index]
            sub_problem_edges_highest = val_problem.batch_edges[highest_mask]
            
            lowest_auroc_index = torch.argmin(auroc_tensor)
            lowest_mask = utils.get_batch_mask(torch.tensor(val_problem.batch_edges), batch_idx=lowest_auroc_index, batch_size=opts['min_n'], n_variables=val_problem.n_variables)
            lowest_auroc = auroc_tensor[lowest_auroc_index]
            lowest_edge_ij = edge_ij_eval[lowest_mask]
            lowest_gt = val_reals[lowest_auroc_index]
            sub_problem_edges_lowest = val_problem.batch_edges[lowest_mask]
        
        
        print(f"mean auroc score: {mean_auroc}")
        
        if (epoch+1) % 10 == 0:
            pos = utils.visualize_edge_index_interactive(sub_problem_edges, edge_ij_eval_masked, f"results/experiments/seed{runSeed}_vis_edge_ij_{epoch+1}", topK=len(val_gt_edges_per_problem[-1]))
        
        
        wandb.log({"train/Loss": loss, "val/Loss": valLoss, "val/auroc": mean_auroc, "edge_importance/min": edge_ij_eval.min().item(), "edge_importance/max": edge_ij_eval.max().item(),
                "edge_importance/mean": edge_ij_eval.mean().item()})
        
        
    # print sub_problem 0 with calculated edge weights
    pos = utils.visualize_edge_index_interactive(sub_problem_edges, edge_ij_eval_masked, f"results/experiments/seed{runSeed}_vis_edge_ij_{epoch+1}", topK=len(val_gt_edges_per_problem[-1]))
    # print sub_problem 0 with gt
    pos = utils.visualize_edge_index_interactive(sub_problem_edges, val_reals[-1], f"results/experiments/seed{runSeed}_vis_gt", pos)
    
    # TODO: Show difference between topK edges and gt edges -> logical and on topK and gt, visualize
    
    sorted_weights, topk_indices_highest = torch.topk(highest_edge_ij, len(val_gt_edges_per_problem[highest_auroc_index]))
    mask_topK_highest = torch.zeros_like(highest_edge_ij, dtype=torch.float32)
    mask_topK_highest[topk_indices_highest] = 1    
    
    common_edges_highest = torch.logical_and(highest_gt, mask_topK_highest.flatten())
    print(f"Highest individual auroc: {highest_auroc}")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest, highest_edge_ij, f"results/experiments/seed{runSeed}_highestAUC", topK=len(val_gt_edges_per_problem[highest_auroc_index]))
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest, highest_gt, f"results/experiments/seed{runSeed}_highestAUC_gt")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest, common_edges_highest, f"results/experiments/seed{runSeed}_highestAUC_commonEdges")
    
    # TO EVALUATE SATISFIABILITY OF EXPLANATION:
    print(f"Highest AUC example masked edges: {sub_problem_edges_highest[mask_topK_highest.bool()]}")
    
    # Group literals by clause number
    clause_dict = defaultdict(list)

    for literal, clause in sub_problem_edges_highest[mask_topK_highest.bool()]:
        literal = int(literal)
        clause = int(clause)
        if literal >= val_problem.n_variables:
            literal = -(literal - val_problem.n_variables)
        clause_dict[clause].append(literal)

    # Convert to list of clauses (optional: sort by clause number)
    clauses = list(clause_dict.values())
    
    print(clauses)
    
    sorted_weights, topk_indices_lowest = torch.topk(lowest_edge_ij, len(val_gt_edges_per_problem[lowest_auroc_index]))
    mask_topK_lowest = torch.zeros_like(lowest_edge_ij, dtype=torch.float32)
    mask_topK_lowest[topk_indices_lowest] = 1
    
    common_edges_lowest = torch.logical_and(lowest_gt, mask_topK_lowest.flatten())
    print(f"Lowest individual auroc: {lowest_auroc}")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_lowest, lowest_edge_ij, f"results/experiments/seed{runSeed}_lowestAUC", topK=len(val_gt_edges_per_problem[lowest_auroc_index]))
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_lowest, lowest_gt, f"results/experiments/seed{runSeed}_lowestAUC_gt")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_lowest, common_edges_lowest, f"results/experiments/seed{runSeed}_lowestAUC_commonEdges")
    
    
    # TESTING
    mlp.eval()
    
    if hard_constraint:
        w_ij_test, unique_clauses_test, inverse_indices_test = mlp.forward(downstreamTask, test_problem, nodeToPred=node_to_predict)
        edge_ij_test = mlp.sampleGraph(w_ij_test, unique_clauses_test, inverse_indices_test, temperature).detach()
    else:
        w_ij_test = mlp.forward(downstreamTask, test_problem, nodeToPred=node_to_predict)
        edge_ij_test = mlp.sampleGraph(w_ij_test).detach()
    
    pOriginal_test, _, _, _ = downstreamTask.forward(test_problem)
    pOriginal_test = fn.softmax(pOriginal_test, dim=0)
    
    pSample_test, _, _, _ = downstreamTask.forward(test_problem, edge_weights=edge_ij_test)
    pSample_test = fn.softmax(pSample_test, dim=0)
    
    test_auroc_list = []
    
    for current_batch_num in range(len(test_problem.is_sat)):
        
        test_batch_mask = utils.get_batch_mask(torch.tensor(test_problem.batch_edges), batch_idx=current_batch_num, batch_size=opts['min_n'], n_variables=test_problem.n_variables)
        
        edge_ij_test_masked = edge_ij_test[test_batch_mask]
        
        sub_problem_edges = test_problem.batch_edges[test_batch_mask]
        
        curr_auroc = binary_auroc(edge_ij_test_masked, test_reals[current_batch_num])
        test_auroc_list.append(curr_auroc.item())
        
    test_auroc_tensor = torch.tensor(test_auroc_list)
    test_mean_auroc = test_auroc_tensor.mean()
    if epoch == epochs-1:
        highest_auroc_index_test = torch.argmax(test_auroc_tensor)
        highest_mask_test = utils.get_batch_mask(torch.tensor(test_problem.batch_edges), batch_idx=highest_auroc_index_test, batch_size=opts['min_n'], n_variables=test_problem.n_variables)
        highest_auroc_test = test_auroc_tensor[highest_auroc_index_test]
        wandb.log({"test/highest_individual_auroc": highest_auroc_test})
        highest_edge_ij_test = edge_ij_test[highest_mask_test]
        highest_gt_test = test_reals[highest_auroc_index_test]
        sub_problem_edges_highest_test = test_problem.batch_edges[highest_mask_test]
        
        lowest_auroc_index_test = torch.argmin(test_auroc_tensor)
        lowest_mask_test = utils.get_batch_mask(torch.tensor(test_problem.batch_edges), batch_idx=lowest_auroc_index_test, batch_size=opts['min_n'], n_variables=test_problem.n_variables)
        lowest_auroc_test = test_auroc_tensor[lowest_auroc_index_test]
        lowest_edge_ij_test = edge_ij_test[lowest_mask_test]
        lowest_gt_test = test_reals[lowest_auroc_index_test]
        sub_problem_edges_lowest_test = test_problem.batch_edges[lowest_mask_test]
    
    
    print(f"mean TEST auroc score: {test_mean_auroc}")
    
    wandb.log({"test/auroc": test_mean_auroc})
    
    sorted_weights, topk_indices_highest_test = torch.topk(highest_edge_ij_test, len(test_gt_edges_per_problem[highest_auroc_index_test]))
    mask_topK_highest_test = torch.zeros_like(highest_edge_ij_test, dtype=torch.float32)
    mask_topK_highest_test[topk_indices_highest_test] = 1    
    
    common_edges_highest_test = torch.logical_and(highest_gt_test, mask_topK_highest_test.flatten())
    print(f"Highest individual TEST auroc: {highest_auroc_test}")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest_test, highest_edge_ij_test, f"results/experiments/seed{runSeed}_highestAUC_TEST", topK=len(test_gt_edges_per_problem[highest_auroc_index_test]))
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest_test, highest_gt_test, f"results/experiments/seed{runSeed}_highestAUC_gt_TEST")
    pos = utils.visualize_edge_index_interactive(sub_problem_edges_highest_test, common_edges_highest_test, f"results/experiments/seed{runSeed}_highestAUC_commonEdges_TEST")
    
    
    
    
    
    if save_model == "True":
        torch.save(mlp.state_dict(), f"models/explainer_{dataset}_{mean_auroc}_{wandb.run.name}")

    wandb.finish()
    
    return mlp, downstreamTask, test_auroc_list
