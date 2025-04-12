import torch
import pickle
from pysat.formula import CNF
from pysat.examples.musx import MUSX
import utils
    
def loadSatBatches (directory):
    with open(directory, 'rb') as file:
        data = pickle.load(file)
        
    return data


# Saves reals and gt_edges as seperate files
def storeGTforData (eval_problem, batch_size, output_directory):
    sub_problem_start_clause = 0
    reals = []
    gt_edges_per_problem = []
    for current_batch_num in range(len(eval_problem.is_sat)):
        clauses_problem_i = eval_problem.get_clauses_for_problem(current_batch_num)
        cnf = CNF()
        for i, clause in enumerate(clauses_problem_i):
            cnf.append(clause)
        musx = MUSX(cnf, verbosity=0)
        # mus contains list of relative clause numbers for the current problem e.g. [4, 7, ..., 53], starting at 1
        mus = musx.compute()
        eval_batch_mask = utils.get_batch_mask(torch.tensor(eval_problem.batch_edges), batch_idx=current_batch_num, batch_size=batch_size, n_variables=eval_problem.n_variables)
        masked_batch_edges = torch.tensor(eval_problem.batch_edges[eval_batch_mask])
        
        # ADD offset to mus
        mus_tensor = torch.tensor(mus) + sub_problem_start_clause
        
        gt_mask = torch.isin(masked_batch_edges[:, 1],mus_tensor)
        # Update offset
        sub_problem_start_clause += masked_batch_edges[:, 1].unique().numel()
        
        #reals.append(gt_mask.int().flatten().numpy())
        reals.append(gt_mask.int())
        gt_edges_per_problem.append(masked_batch_edges[gt_mask])
        
    dataset_filename = output_directory + "_reals"
    with open(dataset_filename, 'wb') as f_dump:
            pickle.dump(reals, f_dump)
        
    dataset_filename = output_directory + "_gt_edges_per_problem"
    with open(dataset_filename, 'wb') as f_dump:
            pickle.dump(gt_edges_per_problem, f_dump)