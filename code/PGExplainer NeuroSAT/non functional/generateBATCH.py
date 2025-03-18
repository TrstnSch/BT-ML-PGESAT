import numpy as np

def sign_variable(x):
    assert(abs(x) > 0)
    variable = abs(x) - 1
    sign = x < 0
    return variable, sign

def literal(x, n_variables):
    assert(x != 0)
    variable, sign = sign_variable(x)
    if sign:
        return variable + n_variables
    else:
        return variable

class Problem(object):
    def __init__(self, n_variables, clauses, is_sat, n_cell_per_batch):

        self.n_variables = n_variables
        self.n_literals = 2 * n_variables
        self.clauses = clauses


        self.n_cells = sum(n_cell_per_batch)
        self.n_cells_per_batch = n_cell_per_batch

        self.is_sat = is_sat
        self.edges(clauses)


    def edges(self, clauses):

        self.batch_edges = np.zeros([self.n_cells, 2], dtype=int)
        cell = 0
        for clause_id, clause in enumerate(clauses):
            literals = [literal(x, self.n_variables) for x in clause]
            for lit in literals:
                self.batch_edges[cell, :] = [lit, clause_id]
                cell += 1
                
        assert cell == self.n_cells

def shift_literal(x, offset):
        assert(x != 0)
        if x > 0:
            return x + offset
        else:
            return x - offset

def shift_clauses(clauses, offset):
    return [[shift_literal(x, offset) for x in clause] for clause in clauses]

def batch_of_problems(problems):
    clauses = []
    is_sat = []
    n_cells = []
    offset = 0

    prev_n_vars = None
    for n_vars, prob_clauses, prob_is_sat in problems:

        assert (prev_n_vars is None or n_vars == prev_n_vars)

        prev_n_vars = n_vars

        clauses.extend(shift_clauses(prob_clauses, offset)) 
        #clauses.extend(shift_clauses(clauses, offset))
        is_sat.append(prob_is_sat)
        n_cells.append(sum([len(clause) for clause in clauses]))

        offset += n_vars

    return Problem(offset, clauses, is_sat, n_cells)

