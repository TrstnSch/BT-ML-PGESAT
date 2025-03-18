import random
import pickle
import numpy as np

from generateBATCH import batch_of_problems


from pysat.solvers import Solver


def generate_k_clause(n, k):
    variable_list = np.random.choice(n, size=min(n, k), replace=False)
    return [int(variable + 1) if random.random() < 0.5 else int(-(variable + 1)) for variable in variable_list]

def generate_clause_pair(opts, n_variables):
    solver = Solver(name='m22')
    clauses = []
    while True:

        base = 1 if random.random() < opts['p_k_2'] else 2
        clause_length = base + np.random.geometric(opts['p_geo'])

        clause = generate_k_clause(n_variables, clause_length)

        solver.add_clause(clause=clause)
        is_sat = solver.solve()

        if is_sat:
            clauses.append(clause)
        else:
            break

    clause_unsat = clause

    clause_sat = [- clause_unsat[0]] + clause_unsat[1:]

    return n_variables, clauses, clause_unsat, clause_sat

def generate_problem_pair(opts):

    file = open(opts['logging'], 'w')

    n_count = opts['max_n'] - opts['min_n'] + 1

    problems_per_n = opts['n_pairs'] * 1.0 / n_count

    problems = []
    batches = []
    nodes_in_batch = 0
    prev_n_vars = None

    for n_var in range(opts['min_n'], opts['max_n'] + 1):

        lower_bound = int((n_var - opts['min_n']) * problems_per_n) # 0
        upper_bound = int((n_var - opts['min_n'] + 1) * problems_per_n)


        for problems_id in range(lower_bound, upper_bound):
            n_variables, clauses, clause_unsat, clause_sat = generate_clause_pair(opts, n_var)


            clauses_with_unsat = clauses.copy()
            clauses_with_unsat.append(clause_unsat)

            clauses_with_sat = clauses.copy()
            clauses_with_sat.append(clause_sat)

            n_clauses_with_unsat = len(clauses_with_unsat)
            n_clauses_with_sat = len(clauses_with_sat)

            n_cells_with_unsat = sum(len(clauses_with_unsat) for clauses in clauses_with_unsat)
            n_cells_with_sat = sum(len(clauses_with_sat) for clauses in clauses_with_sat)

            n_nodes_sat = 2 * n_variables + n_clauses_with_sat
            n_nodes_unsat = 2 * n_variables + n_clauses_with_unsat

            n_nodes = n_nodes_sat + n_nodes_unsat

            if n_nodes > opts['max_nodes_per_batch']:
                continue

            batch_ready = False

            if (opts['one_pair'] and len(problems) > 1):
                batch_ready = True


            elif (prev_n_vars and n_variables != prev_n_vars):
                batch_ready = True


            elif (not opts['one_pair']) and nodes_in_batch + n_nodes > opts['max_nodes_per_batch']:
                batch_ready = True

            if batch_ready:
                batches.append(batch_of_problems(problems))
                print("batch %d done (%d variables, %d problems)..." % (len(batches), prev_n_vars, len(problems)), file=file)
                del problems[:]
                nodes_in_batch = 0

            prev_n_vars = n_variables

            is_sat, stats = solve_sat(n_variables, clauses_with_sat)
            is_unsat, stats = solve_sat(n_variables, clauses_with_unsat)

            problems.append((n_variables, clauses_with_sat, is_sat))
            problems.append((n_variables, clauses_with_unsat, is_unsat))

            nodes_in_batch += n_nodes

            print(len(problems))

    if len(problems) > 0:

        batches.append(batch_of_problems(problems))
        print(" %d batch, it contains (%d problems with %d variables each)..."
              % (len(batches), len(problems), prev_n_vars), file=file)
        del problems[:]

    return batches



def solve_sat(n_vars, clauses):

    solver = Solver(name='m22')

    for clause in clauses:
        solver.add_clause(clause)

    is_sat = solver.solve()

    stats = solver.accum_stats()

    solver.delete()

    return is_sat, stats


if __name__ == '__main__':
    opts = {
        'out_dir': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/data/dataset_val2_SR(40)',
        'logging': '/Users/trist/Documents/Bachelor-Thesis/NeuroSAT/test/files/log/dataset_val_SR(40).log',
        'n_pairs': 100,  # Anzahl der zu generierenden Paare
        'min_n': 40,
        'max_n': 40,
        'p_k_2': 0.3,
        'p_geo': 0.4,
        'max_nodes_per_batch': 12000,
        'one_pair': False,
    }


    batches = generate_problem_pair(opts)

    dataset_filename = opts['out_dir']

    print("Writing %d batches to %s..." % (len(batches), dataset_filename))
    with open(dataset_filename, 'wb') as f_dump:
        pickle.dump(batches, f_dump)



