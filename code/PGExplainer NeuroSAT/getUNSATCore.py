from pysat.solvers import Minisat22

m = Minisat22()
m.add_clause([-1, 2])
m.add_clause([-2, 3])
m.add_clause([-3, 4])
m.solve(assumptions=[1, 2, 3, -4])

print(m.get_core())  # literals 2 and 3 are not in the core

m.delete()