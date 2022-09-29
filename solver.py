import gurobipy as gp
import numpy as np


def steiner(g, terminals, p, theta=1, static=False):

    # initialize
    m = gp.Model('steiner')
    m.setParam('Method', 2)
    m.setParam('MIPFocus', 1)
    m.setParam('MIPGap', 1e-2)
    m.setParam('OutputFlag', 1)
    m.setParam('TimeLimit', 600)

    # pre-process
    max_k = len(terminals)
    dg = g.to_directed()

    # add variables
    x_vars_keys = [edge for edge in g.edges()]
    x = m.addVars(x_vars_keys, vtype=gp.GRB.BINARY, name='x')
    for edge in g.edges():
        x[edge[::-1]] = x[edge]
    z_vars_keys = [(k, edge) for k in range(max_k) for edge in g.edges()]
    z = m.addVars(z_vars_keys, vtype=gp.GRB.INTEGER, lb=0, ub=0 if static else 1, name='z')
    for k in range(max_k):
        for edge in g.edges():
            z[(k, edge[::-1])] = z[(k, edge)]

    f_vars_keys = [(k, u, edge) for k in range(max_k) for u in g.nodes() for edge in dg.edges()]
    f = m.addVars(f_vars_keys, vtype=gp.GRB.BINARY, name='f')
    m.update()

    # add constraints
    for k in range(max_k):
        k_terminals, = np.where(terminals[k] == 1)
        if k_terminals.size:
            r = k_terminals[0]
            for t in k_terminals[1:]:
                for v in g.nodes():
                    lhs = gp.quicksum(f[(k, t, edge)] for edge in dg.in_edges(v))
                    rhs = gp.quicksum(f[(k, t, edge)] for edge in dg.out_edges(v))
                    if v == r:
                        lhs += 1
                    elif v == t:
                        rhs += 1
                    m.addConstr(lhs == rhs)
                for edge in g.edges():
                    lhs = f[(k, t, edge)] + f[(k, t, edge[::-1])]
                    rhs = x[edge] + z[(k, edge)]
                    m.addConstr(lhs <= rhs)
    m.update()

    # set objective
    m.setObjective(gp.quicksum(p[k] * (x[edge] + theta * z[(k, edge)]) for edge in g.edges() for k in range(max_k)))
    m.update()

    # optimizing
    m.optimize()

    return m.ObjVal
