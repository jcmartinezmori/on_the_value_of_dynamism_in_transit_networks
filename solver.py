import gurobipy as gp
import networkx as nx
import numpy as np


def steiner_forest(g, pairs, g_dists, rho=None, time_limit=None):

    # initialize
    m = gp.Model('steiner_forest')
    m.setParam('Method', 2)
    m.setParam('MIPFocus', 1)
    m.setParam('MIPGap', 5e-2)
    m.setParam('OutputFlag', 1)
    if time_limit is None:
        m.setParam('TimeLimit', 600)
    else:
        m.setParam('TimeLimit', time_limit)

    # pre-process
    m._g = g
    m._dg = m._g.to_directed()
    m._dg_rev = m._dg.reverse()
    m._edges, m._costs = zip(*[((u, v), data['length']) for u, v, data in m._g.edges(data=True)])
    m._arcs = [(u, v) for u, v in m._dg.edges()]

    # add variables
    m._xvars = m.addVars(m._edges, vtype=gp.GRB.BINARY, name='x')
    for u, v in m._xvars:
        m._xvars[(v, u)] = m._xvars[(u, v)]

    # set objective
    obj = gp.quicksum(cost * m._xvars[edge] for cost, edge in zip(m._costs, m._edges))
    m.setObjective(obj)

    # add variables and constraints
    m._fvars = {}
    for s, t in pairs:
        if rho is not None:
            s_edges = set(
                edge for edge in nx.ego_graph(m._dg, s, radius=g_dists.loc[s, t] + 1, distance='length').edges())
            t_edges = set(
                edge for edge in nx.ego_graph(m._dg_rev, t, radius=g_dists.loc[s, t] + 1, distance='length').edges())
            rho_edges = s_edges & t_edges
            st_edges = set()
            for edge in rho_edges:
                u, v = edge
                min_dist = min(g_dists.loc[s, u] + g_dists.loc[v, t], + g_dists.loc[s, v] + g_dists.loc[u, t])
                if min_dist + m._dg.edges[edge]['length'] <= rho * g_dists.loc[s, t] + 1:
                    st_edges.add(edge)
            st_dg = nx.edge_subgraph(m._dg, st_edges)
        else:
            st_dg = m._dg
        m._fvars[(s, t)] = m.addVars(m._arcs, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='f_{0}-{1}'.format(s, t))
        for u in st_dg.nodes():
            lhs = gp.quicksum(m._fvars[(s, t)][arc] for arc in st_dg.out_edges(u)) \
                  - gp.quicksum(m._fvars[(s, t)][arc] for arc in st_dg.in_edges(u))
            if u == s:
                rhs = 1
            elif u == t:
                rhs = -1
            else:
                rhs = 0
            m.addConstr(lhs == rhs)
        for u, v in st_dg.edges():
            lhs = m._fvars[(s, t)][(u, v)]
            rhs = m._xvars[(u, v)]
            m.addConstr(lhs <= rhs)
        if rho is not None:
            lhs = gp.quicksum(data['length'] * m._fvars[(s, t)][(u, v)] for u, v, data in st_dg.edges(data=True))
            rhs = rho * g_dists.loc[s, t]
            m.addConstr(lhs <= rhs)

    # optimize
    m.update()
    m.optimize()

    # post-process
    edges = []
    for u, v in m._g.edges():
        if m._xvars[(u, v)].x > 0:
            edges.append((u, v))

    return m.ObjVal, edges


def steiner(g, terminals, p, thetas=1, static=False):

    # initialize
    m = gp.Model('steiner')
    m.setParam('Method', 2)
    m.setParam('MIPFocus', 3)
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

    # set objective and optimize
    if np.isscalar(thetas):
        theta = thetas
        m.setObjective(
            gp.quicksum(p[k] * (x[edge] + theta * z[(k, edge)]) for edge in g.edges() for k in range(max_k))
        )
        m.update()
        m.optimize()
        obj = m.ObjVal
        return obj

    else:
        objs = []
        for theta in thetas:
            m.setObjective(
                gp.quicksum(p[k] * (x[edge] + theta * z[(k, edge)]) for edge in g.edges() for k in range(max_k))
            )
            m.update()
            m.optimize()
            obj = m.ObjVal
            objs.append(obj)
        return objs
