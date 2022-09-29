import networkx as nx
import numpy as np
from solver import steiner


def scatter_df_helper(p_rule, g, terminals):

    lap_spec = nx.laplacian_spectrum(g)
    g._avg_deg = 2 * g.number_of_edges() / g.number_of_nodes()
    g._v_conn = nx.average_node_connectivity(g)
    g._alg_conn = lap_spec[1]
    p = p_generator(p_rule, g, terminals)
    g._obj_static = steiner(g, terminals, p, static=True)
    obj = steiner(g, terminals, p, theta=1)
    g._dg1 = g._obj_static / obj

    return g


def curve_df_helper(p_rule, g, terminals, theta):

    p = p_generator(p_rule, g, terminals)
    obj = steiner(g, terminals, p, theta=theta)
    return g, theta, obj


def p_generator(p_rule, g, terminals):

    if p_rule == 'unif':
        bias = [1/2 for _ in g.nodes()]
    elif p_rule == 'closeness_cent':
        closeness_cent = nx.closeness_centrality(g)
        factor = (g.number_of_nodes() - 1) / g.number_of_nodes()
        bias = [closeness_cent[node] * factor for node in g.nodes()]
    elif p_rule == 'inv_closeness_cent':
        closeness_cent = nx.closeness_centrality(g)
        factor = (g.number_of_nodes() - 1) / g.number_of_nodes()
        bias = [1 - closeness_cent[node] * factor for node in g.nodes()]
    else:
        raise Exception('Probability rule {0} not supported!'.format(p_rule))

    p = []
    for k in range(len(terminals)):
        p.append(
            np.product([bias[node] if terminals[k][node] else 1 - bias[node] for node in g.nodes()])
        )

    return p
