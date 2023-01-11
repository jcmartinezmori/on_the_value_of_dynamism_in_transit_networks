import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pickle
import pandas as pd
import plotly.express as px
import swifter  # parallelize pandas apply() function
from datetime import datetime
from solver import steiner_forest


def nearest_s_node(graph, row):
    node, e_dist = ox.nearest_nodes(graph, row.s_lon, row.s_lat, return_dist=True)
    if e_dist < 250:
        return node
    else:
        return -1


def nearest_t_node(graph, row):
    node, e_dist = ox.nearest_nodes(graph, row.t_lon, row.t_lat, return_dist=True)
    if e_dist < 250:
        return node
    else:
        return -1


def pre_process():

    g = ox.graph_from_place('Manhattan', network_type='drive', simplify=True)
    g = nx.DiGraph(g)
    g = nx.edge_subgraph(
        g, [(u, v) for u, v, data in g.edges(data=True) if
            data['highway'] not in ['motorway', 'motorway_link', 'trunk', 'trunk_link']]
    ).copy()
    lat_threshold = 40.742202311929816
    north_nodes = set(u for u, data in g.nodes(data=True) if data['y'] >= lat_threshold)
    g.remove_nodes_from(north_nodes)
    g = g.to_undirected()
    proceed = True
    while proceed:
        proceed = False
        to_contract = []
        no_to_contract = 0
        for edge in g.edges():
            if g.edges[edge]['length'] < 30:
                to_contract.append(edge)
                no_to_contract += 1
        contracted = set()
        no_contracted = 0
        for edge in to_contract:
            u, v = edge
            if u not in contracted and v not in contracted:
                contracted = contracted.union(edge)
                g = nx.contracted_edge(g, edge, self_loops=False)
                no_contracted += 1
        if no_contracted < no_to_contract:
            proceed = True
    g.remove_edges_from(set(nx.selfloop_edges(g)))
    for _, _, data in g.edges(data=True):
        if data['highway'] in ['residential', 'unclassified']:
            data['length'] *= 1.5
    proceed = True
    while proceed:
        proceed = False
        leafs = set(u for u in g.nodes() if g.degree[u] == 1)
        if leafs:
            proceed = True
            g.remove_nodes_from(leafs)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    with open('./realistic/g.pkl', 'wb') as file:
        pickle.dump(g, file)

    g_dists = dict(nx.all_pairs_dijkstra_path_length(g, weight='length'))
    g_dists = pd.DataFrame.from_dict(g_dists)
    g_dists.to_csv('./realistic/g_dists.csv')

    dt_start, dt_end = '2016-06-01 00:00:00', '2016-06-30 23:59:59'
    t_start, t_end = '07:00:00', '08:00:00'
    dt_start, dt_end = datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')
    t_start, t_end = datetime.strptime(t_start, '%H:%M:%S').time(), datetime.strptime(t_end, '%H:%M:%S').time()

    df = pd.read_csv(
        'yellow_tripdata_2016-06.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'],
        usecols=[
            'tpep_pickup_datetime', 'pickup_latitude', 'pickup_longitude',
            'tpep_dropoff_datetime', 'dropoff_latitude', 'dropoff_longitude'
        ]
    )
    df = df.rename(columns={
        'tpep_pickup_datetime': 's_dt', 'pickup_latitude': 's_lat', 'pickup_longitude': 's_lon',
        'tpep_dropoff_datetime': 't_dt', 'dropoff_latitude': 't_lat', 'dropoff_longitude': 't_lon'}
    )

    print('     filtering weekends')
    df = df[(df['s_dt'].dt.weekday != 5) & (df['s_dt'].dt.weekday != 6)].copy()

    print('     filtering dates and time')
    df = df[(df['s_dt'] >= dt_start) & (df['s_dt'] < dt_end)].copy()
    df = df[df['s_dt'].swifter.apply(lambda row: row.time()) >= t_start].copy()
    df = df[df['s_dt'].swifter.apply(lambda row: row.time()) < t_end].copy()

    print('     filtering e_dist')  # euclidean distance
    df['e_dist'] = df.swifter.apply(
        lambda row: ox.distance.great_circle_vec(row.s_lat, row.s_lon, row.t_lat, row.t_lon), axis=1
    )
    df = df[df.e_dist >= 1000].copy()

    print('     filtering by s_lat and t_lat')
    df = df[(df['s_lat'] <= lat_threshold + 0.01) & (df['t_lat'] <= lat_threshold + 0.01)].copy()

    print('     filtering s_node')
    df['s_node'] = df.swifter.apply(lambda row: int(nearest_s_node(g, row)), axis=1)
    df = df[df.s_node > 0].copy()

    print('     filtering t_node')
    df['t_node'] = df.swifter.apply(lambda row: int(nearest_t_node(g, row)), axis=1)
    df = df[df.t_node > 0].copy()

    print('     filtering g_dist')  # shortest path distance
    df['g_dist'] = df.swifter.apply(lambda row: g_dists[row.s_node][row.t_node], axis=1)
    df = df[df['g_dist'] >= 1000].copy()

    df.to_csv('./realistic/df.csv', index=False)


def main():

    dt_start, dt_end = '2016-06-01 00:00:00', '2016-06-30 23:59:59'
    dt_start, dt_end = datetime.strptime(dt_start, '%Y-%m-%d %H:%M:%S'), datetime.strptime(dt_end, '%Y-%m-%d %H:%M:%S')

    delta_t_l = [1 * 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60, 10 * 60, 12 * 60, 15 * 60, 20 * 60]  # seconds
    rho_l = [5/4, 4/3, 3/2]

    _pre_process = False
    if _pre_process:
        pre_process()

    print('     loading data')
    with open('./g.pkl', 'rb') as file:
        g = pickle.load(file)
    g_dists = pd.read_csv('./g_dists.csv', index_col=0).rename(columns=lambda x: int(x))
    df = pd.read_csv('./df.csv', parse_dates=['s_dt', 't_dt'])

    mst_cost = sum(data['length'] for _, _, data in nx.minimum_spanning_tree(g, weight='length').edges(data=True))

    results_df_jobs = [(delta_t, rho) for delta_t in delta_t_l for rho in rho_l]
    results = []
    for job in results_df_jobs:

        job_delta_t, job_rho = job
        job_g = g.copy()
        job_df = df.copy()  # will not be stored in results_df
        job_ct = 0
        job_mst_cost = mst_cost
        job_opt = None
        job_a = None

        results.append([job_delta_t, job_rho, job_g, job_ct, job_mst_cost, job_opt, job_a])

        for _, _, data in job_g.edges(data=True):
            data['edge_ct'] = 0
            data['p'] = 0

        ts_bins = np.arange(dt_start.timestamp(), dt_end.timestamp(), step=job_delta_t)
        no_bins = len(ts_bins) - 1

        job_df['bin_no'] = pd.cut(
            job_df['s_dt'].swifter.apply(lambda row: row.timestamp()), bins=ts_bins, right=False, labels=range(no_bins)
        )

        for bin_no in range(no_bins):
            bin_df = job_df[job_df['bin_no'] == bin_no].drop_duplicates(subset=['s_node', 't_node'])
            pairs = list(zip(bin_df['s_node'], bin_df['t_node']))
            if len(pairs) > 0:
                job_ct += 1
                _, edges = steiner_forest(job_g, pairs, g_dists, rho=job_rho, time_limit=max(600, job_delta_t))
                for edge in edges:
                    job_g.edges[edge]['edge_ct'] += 1
                for _, _, data in job_g.edges(data=True):
                    data['p'] = data['edge_ct'] / job_ct
                job_opt = sum(data['p'] * data['length'] for _, _, data in job_g.edges(data=True))
                job_a = job_mst_cost / job_opt

                results[-1] = [job_delta_t, job_rho, job_g, job_ct, job_mst_cost, job_opt, job_a]

                results_df = pd.DataFrame(results, columns=['delta_t', 'rho', 'g', 'ct', 'mst_cost', 'opt', 'a'])

                pd.to_pickle(results_df, './realistic/results_df.pkl')


if __name__ == '__main__':
    # main()
    pass

#
#
results_df_new = pd.read_pickle('./results_nyc/results_df_new_new.pkl')
results_df = pd.read_pickle('./results_nyc/results_df_new.pkl')
results_df = results_df[results_df['delta_t'] < 600]
results_df = pd.concat([results_df, results_df_new])
results_df.sort_values('delta_t', inplace=True)
results_df['delta_t'] /= 60
results_df['numerator'] = results_df['a'] * results_df['delta_t']

w = 1800 * 0.8
h = 600

from plotly.subplots import make_subplots
import plotly.graph_objects as go

delta_list = np.linspace(0.15, 20, 40)

fig = make_subplots(
    rows=1, cols=3, shared_xaxes=True, horizontal_spacing=w/40000, vertical_spacing=h/10000,
    column_titles=[r'$\huge{\rho} = 5/4$', r'$\huge{\rho} = 4/3$', r'$\huge{\rho} = 3/2$']
)

results_df['data'] = results_df.apply(
    lambda x: [(delta_s, x.numerator / delta_s) for delta_s in np.arange(int(x.delta_t), 30, 0.1) if x.numerator / delta_s >= 1],
    axis=1
)
#
results_df_125 = results_df[results_df['rho'] == 1.25]
results_df_133 = results_df[results_df['rho'] == 4/3]
results_df_150 = results_df[results_df['rho'] == 1.5]

deltas = {1, 3, 6, 10, 12, 15}
colors = ['#000000', '#a9a9a9', '#4363d8', '#000000', '#a9a9a9', '#4363d8']
dashes = ['solid', 'solid', 'solid', 'dot', 'dot', 'dot']

idx = 0
for i, row in results_df_125.iterrows():
    if row.delta_t not in deltas:
        continue
    delta_s, estimate = list((zip(*row.data)))
    fig.add_trace(go.Scatter(
        x=delta_s,
        y=estimate,
        name=r'$\delta_d = {0}$'.format(row.delta_t),
        marker={'color': colors[idx]},
        mode='lines',
        line=dict(dash=dashes[idx]),
        showlegend=True
    ), row=1, col=1)
    idx += 1

idx = 0
for i, row in results_df_133.iterrows():
    if row.delta_t not in deltas:
        continue
    delta_s, estimate = list((zip(*row.data)))
    fig.add_trace(go.Scatter(
        x=delta_s,
        y=estimate,
        name=r'$\delta_d = {0}$'.format(row.delta_t),
        marker={'color': colors[idx]},
        mode='lines',
        line=dict(dash=dashes[idx]),
        showlegend=False
    ), row=1, col=2)
    idx += 1

idx = 0
for i, row in results_df_150.iterrows():
    if row.delta_t not in deltas:
        continue
    delta_s, estimate = list((zip(*row.data)))
    fig.add_trace(go.Scatter(
        x=delta_s,
        y=estimate,
        name=r'$\delta_d = {0}$'.format(row.delta_t),
        marker={'color': colors[idx]},
        mode='lines',
        line=dict(dash=dashes[idx]),
        showlegend=False
    ), row=1, col=3)
    idx += 1


fig.update_yaxes(range=[-0.125, 5.625], row=1, col=1)
fig.update_yaxes(range=[-0.125, 5.625], row=1, col=2)
fig.update_yaxes(range=[-0.125, 5.625], row=1, col=3)
fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="green", row=1, col=1)
fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="green", row=1, col=2)
fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="green", row=1, col=3)


fig.update_xaxes(range=[0, 21], row=1, col=1)
fig.update_xaxes(range=[0, 21], row=1, col=2)
fig.update_xaxes(range=[0, 21], row=1, col=3)

fig.update_yaxes(title_text=r'$\huge{\hat{\theta}^\dagger \cdot (\delta_d / \delta_s)}$', row=1, col=1, title_standoff=30)

fig.update_xaxes(title_text=r'$\huge{\delta_s \ [\text{min}]}$', row=1, col=1)
fig.update_xaxes(title_text=r'$\huge{\delta_s \ [\text{min}]}$', row=1, col=2)
fig.update_xaxes(title_text=r'$\huge{\delta_s \ [\text{min}]}$', row=1, col=3)

annotations = fig.layout.annotations
annotations[0]['y'] = 1.025
annotations[1]['y'] = 1.025
annotations[2]['y'] = 1.025

fig.update_layout(annotations=annotations,
                  legend=dict(
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.9125
                  )
                  )


fig.write_image('./results_nyc/scatter.png', width=w, height=h, scale=1)
#
#
# mst_cost = results_df.mst_cost.max()
# delta_list = np.linspace(0.5, 20, 40)
# cost_list = [mst_cost * 60 / delta for delta in delta_list]
# fig = px.line(x=delta_list, y=cost_list)
# fig.update_yaxes(title_text=r'$\huge{\text{Cost} \ [\text{m}/\text{hr}]}$')
# fig.update_xaxes(title_text=r'$\huge{\delta \ [\text{min}]}$')
# fig.write_image('./results_nyc/mst_cost.png', scale=1)
#
#
#
# #
# #
# # for _, row in results_df.iterrows():
# #     g = row.g
# #     _map = folium.Map(location=(row.g.nodes[42421877]['y'], row.g.nodes[42421877]['x']), zoom_start=13.5)
# #     for u, v, data in row.g.edges(data=True):
# #         points = [(row.g.nodes[u]['y'], row.g.nodes[u]['x']), (row.g.nodes[v]['y'], row.g.nodes[v]['x'])]
# #         folium.PolyLine(points, color='blue', weight=3, opacity=(np.exp(data['p'])-1)/(np.exp(1)-1)).add_to(_map)
# #     _map.fit_bounds(bounds=[(40.70038507925808, -74.02759529869529), (40.763849522046876, -73.94755407175593)])
# #     _map.save('./results_nyc/delta_t_{0}_rho_{1}_g.html'.format(row.delta_t, round(row.rho, 2)))
# #
# #
# #
#
#
# # --- old --- #
#
# # --- plot solution --- #
# # g_copy = g.copy()
# # delta_t = delta_t_l[2]
# # for u, v, data in g_copy.edges(data=True):
# #     data['p'] = p_dicts[delta_t][(u, v)]
# # _map = folium.Map(location=(g_copy.nodes[42421877]['y'], g_copy.nodes[42421877]['x']))
# # ps = [data['p'] for _, _, data in g_copy.edges(data=True)]
# # for u, v, data in g_copy.edges(data=True):
# #     points = [(g_copy.nodes[u]['y'], g_copy.nodes[u]['x']), (g_copy.nodes[v]['y'], g_copy.nodes[v]['x'])]
# #     folium.PolyLine(points, color='blue', weight=3, opacity=data['p']).add_to(_map)
# # _map.save('./solution.html')
#
# # --- plot stops --- #
# # _map = folium.Map(location=(g.nodes[42421728]['y'], g.nodes[42421728]['x']))
# # for u, data in g.nodes(data=True):
# #     if data['stop']:
# #         folium.Circle(radius=20, location=[data['y'], data['x']]).add_to(_map)
# # _map.save('./stops.html')
#
#
# # --- plot g --- #
# # _map = folium.Map(location=(g.nodes[42421877]['y'], g.nodes[42421877]['x']))
# # for u, v, data in g.edges(data=True):
# #     points = [(g.nodes[u]['y'], g.nodes[u]['x']), (g.nodes[v]['y'], g.nodes[v]['x'])]
# #     folium.PolyLine(points, color='blue', weight=2).add_to(_map)
# # _map.save('./g.html')
#
# # two y axes
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots
# # fig = make_subplots(rows=1, cols=3, specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]])
# # for idx, rho in enumerate(results_df['rho'].unique()):
# #     rho_df = results_df[results_df['rho'] == rho]
# #     fig.add_trace(
# #         go.Scatter(x=rho_df['delta_t'], y=rho_df['a'], name="yaxis data"),
# #         secondary_y=False, row=1, col=idx+1
# #     )
# #     fig.add_trace(
# #         go.Scatter(x=rho_df['delta_t'], y=rho_df['c'], name="yaxis2 data"),
# #         secondary_y=True, row=1, col=idx+1
# #     )
# # fig.write_image('./results_nyc/scatter2.png', width=w, height=h, scale=1)