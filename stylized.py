import itertools as it
import joblib as jb
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs
import plotly.subplots
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
from helper import scatter_df_helper, curve_df_helper
pio.renderers.default = "browser"


def main():

    no_nodes_l = [3, 4, 5, 6, 7]
    p_rule_l = ['unif', 'closeness_cent', 'inv_closeness_cent']
    for no_nodes in no_nodes_l:
        for p_rule in p_rule_l:
            print('Working on: {0}, {1}'.format(no_nodes, p_rule))
            generate_data(no_nodes, p_rule, curve=True)


def generate_data(no_nodes=6, p_rule='unif', curve=True):

    # collect graphs
    graphs = [g for g in nx.graph_atlas_g() if nx.number_of_nodes(g) == no_nodes and nx.is_connected(g)]

    # list possible terminal sets
    terminals = []
    for r in range(no_nodes + 1):
        for combination in it.combinations(range(no_nodes), r):
            cur_terminals = [1 if idx in combination else 0 for idx in range(no_nodes)]
            terminals.append(cur_terminals)
    terminals = np.array(terminals)

    # collect data
    scatter_data = jb.Parallel(n_jobs=-1, verbose=1)(
        jb.delayed(scatter_df_helper)(*job) for job in [(p_rule, g, terminals) for g in graphs]
    )

    # process and store data
    scatter_df = pd.DataFrame(
        [(g, g.name, g._avg_deg, g._v_conn, g._alg_conn, g._dg1) for g in scatter_data],
        columns=['graph', 'g_name', 'avg_deg', 'v_conn', 'alg_conn', 'dg1']
    )
    pd.to_pickle(scatter_df, './stylized/scatter_df_{0}_nodes_{1}_p_rule.pkl'.format(no_nodes, p_rule))

    if curve:

        # noinspection PyUnresolvedReferences
        thetas = np.linspace(0.25, 3.5, 53)  # must contain 1

        # collect data
        curve_data = jb.Parallel(n_jobs=-1, verbose=1)(
            jb.delayed(curve_df_helper)(*job) for job in [
                (p_rule, g, terminals, theta) for g in scatter_data for theta in thetas
            ]
        )

        # process and store data
        curve_df = pd.DataFrame(
            [(g.name, theta, obj, g._obj_static / obj, max(1, g._dg1 / theta)) for g, theta, obj in curve_data],
            columns=['g_name', 'theta', 'obj', 'dg', 'model_dg']
        )
        curve_df['diff_dg'] = curve_df['dg'] / curve_df['model_dg']
        curve_df = curve_df.sort_values(by=['g_name', 'theta'])

        pd.to_pickle(curve_df, './stylized/curve_df_{0}_nodes_{1}_p_rule.pkl'.format(no_nodes, p_rule))


def load_data(no_nodes=6, p_rule='unif', curve=True):

    scatter_df = pd.read_pickle('./stylized/scatter_df_{0}_nodes_{1}_p_rule.pkl'.format(no_nodes, p_rule))

    if curve:
        curve_df = pd.read_pickle('./stylized/curve_df_{0}_nodes_{1}_p_rule.pkl'.format(no_nodes, p_rule))
    else:
        curve_df = None

    return scatter_df, curve_df


def scatter_plotter():

    no_nodes = 4
    p_rule_l = ['unif', 'closeness_cent', 'inv_closeness_cent']
    w = 1800 * 0.8
    h = 600
    file_format_l = ['pdf', 'png']

    dist_names = {
        'unif': r'$\huge{\mathcal{U}}$',
        'closeness_cent': r'$\huge{\mathcal{D}^{+\text{cent}}}$',
        'inv_closeness_cent': r'$\huge{\mathcal{D}^{-\text{cent}}}$'
    }

    for p_rule in p_rule_l:

        if p_rule == 'unif':
            r2_idx = -1
            r2_pos = 'top left',
        elif p_rule == 'closeness_cent':
            r2_idx = -1
            r2_pos = 'bottom left'
        else:
            r2_idx = -1
            r2_pos = 'top left'

        scatter_df, _ = load_data(no_nodes, p_rule, curve=False)
        scatter_df['n'] = no_nodes
        scatter_df['p_rule'] = dist_names[p_rule]

        fig = plotly.subplots.make_subplots(rows=1, cols=3, shared_yaxes=True)
        fig.update_layout(title_text=dist_names[p_rule])

        fig.add_trace(plotly.graph_objs.Scatter(
            x=scatter_df['avg_deg'],
            y=scatter_df['dg1'],
            marker={'color': 'blue'},
            mode='markers',
            showlegend=False
        ), row=1, col=1)
        dg1 = scatter_df['dg1']
        avg_deg = np.array(scatter_df['avg_deg'])
        fit_results = sm.OLS(dg1, sm.add_constant(avg_deg), missing='drop').fit()
        fit_dg1 = fit_results.predict()
        fit_avg_deg = avg_deg[np.logical_not(np.logical_or(np.isnan(dg1), np.isnan(avg_deg)))]
        text = ['' for _ in fit_dg1]
        text[r2_idx] = r'$R^2: {0:0.2f}$'.format(fit_results.rsquared)
        fig.add_trace(plotly.graph_objs.Scatter(
            x=fit_avg_deg, y=fit_dg1, mode='lines+text', line=dict(color='red', dash='dash', width=1),
            text=text, textposition=r2_pos, textfont=dict(size=14), showlegend=False),
            row=1, col=1)
        fig.update_xaxes(title_text=r'$\huge{\overline{d}(G)}$', row=1, col=1)
        fig.update_yaxes(title_text=r'$\huge{\theta^\dagger}$', row=1, col=1)

        fig.add_trace(plotly.graph_objs.Scatter(
            x=scatter_df['v_conn'],
            y=scatter_df['dg1'],
            marker={'color': 'blue'},
            mode='markers',
            showlegend=False
        ), row=1, col=2)
        dg1 = scatter_df['dg1']
        v_conn = np.array(scatter_df['v_conn'])
        fit_results = sm.OLS(dg1, sm.add_constant(v_conn), missing='drop').fit()
        fit_dg1 = fit_results.predict()
        fit_v_conn = v_conn[np.logical_not(np.logical_or(np.isnan(dg1), np.isnan(v_conn)))]
        text = ['' for _ in fit_dg1]
        text[r2_idx] = r'$R^2: {0:0.2f}$'.format(fit_results.rsquared)
        fig.add_trace(plotly.graph_objs.Scatter(
            x=fit_v_conn, y=fit_dg1, mode='lines+text', line=dict(color='red', dash='dash', width=1),
            text=text, textposition=r2_pos, textfont=dict(size=14), showlegend=False),
            row=1, col=2)
        fig.update_xaxes(title_text=r'$\huge{\overline{\kappa}(G)}$', row=1, col=2)

        fig.add_trace(plotly.graph_objs.Scatter(
            x=scatter_df['alg_conn'],
            y=scatter_df['dg1'],
            marker={'color': 'blue'},
            mode='markers',
            showlegend=False
        ), row=1, col=3)
        dg1 = scatter_df['dg1']
        alg_conn = np.array(scatter_df['alg_conn'])
        fit_results = sm.OLS(dg1, sm.add_constant(alg_conn), missing='drop').fit()
        fit_dg1 = fit_results.predict()
        fit_alg_conn = alg_conn[np.logical_not(np.logical_or(np.isnan(dg1), np.isnan(alg_conn)))]
        text = ['' for _ in fit_dg1]
        text[r2_idx] = r'$R^2: {0:0.2f}$'.format(fit_results.rsquared)
        fig.add_trace(plotly.graph_objs.Scatter(
            x=fit_alg_conn, y=fit_dg1, mode='lines+text', line=dict(color='red', dash='dash', width=1),
            text=text, textposition=r2_pos, textfont=dict(size=14), showlegend=False),
            row=1, col=3)
        fig.update_xaxes(title_text=r'$\huge{a(G)}$', row=1, col=3)

        for file_format in file_format_l:
            fig.write_image(
                './images/{0}_fit_{1}.{2}'.format(p_rule, no_nodes, file_format), width=w, height=h, scale=1
            )


def curve_plotter():

    no_nodes = 4
    p_rule_l = ['unif', 'closeness_cent', 'inv_closeness_cent']
    w = 1800 * 0.8
    h = 600
    file_format_l = ['pdf', 'png']

    dist_names = {
        'unif': r'$\huge{\mathcal{U}}$',
        'closeness_cent': r'$\huge{\mathcal{D}^{+\text{cent}}}$',
        'inv_closeness_cent': r'$\huge{\mathcal{D}^{-\text{cent}}}$'
    }

    curves_df = []

    for p_rule in p_rule_l:

        if p_rule == 'unif':
            r2_idx = -1
            r2_pos = 'top left',
        elif p_rule == 'closeness_cent':
            r2_idx = -1
            r2_pos = 'bottom left'
        else:
            r2_idx = -1
            r2_pos = 'top left'

        _, curve_df = load_data(no_nodes, p_rule, curve=True)
        curve_df['n'] = no_nodes
        curve_df['p_rule'] = dist_names[p_rule]

        curves_df.append(curve_df)

    curves_df = pd.concat(curves_df)
    fig = px.line(
        curves_df, x='theta', y='diff_dg', facet_row='n', facet_col='p_rule', line_group='g_name', hover_name='g_name',
        hover_data={
            'n': False,
            'p_rule': False
        },
        labels={
             'g_name': 'Graph',
             'n': r'$\huge{n}$',
             'theta': r'$\huge{\theta}$',
             'diff_dg': r'$\huge{\alpha(\theta) / \hat{\alpha}(\theta)}$'
        }
    )
    fig.update_layout(font_size=20)
    fig.update_xaxes(tickfont_size=25)
    fig.update_yaxes(tickfont_size=25, range=[-0.125, 5.125])
    fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="green")
    fig.add_vline(x=1, line_width=1, line_dash="dash", line_color="green")
    for a in fig.layout.annotations:
        if a.text.split("=")[0] == '$\\huge{n}$':
            a.text = r'$\huge{n = ' + '{0}}}$'.format(a.text.split("=")[-1])
            a.x += 0.005
        else:
            a.text = a.text.split("=")[1]

    for file_format in file_format_l:
        fig.write_image('./images/{0}.{1}'.format('diff_dg', file_format), width=w, height=h, scale=1)

    # if full_curve:
       #      results_dfs = []
       #      for no_nodes in no_nodes_l:
       #          for p_rule in p_rule_l:
       #              _, results_df = load_data(problem_solver, no_nodes, p_rule, full_curve=full_curve)
       #              results_df['n'] = no_nodes
       #              results_df['p_rule'] = dist_names[p_rule]
       #              results_dfs.append(results_df)
       #      results_df = pd.concat(results_dfs)
       #
       #      for alpha in alpha_l:
       #          fig = px.line(
       #              results_df, x='theta', y=alpha, facet_row='n', facet_col='p_rule', line_group='g_name',
       #              hover_name='g_name',
       #              hover_data={
       #                  'n': False,
       #                  'p_rule': False,
       #              },
       #              labels={
       #                  'g_name': 'Graph',
       #                  'n': r'$\huge{n}$',
       #                  'theta': r'$\huge{\theta}$',
       #                  'alpha': r'$\huge{\alpha(\theta)}$',
       #                  'model_alpha': r'$\huge{\hat{\alpha}^-(\theta)}$',
       #                  'diff_alpha': r'$\huge{\alpha(\theta) - \hat{\alpha}^-(\theta)}$'
       #              },
       #              title='Problem: {0}'.format(problem_names[problem_solver.__name__])
       #          )
       #          fig.update_layout(font_size=20)
       #          fig.update_xaxes(tickfont_size=25)
       #          fig.update_yaxes(tickfont_size=25, range=[-0.125, 5.125])
       #          fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="green")
       #          fig.add_vline(x=1, line_width=1, line_dash="dash", line_color="green")
       #          for a in fig.layout.annotations:
       #              if a.text.split("=")[0] == '$\\huge{n}$':
       #                  a.text = r'$\huge{n = ' + '{0}}}$'.format(a.text.split("=")[-1])
       #                  a.x += 0.005
       #              else:
       #                  a.text = a.text.split("=")[1]
       #          #fig.show()
       #
       #          for file_format in file_format_l:
       #              fig.write_image('./images/{0}_{1}.{2}'.format(problem_solver.__name__, alpha, file_format),
       #                              width=w, height=h, scale=1)


main()
# scatter_plotter()
# curve_plotter()