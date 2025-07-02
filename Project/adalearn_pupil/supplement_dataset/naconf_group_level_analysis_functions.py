import pandas as pd
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'



def compute_group_level_permutation_cluster_test(group_level_regression_coefs_df, vars = []):
    """
    Run permutation cluster 1-sample t-tests on specified regression coefficient variables.

    Parameters:
    - group_level_regression_coefs_df: pd.DataFrame, contains the regression coefficients. Must contain columns ['participant', 'count'] + variables of interest.
    - vars: list of strings, List of variable names (e.g., ['prior_conf', 'cpp']) to test. Must be within the group_level_regression_coefs_df.

    Returns:
    results: Dictionary, where keys are var names, and values are dicts with T_obs, clusters, p_values.
    """
    results = {}
    
    for var in vars:
        data_matrix = group_level_regression_coefs_df.pivot(
            index='participant', columns='count', values=var
        ).values

        data_matrix = data_matrix[:, ~np.isnan(data_matrix).all(axis=0)]

        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
            data_matrix, n_permutations=1000, threshold=None, tail=0, out_type='mask', verbose=True
        )

        results[var] = {
            'T_obs': T_obs,
            'clusters': clusters,
            'p_values': p_values
        }

    return results  



def group_level_peristimulus_regression_analysis(group_level_regression_coefs_df, vars = [], baseline_corrected = True, plot_regression_analysis = True):
    """
    Perform group-level analysis on regression coefficients.

    Parameters:
    - group_level_regression_coefs_df: pd.DataFrame, contains the regression coefficients. Must contain columns ['participant', 'count'] + variables of interest.
    - vars: list of strings, List of variable names (e.g., ['prior_conf', 'cpp']) to test. Must be within the group_level_regression_coefs_df.
    - baseline_corrected: boolean, if true, will correct labels and titles (default = True). Note* must be matched to the correct data.
    - plot_regression_analysis: boolean, if true, will plot regression coefficients and permutation cluster analysis of regression weights against zero (default = True).

    Returns:
    results: Dictionary, where keys are var names, and values are dicts with T_obs, clusters, p_values.
    """
    # Create labels and tiltes based on baseline correction
    if baseline_corrected:
        title = 'baseline corrected'
    else:
        title = 'non-baseline corrected'

    # Permutation cluster analysis
    cluster_analysis_results = compute_group_level_permutation_cluster_test(group_level_regression_coefs_df, vars=vars)

    # Create dictionary for plotting
    coefs_dict = {'cpp': [], 'cpp_error': [], 'conf': [], 'conf_error': [], 'time': [], 'count': []}

    for count in group_level_regression_coefs_df['count'].unique():
        data = group_level_regression_coefs_df[group_level_regression_coefs_df['count'] == count].sort_values(by='participant')
        mean_cpp = data['cpp_zscore'].mean()
        sem_cpp = stats.sem(data['cpp_zscore'])
        mean_conf = data['prior_conf_zscore'].mean()
        sem_conf = stats.sem(data['prior_conf_zscore'])
        mean_time = data['time'].mean()
        i_count = count

        coefs_dict['cpp'].append(mean_cpp)
        coefs_dict['cpp_error'].append(sem_cpp)
        coefs_dict['conf'].append(mean_conf)
        coefs_dict['conf_error'].append(sem_conf)
        coefs_dict['time'].append(mean_time)
        coefs_dict['count'].append(i_count)

    coefs_df = pd.DataFrame(coefs_dict)

    # Option for plotting data
    if plot_regression_analysis:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=coefs_df['time'], 
            y=coefs_df['cpp'], 
            mode= 'lines',
            line=dict(color='rgba(220, 120, 0, 0.8)'),  
            name='Change point probability'))

        fig.add_trace(go.Scatter(
            x=list(coefs_df['time']) + list(coefs_df['time'][::-1]),
            y=list(coefs_df['cpp'] + coefs_df['cpp_error']) + list((coefs_df['cpp'] - coefs_df['cpp_error'])[::-1]),
            fill='toself',
            fillcolor='rgba(220, 120, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        cpp_cluster_slice = cluster_analysis_results['cpp_zscore']['clusters'][0][0]
        cpp_cluster_indices = list(range(cpp_cluster_slice.start, cpp_cluster_slice.stop))  
        cpp_significant_times = coefs_df['time'].iloc[cpp_cluster_indices]

        # Add dots for significant clusters (at the time points of the cluster)
        fig.add_trace(go.Scatter(
            x=cpp_significant_times,  # Time points where the cluster is significant
            y=[-0.02] * len(cpp_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
            mode='markers',
            marker=dict(color='rgba(220, 120, 0, 0.6)', size=6),
            name='Significant Cluster - CPP',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=coefs_df['time'], 
            y=coefs_df['conf'], 
            mode= 'lines',
            line=dict(color='rgba(0, 130, 150, 0.8)'), 
            name='Prior confidence'))

        fig.add_trace(go.Scatter(
            x=list(coefs_df['time']) + list(coefs_df['time'][::-1]),
            y=list(coefs_df['conf'] + coefs_df['conf_error']) + list((coefs_df['conf'] - coefs_df['conf_error'])[::-1]),
            fill='toself',
            fillcolor='rgba(0, 130, 150, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        conf_cluster_slice = cluster_analysis_results['prior_conf_zscore']['clusters'][0][0]
        conf_cluster_indices = list(range(conf_cluster_slice.start, conf_cluster_slice.stop))  
        conf_significant_times = coefs_df['time'].iloc[conf_cluster_indices]

        # Add dots for significant clusters (at the time points of the cluster)
        fig.add_trace(go.Scatter(
            x=conf_significant_times,  # Time points where the cluster is significant
            y=[-0.025] * len(conf_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
            mode='markers',
            marker=dict(color='rgba(0, 130, 150, 0.6)', size=6),
            name='Significant Cluster - Prior Confidence',
            showlegend=False
        ))

        fig.add_vline(x=0, line_dash='dot')
        fig.add_hline(y=0, line_dash = 'dot')

        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Time relative to outcome presentation (s)", 
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Regression weights",  
        )

        fig.update_layout(
            title=dict(
                    text="Peristimulus regression weights for prior confidence and change point probability on the {title} pupil signal"
                ),
            plot_bgcolor='rgba(255, 255, 255, 1)',
            height=950,
            width=1800,
            font=dict(size=18),
            legend=dict(
                x=0.3,
                y=0.9,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)'))


        fig.show()


    return cluster_analysis_results, coefs_df