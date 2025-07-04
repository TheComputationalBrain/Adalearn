import pandas as pd
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'



def compute_group_level_permutation_cluster_test(group_level_df, pivot_column = 'count', vars = [], alpha = 0.05):
    """
    Run permutation cluster 1-sample t-tests on specified regression coefficient variables.

    Parameters:
    - group_level_df: pd.DataFrame, contains the group level data. For eaxample, for the peristimulus regression analysis it contains the regression coefficients for the parameters of uncertainty. Must contain columns ['participant', 'count'] + variables of interest.
    - pivot_column: str, column name to pivot on (default is 'count').
    - vars: list of strings, List of variable names (e.g., ['prior_conf', 'cpp']) to test against the criterion. Must be within the group_level_df.
    - alpha: float, significance threshold for clusters.

    Returns:
    - results: dict, keys are var names, values are dicts with T_obs, clusters, p_values.
    - sig_clusters_df: pd.DataFrame summarizing significant clusters with columns ['variable', 'cluster', 'timepoint', 'T_obs', 'p_value'].
    """
    results = {}
    cluster_data = []  

    for var in vars:
        data_matrix = group_level_df.pivot(
            index='participant', columns=pivot_column, values=var
        ).values

        # Remove columns that are all NaNs
        data_matrix = data_matrix[:, ~np.isnan(data_matrix).all(axis=0)]

        # Run one-sample permutation analysis
        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
            data_matrix, n_permutations=1000, threshold=None, tail=0,
            out_type='mask', verbose=False, seed=42
        )

        results[var] = {
            'T_obs': T_obs,
            'clusters': clusters,
            'p_values': p_values
        }

        num_sig_clusters = np.sum(p_values < alpha)
        print(f"{var}: {num_sig_clusters} significant cluster(s) (p < {alpha})")

        # Extract info for significant clusters
        for cluster_idx, p_val in enumerate(p_values):
            if p_val < alpha:
                cluster_slice = clusters[cluster_idx][0]  # the slice object for this cluster
                cluster_start = cluster_slice.start
                cluster_end = cluster_slice.stop
                cluster_data.append({
                    'variable': var,
                    'cluster_idx': cluster_idx,
                    'cluster_start': cluster_start,
                    'cluster_end': cluster_end,
                    'p_value': p_val
                })

    significant_clusters_df = pd.DataFrame(cluster_data)
    return results, significant_clusters_df



def corr_with_alt(x, y):
    """
    Helper function for Pearson correlation with alternative hypothesis included

    Parameters:
    - x: pd.Dataframe, column of the x variable
    - y: pd.Dataframe, column of the y variable

    Returns
    - Output of the stats.pearsonr
    """
    return stats.pearsonr(x, y, alternative='two-sided')
