import pandas as pd
import os
import numpy as np
from scipy import stats
import pingouin
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'



def safe_adjust_index(idx, max_idx):
    """
    Function to safely adjust a 1-based index to 0-based, ensuring it doesn't become negative or exceed bounds.

    Parameters:
    - idx: int, the index to adjust, typically from cluster permutation output (may be 1-based or include 0).
    - max_idx: int, the maximum valid index for the target data (e.g., length of a DataFrame).

    Returns:

    - int: A valid 0-based index within the range [0, max_idx - 1].

    """
    if idx == 0:
        return 0
    
    return min(idx - 1, max_idx - 1)



def extract_cluster_times(cluster_df, time_series, alpha=0.05):
    """
    A function to extract the start and end times for the longest significant cluster from a cluster permutation result. Filters clusters by statistical significance, computes their durations, selects the longest one, and returns the corresponding time values from the time series.

    Parameters:
    
    - cluster_df: pd.DataFrame, DataFrame containing cluster information. Must include 'cluster_start', 'cluster_end', and 'p_value' columns.
    - time_series: pd.Series, time values associated with the rows of the data used in the analysis.
    - alpha: float, optional significance threshold for filtering clusters (default is 0.05).

    Returns:

    - tuple of (float, float) or None: The (start_time, end_time) of the longest significant cluster. Returns None if no clusters are significant.
    """
    # Filter for significant clusters
    significant_clusters = cluster_df[cluster_df['p_value'] < alpha].copy()
    if significant_clusters.empty:
        return None  # or raise warning

    # Calculate duration and select longest cluster
    significant_clusters['duration'] = significant_clusters['cluster_end'] - significant_clusters['cluster_start']
    longest_cluster = significant_clusters.loc[significant_clusters['duration'].idxmax()]

    max_idx = len(time_series)

    # Adjust indices safely
    start_idx = safe_adjust_index(longest_cluster['cluster_start'], max_idx)
    end_idx = safe_adjust_index(longest_cluster['cluster_end'], max_idx)

    return time_series.iloc[start_idx], time_series.iloc[end_idx]



def compute_peristim_mean_sem_by_count(df, value_cols, time_col='time', group_col='count', sort_col='participant'):
    """
    Function to calculate the mean and SEM for specified columns following peristimulus regression of pupil epochs,  grouped by a counter variable.

    Parameters: 

    - pupil_regress_coef_df : pd.DataFrame, input DataFrame with regression coefficients across the epoch for each participant.
    - value_cols : list of strings, columns for which to compute the mean and SEM.
    - time_col : str, (optional) name of the column representing time (default is 'time').
    - group_col : str, (optional) column to group by for epoch-level averaging (default is 'count').
    - sort_col : str, (optional) column to sort by within each group (default is 'participant').

    Returns:
    
    - pd.DataFrame: A DataFrame containing the group-level means and SEMs for each value column, along with average time.
    """
    result = {col: [] for col in value_cols}
    result.update({f"{col}_error": [] for col in value_cols})
    result['time'] = []

    for count, group in df.groupby(group_col):
        group = group.sort_values(by=sort_col)

        for col in value_cols:
            result[col].append(group[col].mean())
            result[f"{col}_error"].append(stats.sem(group[col], nan_policy='omit'))

        result['time'].append(group[time_col].mean())

    return pd.DataFrame(result)