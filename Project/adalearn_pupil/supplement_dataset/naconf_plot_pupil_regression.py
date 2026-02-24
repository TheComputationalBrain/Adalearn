"""
This script plots figures for the peristimulus analysis of the supplemental dataset (NACONF) to the Pupil task of the Adalearn project. The following figures are generated:
- Perstimulus regression coeficients for prior uncertainty and change point probability on the baseline correct pupil signal
- Perstimulus regression coeficients for prior uncertainty and change point probability on the non-baseline correct pupil signal
"""

import os
import sys
import numpy as np
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import NACONF_BASE_DIR


dataframes = group_data_utils.load_sample_csvs(
    directory_path=NACONF_BASE_DIR, 
    substrings=[ 
        'coefs_df', 
        'coefs_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc',
        'between_coef_significant_clusters_df',
        'between_coef_significant_clusters_nobc_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil_NACONF-')

###################################################Baseline corrected peristimulus regression 
coefs_df = dataframes['coefs_df']
clusters_df = dataframes['cluster_permutation_analysis']
between_coefs_clusters_df = dataframes['between_coef_significant_clusters_df']

time = coefs_df['time'][:-1]
conf = coefs_df['prior_conf_zscore'][:-1]
conf_error = coefs_df['prior_conf_zscore_error'][:-1]
cpp = coefs_df['cpp_zscore'][:-1]
cpp_error = coefs_df['cpp_zscore_error'][:-1]

fig = go.Figure()

# CPP line
fig.add_trace(go.Scatter(
    x=time, 
    y=cpp, 
    mode='lines',
    line=dict(color='rgba(220, 120, 0, 0.8)', width=4),  
    name='Change point probability'))

# CPP error fill
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(cpp + cpp_error) + list((cpp - cpp_error)[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

# Confidence line
fig.add_trace(go.Scatter(
    x=time, 
    y=conf, 
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4), 
    name='Prior uncertainty'))

# Confidence error fill
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(conf + conf_error) + list((conf - conf_error)[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vrect(
    x0=-0.25, 
    x1=0,
    fillcolor='rgba(210, 0, 150, 0.2)', 
    opacity=0.4,               
    layer="above",            
    line_width=0
)

# Reference lines
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=3))

# Cluster significance markers
var_style_map = {
    'cpp_zscore': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.03},
    'prior_conf_zscore': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.04}
}

for _, row in clusters_df.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})
        cluster_range = coefs_df['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']] * len(cluster_range),
            mode='lines',
            line=dict(color=style['color'], width=8),
            showlegend=False,
            hoverinfo='skip'
        ))

# Between-coefficient cluster significance markers
for _, row in between_coefs_clusters_df.iterrows():
    if row['p_value'] < 0.05:
        cluster_range = coefs_df['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']- 0.005] * len(cluster_range),
            mode='lines',
            line=dict(color='black', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        continue

# Axes & layout
fig.update_xaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Regression weights",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    font=dict(size=30, family="Times New Roman"),
    legend=dict(
        x=0.75,
        y=0.4,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)')
)

# Save
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

base_filename = "adalearn_pupil-NACONF_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

fig.write_image(f"{base_filepath}.png", scale=3.125)
fig.write_image(f"{base_filepath}.svg", scale=3.125)

fig.show()


###################################################Nonbaseline corrected peristimulus regression 
coefs_nobc_df = dataframes['coefs_nobc_df']
clusters_nobc_df = dataframes['cluster_permutation_analysis_nobc']
between_coefs_clusters_nobc_df = dataframes['between_coef_significant_clusters_nobc_df']

time = coefs_nobc_df['time'][:-1]
conf = coefs_nobc_df['prior_conf_zscore'][:-1]
conf_error = coefs_nobc_df['prior_conf_zscore_error'][:-1]
cpp = coefs_nobc_df['cpp_zscore'][:-1]
cpp_error = coefs_nobc_df['cpp_zscore_error'][:-1]

fig = go.Figure()

# CPP line
fig.add_trace(go.Scatter(
    x=time, 
    y=cpp, 
    mode='lines',
    line=dict(color='rgba(220, 120, 0, 0.8)', width=4),  
    name='Change point probability'))

# CPP error fill
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(cpp + cpp_error) + list((cpp - cpp_error)[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

# Confidence line
fig.add_trace(go.Scatter(
    x=time, 
    y=conf, 
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4), 
    name='Prior uncertainty'))

# Confidence error fill
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(conf + conf_error) + list((conf - conf_error)[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

# Reference lines
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=3))

# Cluster significance markers
var_style_map = {
    'cpp_zscore': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.06},
    'prior_conf_zscore': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.08}
}

for _, row in clusters_nobc_df.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})
        cluster_range = coefs_df['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']] * len(cluster_range),
            mode='lines',
            line=dict(color=style['color'], width=8),
            showlegend=False,
            hoverinfo='skip'
        ))

# Between-coefficient cluster significance markers
for _, row in between_coefs_clusters_nobc_df.iterrows():
    if row['p_value'] < 0.05:
        cluster_range = coefs_nobc_df['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']- 0.01] * len(cluster_range),
            mode='lines',
            line=dict(color='black', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        continue

# Axes & layout
fig.update_xaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Regression weights",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
) 

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    font=dict(size=30, family="Times New Roman"),
    legend=dict(
        x=0.95,
        y=0.4,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)')
)

# Save
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

base_filename = "adalearn_pupil-NACONF_nonbaseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

fig.write_image(f"{base_filepath}.png", scale=3.125)
fig.write_image(f"{base_filepath}.svg", scale=3.125)

fig.show()



