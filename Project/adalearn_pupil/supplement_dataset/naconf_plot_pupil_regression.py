"""
This script plots figures for the peristimulus analysis of the supplemental dataset (NACONF) to the Pupil task of the Adalearn project. The following figures are generated:
- Perstimulus regression coeficients for prior confidence and change point probability on the baseline correct pupil signal
- Perstimulus regression coeficients for prior confidence and change point probability on the non-baseline correct pupil signal
"""

import os
import sys
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import NACONF_base_dir


dataframes = group_data_utils.load_sample_csvs(
    directory_path=NACONF_base_dir, 
    substrings=[ 
        'coefs_df', 
        'coefs_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil_NACONF-')

###################################################Baseline corrected peristimulus regression 
coefs_df = dataframes['coefs_df']
clusters_df = dataframes['cluster_permutation_analysis']

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
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
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
    line=dict(color='rgba(0, 130, 150, 0.8)'), 
    name='Prior confidence'))

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
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=1.5))

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

# Axes & layout
fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)"
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Regression weights"
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=1200,
    width=1800,
    font=dict(size=42, family="Times New Roman"),
    legend=dict(
        x=0.45,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)')
)

# Save
plot_dir = os.path.join(NACONF_base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

base_filename = "adalearn_pupil-NACONF_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

fig.write_image(f"{base_filepath}.png")
fig.write_image(f"{base_filepath}.svg")

fig.show()


###################################################Nonbaseline corrected peristimulus regression 

coefs_nobc_df = dataframes['coefs_nobc_df']
clusters_nobc_df = dataframes['cluster_permutation_analysis_nobc']

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
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
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
    line=dict(color='rgba(0, 130, 150, 0.8)'), 
    name='Prior confidence'))

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
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=1.5))

# Cluster significance markers
var_style_map = {
    'cpp_zscore': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.10},
    'prior_conf_zscore': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.12}
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

# Axes & layout
fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)"
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Regression weights",
    range=[-0.15, 0.06]
) 

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=1200,
    width=1800,
    font=dict(size=42, family="Times New Roman"),
    legend=dict(
        x=0.50,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)')
)

# Save
plot_dir = os.path.join(NACONF_base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

base_filename = "adalearn_pupil-NACONF_nonbaseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

fig.write_image(f"{base_filepath}.png")
fig.write_image(f"{base_filepath}.svg")

fig.show()



