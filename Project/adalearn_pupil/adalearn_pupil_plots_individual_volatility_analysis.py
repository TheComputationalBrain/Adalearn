"""
This script plots figures for analysis examining the effect of defining individual volatility for participant undertaking the Pupil task in the Adalearn project. The effects of individual volataility are examined on learning rate and pupil responses. The following figures are generated:
- Change in change point probability for an example participant in one task session
- Prior confidence and change point probability on learning rate
- Prior confidence and change point probabilty on baseline corrected peristimulus regression
- Prior confidence and change point probabilty on non-baseline corrected peristimulus regression
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import selected_participant,base_dir


# Loading individual volatility data 
dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'individual_p_c_df',
        'ind_vol_comp_df',
        'computational_var_df', 
        'ind_vol_uncertainty_params_on_lr_df',
        'ind_vol_uncertainty_params_ttest_df',
        'ind_vol_coefs_df', 
        'ind_vol_coefs_nobc_df', 
        'ind_vol_cluster_permutation_analysis', 
        'ind_vol_cluster_permutation_analysis_nobc'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

################################################# Example change in change point probability with individual volatility for a single participant 
# Creating dataframe for comaprison of generator and individual volatility
ind_vol = dataframes['individual_p_c_df']
ind_vol_comp = dataframes['ind_vol_comp_df']
computational_var_df = dataframes['computational_var_df']

task_session = 15

original_comp_data = computational_var_df[computational_var_df['participant'] == selected_participant]
original_task_session = original_comp_data[original_comp_data['task_session']==task_session]
ind_vol_comp_data = ind_vol_comp[ind_vol_comp['participant'] == selected_participant]
ind_vol_task_session = ind_vol_comp_data[ind_vol_comp_data['task_session'] == task_session]

vol_comparison = pd.DataFrame({
    'participant': selected_participant,
    'outcome_num': original_task_session['outcome_num'].values,
    'cpp': original_task_session['cpp'].values,
    'ind_vol_cpp': ind_vol_task_session['cpp'].values,
})

    
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=vol_comparison['outcome_num'],
    y=vol_comparison['cpp'],
    mode='lines+markers',
    name='Generative volatility',
    line=dict(color='rgba(0, 130, 150, 0.3)', width=2, dash='dash'),
    marker=dict(size=8, color='rgba(0, 130, 150, 0.3)'),
    hoverinfo="skip",
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=vol_comparison['outcome_num'],
    y=vol_comparison['ind_vol_cpp'],
    mode='lines+markers',
    name='Individual volatility',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=2, dash='solid'),
    marker=dict(size=8, color='rgba(0, 130, 150, 0.8)'),
    hoverinfo="skip",
    showlegend=True
))

fig.update_xaxes(
    title_text="Outcome",
    showline=True,
    linewidth=1,
    linecolor='black'
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text='Change point probability'
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=42, family="Times New Roman"),
    height=1200,
    width=1800,
    legend=dict(
        x=0.95,
        y=0.95,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-example_participant_change_in_cpp_individual_volatility"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()

#################################################### Uncertainty parameters on learning rate 
# Evaluating the uncertainty parameters on learning rate for the individual volatility 
uncertainty_params = dataframes['ind_vol_uncertainty_params_on_lr_df']
uncertainty_params_ttest_df = dataframes['ind_vol_uncertainty_params_ttest_df']

behav_conf_coef = uncertainty_params[uncertainty_params['variable'] == 'conf_zscore'].sort_values('participant')
behav_cpp_coef = uncertainty_params[uncertainty_params['variable'] == 'cpp_zscore'].sort_values('participant')


behav_conf_coef_mean = behav_conf_coef['coefficient'].mean()
behav_cpp_coef_mean = behav_cpp_coef['coefficient'].mean()
behav_conf_coef_sem = stats.sem(behav_conf_coef['coefficient'])
behav_cpp_coef_sem = stats.sem(behav_cpp_coef['coefficient'])

model_conf_coef_mean = behav_conf_coef['coefficient_model'].mean()
model_cpp_coef_mean = behav_cpp_coef['coefficient_model'].mean()
model_conf_coef_sem = stats.sem(behav_conf_coef['coefficient_model'])
model_cpp_coef_sem = stats.sem(behav_cpp_coef['coefficient_model'])

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Prior confidence (z-score)',
    x=[0], y=[behav_conf_coef_mean],
    error_y=dict(type='data', array=[behav_conf_coef_sem]),
    showlegend=False
))

fig.add_trace(go.Bar(
    name='Change point probability (z-score)',
    x=[1], y=[behav_cpp_coef_mean],
    error_y=dict(type='data', array=[behav_cpp_coef_sem]),
    showlegend=False
))

fig.add_hline(y=0, line_dash='dot')

# Determine y positions for stars (slightly above error bar max)
y_offset = 0.25  # adjust as needed
conf_star_y = behav_conf_coef_mean + behav_conf_coef_sem - y_offset
cpp_star_y = behav_cpp_coef_mean + behav_cpp_coef_sem - y_offset
between_star_y = max(conf_star_y, cpp_star_y) - 0.2

# Add asterisks if significant
if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'conf_zscore', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=0, y=conf_star_y,
        showarrow=False,
        font=dict(size=60)
)

if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'cpp_zscore', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=1, y=cpp_star_y,
        showarrow=False,
        font=dict(size=60)
)

# Optional: Add a bar and star for between-group comparison
if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'between_coef', 'p_value'
].values[0] < 0.05:
    fig.add_shape(
        type='line',
        x0=0, 
        x1=1,
        y0=between_star_y,
        y1=between_star_y,
        line=dict(color='black', width=1.5))
    fig.add_annotation(
        text='*',
        x=0.5, 
        y=between_star_y - 0.05,
        showarrow=False,
        font=dict(size=60)
)    

fig.update_traces(
    marker_color='rgba(0, 130, 150, 0.4)', 
    marker_line_color='rgba(0, 130, 150, 0.8)', 
    marker_line_width=1.0)

x_positions = {'Prior confidence (z-score)': 0, 'Change point probability (z-score)': 1}

# Jitter width
jitter_width = 0.08

# Jittered scatter x values
np.random.seed(42)

# Generate jittered x values for each group
x_jittered_conf = np.random.normal(loc=x_positions['Prior confidence (z-score)'], scale=jitter_width, size=len(behav_conf_coef))
x_jittered_cpp = np.random.normal(loc=x_positions['Change point probability (z-score)'], scale=jitter_width, size=len(behav_cpp_coef))

# Add jittered scatter points — use numeric jittered x, actual y
fig.add_trace(go.Scatter(
    x=x_jittered_conf,
    y=behav_conf_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size=16),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=x_jittered_cpp,
    y=behav_cpp_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size=16),
    showlegend=False
))

fig.update_xaxes(
    tickvals=[0, 1],
    ticktext=['Prior confidence (z-score)', 'Change point probability (z-score)'],showline=True, 
    linewidth=1.5, 
    linecolor='black')

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Regression weight (participant)")

# Add inset bars (model) - assigned to secondary x2/y2 axes
fig.add_trace(go.Bar(
    name='Model Prior confidence',
    x=['Prior conf'], y=[model_conf_coef_mean],
    error_y=dict(type='data', array=[model_conf_coef_sem]),
    showlegend=False,
    xaxis='x2',
    yaxis='y2',
    marker_color='rgba(255, 0, 0, 0.6)',
    marker_line_color='rgba(255, 0, 0, 0.8)',
    marker_line_width=1.0
))

fig.add_trace(go.Bar(
    name='Model Change point probability',
    x=['CPP'], y=[model_cpp_coef_mean],
    error_y=dict(type='data', array=[model_cpp_coef_sem]),
    showlegend=False,
    xaxis='x2',
    yaxis='y2',
    marker_color='rgba(255, 0, 0, 0.6)',
    marker_line_color='rgba(255, 0, 0, 0.8)',
    marker_line_width=1.0
))

fig.add_shape(
    type="line",
    x0=-0.5,  # left edge of first bar
    x1=1.5,   # right edge of second bar
    y0=0, y1=0,
    xref="x2", yref="y2",
    line=dict(
        color="black",
        width=1.5,
        dash="dot"
    ),
    layer="above")

fig.update_layout(
    xaxis2=dict(
        domain=[0.1, 0.35],  # inset x position (change to relocate)
        anchor='y2',
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickfont=dict(size=36),     
        titlefont=dict(size=32)
    ),
    yaxis2=dict(
        title_text="Regression weight (model)",
        domain=[0.75, 1.0],  # inset y position (change to resize/relocate)
        anchor='x2',
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickfont=dict(size=36),       
        titlefont=dict(size=32)
    ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=42, family="Times New Roman"),
    height=1200,
    width=1800
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_uncertainty_params_on_lr_regression_individual_volatility"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


############################################################Peristimulus regression (baseline corrected)
baseline_corrected_coefs = dataframes['ind_vol_coefs_df']
baseline_corrected_clusters = dataframes['ind_vol_cluster_permutation_analysis']

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=baseline_corrected_coefs['time'], 
    y=baseline_corrected_coefs['cpp'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
    name='Change point probability'))

fig.add_trace(go.Scatter(
    x=list(baseline_corrected_coefs['time']) + list(baseline_corrected_coefs['time'][::-1]),
    y=list(baseline_corrected_coefs['cpp'] + baseline_corrected_coefs['cpp_error']) + list((baseline_corrected_coefs['cpp'] - baseline_corrected_coefs['cpp_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=baseline_corrected_coefs['time'], 
    y=baseline_corrected_coefs['prior_conf'], 
    mode= 'lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'), 
    name='Prior confidence'))

fig.add_trace(go.Scatter(
    x=list(baseline_corrected_coefs['time']) + list(baseline_corrected_coefs['time'][::-1]),
    y=list(baseline_corrected_coefs['prior_conf'] + baseline_corrected_coefs['prior_conf_error']) + list((baseline_corrected_coefs['prior_conf'] - baseline_corrected_coefs['prior_conf_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))


fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))


var_style_map = {
    'cpp': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.05},
    'prior_conf': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.07}
}

# Add significant clusters with separation by variable
for _, row in baseline_corrected_clusters.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})  # fallback

        cluster_range = baseline_corrected_coefs['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']] * len(cluster_range),  # Staggered y placement
            mode='lines',
            line=dict(color=style['color'], width=8),
            showlegend=False,
            hoverinfo='skip'
        ))

fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Regression weights",  
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
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_baseline_corrected_peristimulus_regression_individual_volatility"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


fig.show()

####################################################################Peristimulus regression (nonbaseline-corrected)
nonbaseline_corrected_coefs = dataframes['ind_vol_coefs_nobc_df']
nonbaseline_corrected_clusters = dataframes['ind_vol_cluster_permutation_analysis_nobc']

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=nonbaseline_corrected_coefs['time'], 
    y=nonbaseline_corrected_coefs['cpp'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
    name='Change point probability'))

fig.add_trace(go.Scatter(
    x=list(nonbaseline_corrected_coefs['time']) + list(nonbaseline_corrected_coefs['time'][::-1]),
    y=list(nonbaseline_corrected_coefs['cpp'] + nonbaseline_corrected_coefs['cpp_error']) + list((nonbaseline_corrected_coefs['cpp'] - nonbaseline_corrected_coefs['cpp_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=nonbaseline_corrected_coefs['time'], 
    y=nonbaseline_corrected_coefs['prior_conf'], 
    mode= 'lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'), 
    name='Prior confidence'))

fig.add_trace(go.Scatter(
    x=list(nonbaseline_corrected_coefs['time']) + list(nonbaseline_corrected_coefs['time'][::-1]),
    y=list(nonbaseline_corrected_coefs['prior_conf'] + nonbaseline_corrected_coefs['prior_conf_error']) + list((nonbaseline_corrected_coefs['prior_conf'] - nonbaseline_corrected_coefs['prior_conf_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))


fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))


var_style_map = {
    'cpp': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.26},
    'prior_conf': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.28}
}

# Add significant clusters with separation by variable
for _, row in nonbaseline_corrected_clusters.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})  # fallback

        cluster_range = nonbaseline_corrected_coefs['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']] * len(cluster_range),  # Staggered y placement
            mode='lines',
            line=dict(color=style['color'], width=8),
            showlegend=False,
            hoverinfo='skip'
        ))
        

fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Regression weights",  
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=1200,
    width=1800,
    font=dict(size=42, family="Times New Roman"),
    legend=dict(
        x=0.5,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_non_baseline_corrected_peristimulus_regression_individual_volatility"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


fig.show()



