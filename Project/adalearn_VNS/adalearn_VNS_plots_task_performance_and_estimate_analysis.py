"""
This script plots figures related to performance on the task for participants undertaking the VNS task in the Adalearn project. The following plots are generated:
- Participants' estimate absolute error, coefficient of variation and score across task sessions and conditions
- Relationship between participants' and model estimates   
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_participant_level_functions_data_utilities as data_utils 
import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_VNS.adalearn_VNS_participant_info_and_parameters import base_dir



dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'behavioural_df', 
        'change_point_indices_df', 
        'mean_abs_error_df'
        ],
    prefix='sample_level_analysis-ses_combined-task_Adalearn_VNS-')


############################################Plotting perforamnce metrics (absolute error,coefficient of variation and score)
behav_df = dataframes['mean_abs_error_df'].copy()

# Ensure participant is integer
behav_df['participant'] = behav_df['participant'].astype(int)

# Main effect of task session 
task_session_summary = (
    behav_df.groupby('task_session')
      .agg({
          'mean_abs_error': ['mean', stats.sem, 'count'],
          'coef_variation': ['mean', stats.sem, 'count'],
          'score': ['mean', stats.sem, 'count']
      })
)

# Flatten MultiIndex columns
task_session_summary.columns = ['_'.join(col) for col in task_session_summary.columns]
task_session_summary = task_session_summary.reset_index()

# Main efect of condition
condition_summary = (
    behav_df.groupby(['participant', 'condition'])
      .agg({
          'mean_abs_error': ['mean', stats.sem, 'count'],
          'coef_variation': ['mean', stats.sem, 'count'],
          'score': ['mean', stats.sem, 'count']
      })
)

# Flatten columns
condition_summary.columns = ['_'.join(col) for col in condition_summary.columns]
condition_summary = condition_summary.reset_index()

# Interaction effect
condition_session_summary = (
    behav_df.groupby(['condition', 'task_session'])
      .agg({'mean_abs_error': ['mean', stats.sem, 'count'],
          'coef_variation': ['mean', stats.sem, 'count'],
          'score': ['mean', stats.sem, 'count']
      })
      .reset_index()
)

# Flatten MultiIndex columns
condition_session_summary.columns = ['_'.join(col) for col in condition_session_summary.columns]
condition_session_summary = condition_session_summary.reset_index()


# Plotting data
# # Create subplot
fig = make_subplots(rows=3, cols=3)

# Main effect of condition
# Common settings
x_bar = [0, 1]
jitter_strength = 0.05
bar_colors = ['rgba(140, 0, 140, 0.5)', 'rgba(250, 235, 107, 0.5)']
dot_colors = ['rgba(140, 0, 140, 0.8)', 'rgba(250, 235, 107, 0.9)']
x_tickvals = [0, 1]
x_ticktext = ['High Intensity', 'Low Intensity']

# Column info: (column name in df, subplot position: (row, col))
metrics = [
    ('mean_abs_error_mean', (1, 1)),
    ('coef_variation_mean', (1, 2)),
    ('score_mean', (1, 3))
]

for col_name, (a, b) in metrics:
    # Extract values
    high_vals = condition_summary[condition_summary['condition'] == 'high_intensity'][col_name].values
    low_vals = condition_summary[condition_summary['condition'] == 'low_intensity'][col_name].values

    # Means and SEMs
    y_bar = [high_vals.mean(), low_vals.mean()]
    sem_bar = [stats.sem(high_vals), stats.sem(low_vals)]

    # Jittered x values
    x_high = np.random.normal(loc=0, scale=jitter_strength, size=len(high_vals))
    x_low = np.random.normal(loc=1, scale=jitter_strength, size=len(low_vals))

    # Add bar
    fig.add_trace(go.Bar(
        x=x_bar,
        y=y_bar,
        error_y=dict(type='data', array=sem_bar, visible=True),
        marker_color=bar_colors,
        showlegend=False
    ), row=a, col=b)

    # Add dots
    fig.add_trace(go.Scatter(
        x=x_high,
        y=high_vals,
        mode='markers',
        marker=dict(color=dot_colors[0], size=12),
        showlegend=False
    ), row=a, col=b)

    fig.add_trace(go.Scatter(
        x=x_low,
        y=low_vals,
        mode='markers',
        marker=dict(color=dot_colors[1], size=12),
        showlegend=False
    ), row=a, col=b)

    # Update axis
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_tickvals,
        ticktext=x_ticktext,
        showline=True,               
        linewidth=1.5,                  
        linecolor='black', 
        row=a, col=b
    )

    # Update axis
    fig.update_yaxes(
        showline=True,               
        linewidth=1.5,                  
        linecolor='black', 
        row=a, col=b
    )

# Main effect of task session
# Column info: (column name in df, subplot position: (row, col))
metrics = [
    ('mean_abs_error_mean', (2, 1)),
    ('coef_variation_mean', (2, 2)),
    ('score_mean', (2, 3))
]

sem_metrics = [
    ('mean_abs_error_sem', (2, 1)),
    ('coef_variation_sem', (2, 2)),
    ('score_sem', (2, 3))
]

for (mean_col, (c, d)), (sem_col, _) in zip(metrics, sem_metrics):

    x_vals = task_session_summary['task_session']
    y_vals = task_session_summary[mean_col]
    y_sem = task_session_summary[sem_col]

    # Create upper and lower bounds for shaded area
    upper = y_vals + y_sem
    lower = y_vals - y_sem

    # Add SEM shaded area (closed polygon)
    fig.add_trace(go.Scatter(
        x=list(x_vals) + list(x_vals[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(0, 130, 150, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ), row=c, col=d)

    # Add mean line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='rgba(0, 130, 150, 1)', width=4),
        showlegend=False
    ), row=c, col=d)

     # Update axis
    fig.update_xaxes(
        range=[0, 21],
        showline=True,                
        linewidth=1.5,                  
        linecolor='black', 
        row=c, col=d
    )

    # Update axis
    fig.update_yaxes(
        showline=True,                
        linewidth=1.5,                  
        linecolor='black', 
        row=c, col=d
    )

# Main effect of condition and task session interaction
high_vals = condition_session_summary[condition_session_summary['condition_'] == 'high_intensity']
low_vals = condition_session_summary[condition_session_summary['condition_'] == 'low_intensity']

# Column info: (column name in df, subplot position: (row, col))
metrics = [
    ('mean_abs_error_mean', (3, 1)),
    ('coef_variation_mean', (3, 2)),
    ('score_mean', (3, 3))
]

sem_metrics = [
    ('mean_abs_error_sem', (3, 1)),
    ('coef_variation_sem', (3, 2)),
    ('score_sem', (3, 3))
]

for (mean_col, (e, f)), (sem_col, _) in zip(metrics, sem_metrics):

    high_x_vals = high_vals['task_session_']
    high_y_vals = high_vals[mean_col]
    high_y_sem = high_vals[sem_col]
    low_x_vals = low_vals['task_session_']
    low_y_vals = low_vals[mean_col]
    low_y_sem = low_vals[sem_col]

    high_upper = high_y_vals + high_y_sem
    high_lower = high_y_vals - high_y_sem
    low_upper = low_y_vals + low_y_sem
    low_lower = low_y_vals - low_y_sem

    # Add SEM shaded area (closed polygon)
    fig.add_trace(go.Scatter(
        x=list(high_x_vals) + list(high_x_vals[::-1]),
        y=list(high_upper) + list(high_lower[::-1]),
        fill='toself',
        fillcolor='rgba(140, 0, 140, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ), row=e, col=f)

    # Add mean line
    fig.add_trace(go.Scatter(
        x=high_x_vals,
        y=high_y_vals,
        mode='lines',
        line=dict(color='rgba(140, 0, 140, 1.0)', width=4),
        showlegend=False
    ), row=e, col=f)

    # Add SEM shaded area (closed polygon)
    fig.add_trace(go.Scatter(
        x=list(low_x_vals) + list(low_x_vals[::-1]),
        y=list(low_upper) + list(low_lower[::-1]),
        fill='toself',
        fillcolor='rgba(250, 235, 107, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ), row=e, col=f)

    # Add mean line
    fig.add_trace(go.Scatter(
        x=low_x_vals,
        y=low_y_vals,
        mode='lines',
        line=dict(color='rgba(250, 235, 107, 1.0)', width=4),
        showlegend=False
    ), row=e, col=f)

    # Update axis
    fig.update_xaxes(
        range=[0, 21],
        showline=True,                
        linewidth=1.5,                  
        linecolor='black', 
        row=e, col=f
    )

    # Update axis
    fig.update_yaxes(
        showline=True,                
        linewidth=1.5,                  
        linecolor='black', 
        row=e, col=f
    )


# Update axis titles
fig.update_xaxes(title_text="Task Session", row=2, col=1)
fig.update_xaxes(title_text="Task Session", row=2, col=2)
fig.update_xaxes(title_text="Task Session", row=2, col=3)
fig.update_xaxes(title_text="Task Session", row=3, col=1)
fig.update_xaxes(title_text="Task Session", row=3, col=2)
fig.update_xaxes(title_text="Task Session", row=3, col=3)
fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=1)
fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=1)
fig.update_yaxes(title_text="Mean Absolute Error", row=3, col=1)
fig.update_yaxes(title_text="Coefficient of Variation (%)", row=1, col=2)
fig.update_yaxes(title_text="Coefficient of Variation (%)", row=2, col=2)
fig.update_yaxes(title_text="Coefficient of Variation (%)", row=3, col=2)
fig.update_yaxes(title_text="Score", row=1, col=3)
fig.update_yaxes(title_text="Score", row=2, col=3)
fig.update_yaxes(title_text="Score", row=3, col=3)

# Layout
fig.update_layout(
    height=1500,
    width=2400,
    showlegend=True,
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=32, family='Times New Roman')
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_VNS-task_performance_metrics"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


############################################Plotting relationship between participants' and model estimates