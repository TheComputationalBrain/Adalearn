"""
This script plots figures for correlation analysis examining relationships between various variables and uncertainty-associated effects on learning rate and pupil signals. the following files must be run before executing this file to obtain the relevant .csv files: 
- adalearn_pupil_group_level_analysis_uncertainty_correlations
"""
from math import ceil
from scipy.stats import linregress
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str



dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        "uncertainty_lr_cor_data",
        "uncertainty_pupil_cor_data"
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

######################################Correlations for learning rate effects
# Define variables
y_vars = ['conf_lr_coef', 'cpp_lr_coef']
x_vars = ['Age', 'EducationLevel', 'avg_update_percentage', 'optimal_p_c', 'est_corr']

rename_dict = {
    'conf_lr_coef': 'Prior conf. effect on learning rate (Beta)',
    'cpp_lr_coef': 'CPP effect on learning rate (Beta)',
    'Age': 'Age (years)',
    'EducationLevel': 'Education (years)',
    'avg_update_percentage': 'Update freq (%)',
    'optimal_p_c': 'Individual volatility',
    'est_corr': 'Est. v model correlation'
}

n_rows = len(y_vars)
n_cols = len(x_vars)

# Create subplot grid
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
    shared_xaxes=True,
    shared_yaxes=True
)

# Loop with flat indexing to assign correct row and col
for idx, (y_var, x_var) in enumerate([(y, x) for y in y_vars for x in x_vars]):
    if x_var not in dataframes['uncertainty_lr_cor_data'].columns or y_var not in dataframes['uncertainty_lr_cor_data'].columns:
        continue

    x = dataframes['uncertainty_lr_cor_data'][x_var]
    y = dataframes['uncertainty_lr_cor_data'][y_var]

    # Determine subplot position
    row = idx // n_cols + 1
    col = idx % n_cols + 1

    # Add scatterplot
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=14, color='rgba(0, 130, 150, 0.8)'),
        showlegend=False,
    ), row=row, col=col)

    # Add regression line
    if x.notna().sum() > 2 and y.notna().sum() > 2:
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        x_sorted = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_sorted + intercept

        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_fit,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ), row=row, col=col)

    # Only show y-axis title and ticks on first column
    show_y = (col == 1)
    fig.update_yaxes(
        title_text=rename_dict.get(y_var, y_var) if show_y else None,
        showline=True,
        linewidth=1.5, 
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        tickcolor='black',
        ticklen=6,
        row=row, col=col
    )

    # Only show x-axis title and ticks on bottom row
    show_x = (row == n_rows)
    fig.update_xaxes(
        title_text=rename_dict.get(x_var, x_var) if show_x else None,
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        tickcolor='black',
        ticklen=6,
        row=row, col=col
    )

# Layout tweaks
fig.update_layout(
    height=1800,
    width=2400,
    showlegend=False,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=36, family='Times New Roman'),
    margin=dict(t=40),
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-learning_rate_uncertainty_effects_correlations"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()



###################################################################################################Correlations for pupil effects
# Define variables
y_vars = ['bc_pupil', 'nobc_pupil']
x_vars = ['Age', 'EducationLevel', 'avg_update_percentage', 'optimal_p_c', 'est_corr']

rename_dict = {
    'bc_pupil': 'Baseline corrected pupil signal (avg. effect)',
    'nobc_pupil': 'Non-baseline corrected pupil signal (avg. effect)',
    'Age': 'Age (years)',
    'EducationLevel': 'Education (years)',
    'avg_update_percentage': 'Update freq (%)',
    'optimal_p_c': 'Individual volatility',
    'est_corr': 'Est. v model correlation'
}

n_rows = len(y_vars)
n_cols = len(x_vars)

# Create subplot grid
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
    shared_xaxes=True,
    shared_yaxes=True
)

# Loop with flat indexing to assign correct row and col
for idx, (y_var, x_var) in enumerate([(y, x) for y in y_vars for x in x_vars]):
    if x_var not in dataframes['uncertainty_pupil_cor_data'].columns or y_var not in dataframes['uncertainty_pupil_cor_data'].columns:
        continue

    x = dataframes['uncertainty_pupil_cor_data'][x_var]
    y = dataframes['uncertainty_pupil_cor_data'][y_var]

    # Determine subplot position
    row = idx // n_cols + 1
    col = idx % n_cols + 1

    # Add scatterplot
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=14, color='rgba(0, 130, 150, 0.8)'),
        showlegend=False,
    ), row=row, col=col)

    # Add regression line
    if x.notna().sum() > 2 and y.notna().sum() > 2:
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        x_sorted = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_sorted + intercept

        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_fit,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ), row=row, col=col)

    # Only show y-axis title and ticks on first column
    show_y = (col == 1)
    fig.update_yaxes(
        title_text=rename_dict.get(y_var, y_var) if show_y else None,
        showline=True,
        linewidth=1.5, 
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        tickcolor='black',
        ticklen=6,
        row=row, col=col
    )

    # Only show x-axis title and ticks on bottom row
    show_x = (row == n_rows)
    fig.update_xaxes(
        title_text=rename_dict.get(x_var, x_var) if show_x else None,
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        tickcolor='black',
        ticklen=6,
        row=row, col=col
    )

# Layout tweaks
fig.update_layout(
    height=1800,
    width=2400,
    showlegend=False,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=36, family='Times New Roman'),
    margin=dict(t=40),
)

# Base filename (without extension)
base_filename = f"adalearn_pupil-pupil_uncertainty_effects_correlations"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()
