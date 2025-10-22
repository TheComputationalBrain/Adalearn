"""
This script plots figures for the correlation analysis examining relationships between various variables and uncertainty-associated effects on learning rate. The data used is from the present study and the study of Foucaut & Meyniel (2024). The following files must be run before executing this file to obtain the relevant .csv files: 
- adalearn_pupil_group_level_analysis_uncertainty_correlations
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        "present_study_corr_var_df",
        "foucault_meyniel_2024_corr_var_df",
        "present_v_foucault_meyniel_var_ttest_df",
        "uncertainty_determinants_corr_df",
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

foucault_meyniel_dataframes = group_data_utils.load_sample_csvs(
    directory_path=Path(BASE_DIR, 'foucault_meyniel_2024_reanalysis'),
    substrings=[
        'foucault_meyniel_2024_uncert_lr_regress_coefs_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

##################Scatter plot of uncertainty coefficients from both studies
present_study_coefs = dataframes['present_study_corr_var_df'].copy()
foucault_meyniel_coefs = dataframes['foucault_meyniel_2024_corr_var_df'].copy()
between_study_coef_means = dataframes['present_v_foucault_meyniel_var_ttest_df'].copy()


# Plotting figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=foucault_meyniel_coefs['conf_lr_coef'],
    y=foucault_meyniel_coefs['cpp_lr_coef'],
    mode='markers',
    marker=dict(color='rgba(255, 102, 51, 0.6)', size=18),
    name="Foucault & Meyniel (2024)"))

fig.add_trace(go.Scatter(
    x=present_study_coefs['conf_lr_coef'],
    y=present_study_coefs['cpp_lr_coef'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.6)', size=18),
    name="Present data"))

fig.add_hline(y=0, line_dash='dot')
fig.add_vline(x=0, line_dash='dot')

# Add mean line for each study and each variable
# Present study
fig.add_hline(
    y=between_study_coef_means.loc[between_study_coef_means['column'] == 'cpp_lr_coef', 'mean1'].iloc[0],
    line_color='rgba(0, 130, 150, 0.75)',
    line_width=5
)
fig.add_vline(
    x=between_study_coef_means.loc[between_study_coef_means['column'] == 'conf_lr_coef', 'mean1'].iloc[0],
    line_color='rgba(0, 130, 150, 0.75)',
    line_width=4
)

# Foucault & Meyniel (2024)
fig.add_hline(
    y=between_study_coef_means.loc[between_study_coef_means['column'] == 'cpp_lr_coef', 'mean2'].iloc[0],
    line_color='rgba(255, 102, 51, 0.75)',
    line_width=4
)
fig.add_vline(
    x=between_study_coef_means.loc[between_study_coef_means['column'] == 'conf_lr_coef', 'mean2'].iloc[0],
    line_color='rgba(255, 102, 51, 0.75)',
    line_width=4
)

fig.update_xaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Prior uncertainty regression weight", 
    range=[-0.6, 0.6]
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Change point probability regression weight", 
    range=[-0.6, 0.6]
)

fig.update_layout(
    height=2000,
    width=2000,
    showlegend=True,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=42, family='Times New Roman'),
    legend=dict(
        x=0.7,  
        y=0.9,  
        xanchor='left', 
        yanchor='top'
    ),
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-uncertainty_lr_regression_weights_scatterplot"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


######################################Correlations for learning rate effects
# Define variables
combined_df = pd.concat([present_study_coefs, foucault_meyniel_coefs], ignore_index=True)

y_vars = ['conf_lr_coef', 'cpp_lr_coef']
x_vars = ['avg_update_percentage', 'individual_pc', 'est_model_corr']

rename_dict = {
    'conf_lr_coef': 'Prior uncertainty regression weight',
    'cpp_lr_coef': 'Change point probability regression weight',
    'avg_update_percentage': 'Update freq (%)',
    'individual_pc': 'Individual volatility',
    'est_model_corr': 'Participant v model estimate correlation'
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
# Colors for the two studies
present_color = 'rgba(0, 130, 150, 0.6)'
foucault_color = 'rgba(255, 102, 51, 0.6)'

for idx, (y_var, x_var) in enumerate([(y, x) for y in y_vars for x in x_vars]):
    if x_var not in present_study_coefs.columns or y_var not in present_study_coefs.columns:
        continue

    row = idx // n_cols + 1
    col = idx % n_cols + 1
    first_plot = (idx == 0)

    # --- Present study ---
    x_present = present_study_coefs[x_var]
    y_present = present_study_coefs[y_var]
    fig.add_trace(go.Scatter(
        x=x_present,
        y=y_present,
        mode='markers',
        marker=dict(size=14, color=present_color),
        name="Present study",
        showlegend=first_plot
    ), row=row, col=col)

    # --- Foucault & Meyniel (2024) ---
    x_fm = foucault_meyniel_coefs[x_var]
    y_fm = foucault_meyniel_coefs[y_var]
    fig.add_trace(go.Scatter(
        x=x_fm,
        y=y_fm,
        mode='markers',
        marker=dict(size=14, color=foucault_color),
        name="Foucault & Meyniel (2024)",
        showlegend=first_plot
    ), row=row, col=col)

    # --- Combined regression line (black) ---
    x_combined = combined_df[x_var]
    y_combined = combined_df[y_var]
    if x_combined.notna().sum() > 2 and y_combined.notna().sum() > 2:
        slope, intercept, *_ = linregress(x_combined, y_combined)
        x_sorted = np.linspace(x_combined.min(), x_combined.max(), 100)
        y_fit = slope * x_sorted + intercept
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_fit,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ), row=row, col=col)


    # --- Axes ---
    fig.update_yaxes(
        title_text=rename_dict.get(y_var, y_var) if col == 1 else None,
        showline=True, linewidth=1.5, linecolor='black',
        ticks="outside", tickwidth=2, tickcolor='black', ticklen=6,
        row=row, col=col,
    )
    fig.update_xaxes(
        title_text=rename_dict.get(x_var, x_var) if row == n_rows else None,
        showline=True, linewidth=1.5, linecolor='black',
        ticks="outside", tickwidth=2, tickcolor='black', ticklen=6,
        row=row, col=col,
    )

fig.update_layout(
    height=1800,
    width=2400,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=36, family='Times New Roman'),
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-learning_rate_uncertainty_effects_correlations"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()
