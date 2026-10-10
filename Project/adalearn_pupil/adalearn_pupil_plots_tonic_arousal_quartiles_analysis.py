"""
This script plots figures for the analysis split by tonic pupil signal quartiles, to evaluate the effect of arousal level on relationships between uncertainty parameters and learning rate. The following figures are generated:
- Regression weights for prior uncertainty and change point probability on learning rate split between tonic pupil signal quartiles
- Regression weights for change point probability on phasic pupil signals split between tonic pupil signal quartiles
- Scatter plots of the regression weights for change point probability on learning rate against the regression weights for change point probability on phasic pupil signals, split between tonic pupil signal quartiles
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'tonic_quartiles_cpp_lr_betas', 
        'tonic_quartiles_cpp_pupil_betas', 
        'tonic_quartiles_uncertainty_v_lr_regression', 
        'tonic_quartiles_uncertainty_v_lr_regression_anova', 
        'tonic_quartiles_uncertainty_v_lr_regression_posthoc', 
        'tonic_quartiles_cpp_pupil_betas_ttest', 
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')


############################################################################## Dataframes
all_quartile_uncert_lr_regress_df = dataframes['tonic_quartiles_uncertainty_v_lr_regression']
cpp_lr_betas = dataframes['tonic_quartiles_cpp_lr_betas']
cpp_pupil_betas = dataframes['tonic_quartiles_cpp_pupil_betas']
cpp_lr_posthoc_results_df = dataframes['tonic_quartiles_uncertainty_v_lr_regression_posthoc']
cpp_pupil_ttest_results_df = dataframes['tonic_quartiles_cpp_pupil_betas_ttest']


############################################## Parameters
variables = ['cpp_zscore', 'conf_zscore']
var_labels = ['Change point probability <br>(z-score)', 'Prior uncertainty <br>(z-score)']
var_x = {v: i for i, v in enumerate(variables)}

quartiles = all_quartile_uncert_lr_regress_df['pupil_group'].unique()
offsets = {'Q1': -0.3, 'Q2': -0.1, 'Q3': 0.1, 'Q4': 0.3}
quartile_colors = {'Q1': "#c9fcf5", 'Q2': "#85ebda", 'Q3': "#37ddc7", 'Q4': "#0c7c64"}
quartile_point_colors = {'Q1': "#5ad8b9", 'Q2': "#2fb09a", 'Q3': "#0c7c6d", 'Q4': "#04534f"}
scatter_quartile_colors = {'Q1': '#56B4E9','Q2': '#009E73', 'Q3': '#E69F00', 'Q4': "#D54300"}
scatter_text_colors = {'Q1': "#00639C", 'Q2': "#006E32", 'Q3': "#AC7500", 'Q4': "#974200"}
scatter_text_positions = {'Q1': 'top left', 'Q2': 'bottom right', 'Q3': 'top left', 'Q4': 'top left'}

np.random.seed(0)
jitter_width = 0.035
bar_width = 0.18
################################################# Uncertainty v learning rate across tonic arousal quartiles


fig = go.Figure()

for q in quartiles:
    data = all_quartile_uncert_lr_regress_df[all_quartile_uncert_lr_regress_df['pupil_group'] == q]
    x_positions = [var_x[var] + offsets[q] for var in variables]

    fig.add_trace(go.Bar(
        name=f'Tonic pupil ({q})',
        x=x_positions,
        y=[data.loc[data['variable'] == var, 'coefficient'].mean()
           for var in variables],
        error_y=dict(
            type='data',
            array=[data.loc[data['variable'] == var, 'coefficient'].sem()
                   for var in variables]
        ),
        width=bar_width,
        marker_color=quartile_colors[q]
    ))

    for var in variables:
        vals = data.loc[data['variable'] == var, 'coefficient']
        x_base = var_x[var] + offsets[q]
        x_jittered = x_base + np.random.uniform(-jitter_width, jitter_width, size=len(vals))

        fig.add_trace(go.Scatter(
            x=x_jittered,
            y=vals,
            mode='markers',
            marker=dict(
                size=12,
                color=quartile_point_colors[q],
                opacity=0.4,
                line=dict(width=0.5, color='rgba(0,0,0,0.4)')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

fig.update_layout(barmode='overlay')
fig.add_hline(y=0, line_dash='dot', line_color='black')

# --- Significance annotations ---

def bar_top(variable, quartile):
    vals = data.loc[data['variable'] == variable, 'coefficient']
    return vals.mean() + vals.sem()

def bar_bottom(variable, quartile):
    vals = data.loc[data['variable'] == variable, 'coefficient']
    return vals.mean() - vals.sem()

def is_sig(ph, a, b):
    row = ph[((ph['A'] == a) & (ph['B'] == b)) | ((ph['A'] == b) & (ph['B'] == a))]
    return (not row.empty) and (row['p-corr'].values[0] < 0.05)

def points_top(variable, quartile):
    vals = data.loc[data['variable'] == variable, 'coefficient']
    return vals.max()

def points_bottom(variable, quartile):
    vals = data.loc[data['variable'] == variable, 'coefficient']
    return vals.min()

all_tops = [bar_top(v, q) for v in variables for q in quartiles]
all_bottoms = [bar_bottom(v, q) for v in variables for q in quartiles]
y_range = max(all_tops + [0]) - min(all_bottoms + [0])
tick_drop = 0.03 * y_range
bracket_pad = 0.06 * y_range
label_gap = 0.055 * y_range

for variable in variables:
    ph = cpp_lr_posthoc_results_df[cpp_lr_posthoc_results_df['variable'] == variable]
    if ph.empty:
        continue

    sig_partners = [q for q in ['Q1', 'Q2', 'Q3'] if is_sig(ph, 'Q4', q)]
    x_center = var_x[variable]

    if len(sig_partners) == 3:
        x0 = x_center + offsets['Q1']
        x1 = x_center + offsets['Q3']
        bracket_mid = (x0 + x1) / 2
        y_top = max(points_top(variable, q) for q in ['Q1', 'Q2', 'Q3']) + bracket_pad

        fig.add_shape(type='line', x0=x0, x1=x1, y0=y_top + label_gap, y1=y_top + label_gap,
                      line=dict(color='#555', width=1))
        fig.add_shape(type='line', x0=x0, x1=x0, y0=y_top + label_gap, y1=y_top - tick_drop,
                      line=dict(color='#555', width=1))
        fig.add_shape(type='line', x0=x1, x1=x1, y0=y_top + label_gap, y1=y_top - tick_drop,
                      line=dict(color='#555', width=1))
        fig.add_annotation(x=bracket_mid, y=y_top + label_gap + 0.1, text='*',
                            showarrow=False, font=dict(size=24))

        x_q4 = x_center + offsets['Q4']
        q4_top = points_top(variable, 'Q4')
        fig.add_shape(type='line', x0=bracket_mid, x1=bracket_mid, y0=y_top + label_gap, y1=y_top + label_gap + 0.05,
                              line=dict(color='#999', width=1, dash='dot'))
        fig.add_shape(type='line', x0=bracket_mid, x1=x_q4, y0=y_top + label_gap + 0.05, y1=y_top + label_gap + 0.05,
                      line=dict(color='#999', width=1, dash='dot'))
        # connector now drops to clear Q4's own data points, not just its bar+SEM
        fig.add_shape(type='line', x0=x_q4, x1=x_q4, y0=y_top + label_gap + 0.05,
                      y1= y_top - tick_drop,
                      line=dict(color='#999', width=1, dash='dot'))
    else:
        y_offset = bracket_pad
        max_top = max(points_top(variable, q) for q in quartiles)
        for _, row in ph.iterrows():
            if row['p-corr'] < 0.05:
                a, b = row['A'], row['B']
                x0, x1 = sorted([x_center + offsets[a], x_center + offsets[b]])
                y_top = max_top + y_offset
                fig.add_shape(type='line', x0=x0, x1=x1, y0=y_top + label_gap, y1=y_top + label_gap,
                              line=dict(color='#555', width=1))
                fig.add_annotation(x=(x0 + x1) / 2, y=y_top + (label_gap * 2), text='*',
                                    showarrow=False, font=dict(size=30))
                y_offset += 2 * bracket_pad

fig.update_layout(
    xaxis=dict(showline=True, linecolor='black', ticks="inside", tickwidth=1.5,
               tickcolor='black', range=[-0.6, len(variables) - 1 + 0.6], tickvals=list(range(len(variables))),
               ticktext=var_labels, tickfont=dict(size=30)),
    yaxis=dict(showline=True, linecolor='black', ticks="inside", tickwidth=1.5,
               tickcolor='black', tickfont=dict(size=30),
               title_text="Regression weight onto apparent learning rate <br> (participant)"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=30, family="Times New Roman"),
    height=720, width=840,
    legend=dict(x=0.95, y=0.30, xanchor='right', yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)', font=dict(size=24))
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_uncertainty_v_lr_regression_tonic_arousal_quartiles"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


################################################### change point probability v phasic pupil signals across tonic arousal quartiles


fig = go.Figure()

for q in cpp_pupil_betas.columns:
    vals = cpp_pupil_betas[q]

    i = offsets[q]

    fig.add_trace(go.Bar(
        x=[i], y=[vals.mean()],
        error_y=dict(type='data', array=[vals.sem()]),
        width=bar_width,
        marker_color=quartile_colors[q],
        showlegend=False
    ))

    x_jittered = i + np.random.uniform(-jitter_width, jitter_width, size=len(vals))
    fig.add_trace(go.Scatter(
        x=x_jittered, y=vals, mode='markers',
        marker=dict(size=12, color=quartile_point_colors[q], opacity=0.6,
                    line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
        showlegend=False, hoverinfo='skip'
    ))

    # per-quartile asterisk: one-sample t-test of correlation against zero
    for pupil_group in cpp_pupil_ttest_results_df['pupil_group'].unique():
        results = cpp_pupil_ttest_results_df[cpp_pupil_ttest_results_df['pupil_group'] == q]
        p_val = results['p_fdr'].values[0]
        if p_val < 0.05:
            y_pos = vals.max() + 0.05
            fig.add_annotation(x=i, y=y_pos, text='*', showarrow=False, font=dict(size=30))

fig.add_hline(y=0, line_dash='dot', line_color='black')


fig.update_layout(
    xaxis=dict(showline=True, linecolor='black', ticks='inside', tickwidth=1.5,
               tickcolor='black', tickvals=[offsets[q] for q in cpp_pupil_betas.columns],
               ticktext=list(cpp_pupil_betas.columns), tickfont=dict(size=30), 
               title_text='Tonic pupil signal quartile'),
    yaxis=dict(showline=True, linecolor='black', ticks='inside', tickwidth=1.5,
               tickcolor='black', tickfont=dict(size=30),
               title_text='Regression weights onto phasic pupil response'),
    plot_bgcolor='rgba(255,255,255,1)',
    font=dict(size=30, family='Times New Roman'),
    width=720,
    height=840,
)


# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_change_point_probability_v_phasic_pupil_signals_tonic_arousal_quartiles"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 


fig.show()

############################################# Scatter plot of regression weights for change point probability on learning rate against regression weights for change point probability on phasic pupil signals, split between tonic pupil signal quartiles


fig = go.Figure()

fig.add_hline(y=0, line_dash="dash", line_color='rgba(10,10,10,0.2)')
fig.add_vline(x=0, line_dash="dash", line_color='rgba(10,10,10,0.2)')

for q in quartiles:

    # individual participants
    fig.add_trace(go.Scatter(
        x=cpp_pupil_betas[q].values.flatten(),
        y=cpp_lr_betas[q].values.flatten(),
        mode='markers',
        marker=dict(size=12, color=scatter_quartile_colors[q], opacity=0.3,
                            line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
        name=f'{q}',
        showlegend=False 
    ))

    # quartile mean
    fig.add_trace(go.Scatter(
        x=[cpp_pupil_betas[q].mean()],
        y=[cpp_lr_betas[q].mean()],
        error_x=dict(
            type='data',
            array=[cpp_pupil_betas[q].sem()]
        ),
        error_y=dict(
            type='data',
            array=[cpp_lr_betas[q].sem()]
        ),
        mode='lines+markers+text',
        text=[q],
        textposition=scatter_text_positions[q],
        textfont=dict(
            color=scatter_text_colors[q],
            size=30
        ),
        marker=dict(size=20, color=scatter_quartile_colors[q], opacity=0.8,
                            line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
        showlegend=False
    ))

fig.update_layout(
   xaxis=dict(showline=True, linecolor='black', ticks='inside', tickwidth=1.5,
                  tickcolor='black', tickfont=dict(size=30), 
                  title_text='CPP - Phasic pupil signal <br>(Regression weights)'),
       yaxis=dict(showline=True, linecolor='black', ticks='inside', tickwidth=1.5,
                  tickcolor='black', tickfont=dict(size=30),
                  title_text='CPP - Apparent learning rate <br>(Regression weights)'),
       plot_bgcolor='rgba(255,255,255,1)',
       font=dict(size=30, family='Times New Roman'),
       width=720,
       height=840,
   )

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_tonic_arousal_quartiles_scatter_cpp_lr_v_cpp_pupil"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()