"""
This script plots figures for analysis of learning rate for participants undertaking the Pupil task in the Adalearn project.  The following figures are generated:
- The relationship between participants' and model learning rates
- The dynamics of learning rate around change points (with normative predictions)
- Regression coefficients of prior uncertainty and change point probability on learning rate
- Scatterplot of individual regression coefficients for prior uncertainty and change point probability
- Relationship between (binned) prior uncertainty and learning rate
- Dynamics of prior uncertainty and change point probability around change points
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

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
from adalearn_analysis_participant_level_functions_comp import variables_by_change_point

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'behavioural_df', 
        'computational_var_df', 
        'change_point_indices_df', 
        'behav_v_model_lr_corr_df',  
        'averaged_lr_around_cp_df', 
        'lr_around_cp_sig_clusters', 
        'uncertainty_params_on_lr_df', 
        'uncertainty_params_ttest_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

############################################Plotting the relationship between participants' and model learning rates

# Bins and plots behavioural estimates against model estimates
# Evaluating relationship between participants' estimates and model estimates
nbins_model_estimate = 6

_, bin_edges_model_estimate = pd.qcut(dataframes['computational_var_df']["cleaned_learning_rate_ideal_observer"], nbins_model_estimate, retbins=True)

dataframes['computational_var_df']["model_lr_bin"] = pd.cut(dataframes['computational_var_df']["cleaned_learning_rate_ideal_observer"], bins=bin_edges_model_estimate, labels=False, include_lowest=True)

avg_model_estimate_per_bin = dataframes['computational_var_df'].groupby("model_lr_bin")["cleaned_learning_rate_ideal_observer"].mean()

dataframes['computational_var_df'] = dataframes['computational_var_df'].merge(avg_model_estimate_per_bin, on="model_lr_bin", suffixes=("", "_bin_avg"))

# Storing the average learning rate at the participant level
binned_data =  {"model_lr_bin": [], "model_lr_bin_label": [], "avg_lr": [], "participant": []}

for participant in dataframes['computational_var_df']["participant"].unique():
    participant_data = dataframes['computational_var_df'][dataframes['computational_var_df']["participant"] == participant]
    
    for model_lr_bin in sorted(participant_data["model_lr_bin"].unique()):
            
        bin_data = participant_data[
            (participant_data["model_lr_bin"] == model_lr_bin)]
        
        avg_lr = bin_data["cleaned_learning_rate_estimate"].mean()
        bin_label = bin_data['cleaned_learning_rate_ideal_observer_bin_avg'].unique()[0]  

        binned_data['model_lr_bin_label'].append(bin_label)
        binned_data["model_lr_bin"].append(model_lr_bin)
        binned_data["avg_lr"].append(avg_lr)
        binned_data["participant"].append(participant)

participant_v_model_df = pd.DataFrame(binned_data)

group_avg_data = {"model_lr_bin": [], "model_lr_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

for bin in sorted(participant_v_model_df['model_lr_bin'].unique()):
    data = participant_v_model_df[participant_v_model_df['model_lr_bin'] == bin]
    lr = data['avg_lr'].mean()
    lr_sem = stats.sem(data['avg_lr'])
    bin_label = data['model_lr_bin_label'].unique()[0] 

    group_avg_data["model_lr_bin"].append(bin)
    group_avg_data["model_lr_bin_label"].append(bin_label)
    group_avg_data["group_avg_lr"].append(lr)
    group_avg_data["group_sem_lr"].append(lr_sem)

group_behav_v_model_lr_df = pd.DataFrame(group_avg_data)

# Plotting the relationship between participants' estimates and model estimates
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, 0.35],  
    y=[0, 0.35],  
    mode='lines',
    line=dict(color='rgba(50, 50, 50, 0.5)', dash='dot', width=4),
    name='Identity',
    showlegend=True
))

for participant in participant_v_model_df['participant'].unique():
    data = participant_v_model_df[participant_v_model_df['participant']==participant]

    fig.add_trace(go.Scatter(
        x=data['model_lr_bin_label'], 
        y=data['avg_lr'], 
        mode='lines+markers',
        line=dict(color='rgba(0, 130, 150, 0.12)', width=3),
        marker=dict(color='rgba(0, 130, 150, 0.15)', size=14),
        showlegend=False))

fig.add_trace(go.Scatter(
    x=group_behav_v_model_lr_df['model_lr_bin_label'],
    y=group_behav_v_model_lr_df['group_avg_lr'], 
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.8)', size=28), 
    showlegend=False, 
    ))

# Add histogram to the plot
hist_trace = go.Histogram(
    x=dataframes['behav_v_model_lr_corr_df'][dataframes['behav_v_model_lr_corr_df']['variable']=='cleaned_learning_rate_ideal_observer']['correlation'], 
    marker=dict(color='rgba(0, 130, 150, 0.8)'),
    xaxis='x2',  
    yaxis='y2',
    nbinsx=20,
    opacity=0.75,
    showlegend=False
)

fig.add_trace(hist_trace)

fig.update_layout(
    xaxis=dict(
        title=dict(text="Apparent learning rate (model)"),
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',         
    ),
    yaxis=dict(
        title=dict(text="Apparent learning rate (participant)"),
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',

    ),
    xaxis2=dict(
        domain=[0.75, 1.0],
        title=dict(text='Correlation coefficient', font=dict(size=26)),  
        anchor='y2',
        range=[0, 1.0],      
        showline=True,
        linecolor='black',
        linewidth=3,
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
        tickvals=[0.0, 0.5, 1.0],
    ),
    yaxis2=dict(
        domain=[0.2, 0.45],
        title=dict(text='Count', font=dict(size=26)),  
        anchor='x2',
        showline=True,
        linecolor='black',
        linewidth=3,
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
        tickvals=list(range(0, 5, 2))
    ),
    height=720,
    width=840,
    showlegend=False,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=30, family='Times New Roman'),
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_behav_v_model_lr_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()



###################################################Plotting learning rate around change point

lr_by_cp_mean_behav = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean'].mean()
lr_by_cp_count_behav = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean'].count()
lr_by_cp_std_behav = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean'].std()
lr_by_cp_sem_behav = lr_by_cp_std_behav / np.sqrt(lr_by_cp_count_behav)

lr_by_cp_avg_df_behav = pd.DataFrame({
    'mean': lr_by_cp_mean_behav,
    'std': lr_by_cp_std_behav,
    'sem': lr_by_cp_sem_behav,
    'count': lr_by_cp_count_behav
})

# Model stats
lr_by_cp_mean_model = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean_model'].mean()
lr_by_cp_count_model = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean_model'].count()
lr_by_cp_std_model = dataframes['averaged_lr_around_cp_df'].groupby('relative_position')['mean_model'].std()
lr_by_cp_sem_model = lr_by_cp_std_model / np.sqrt(lr_by_cp_count_model)

lr_by_cp_avg_df_model = pd.DataFrame({
    'mean': lr_by_cp_mean_model,
    'std': lr_by_cp_std_model,
    'sem': lr_by_cp_sem_model,
    'count': lr_by_cp_count_model
})

# Create plot
fig = go.Figure()

# --- Behavioral Learning Rate Trace
fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_behav.index, 
    y=lr_by_cp_avg_df_behav['mean'], 
    mode='lines', 
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4),
    name="Behavior Mean"
))

# SEM Shading (upper + lower)
fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_behav.index, 
    y=lr_by_cp_avg_df_behav['mean'] + lr_by_cp_avg_df_behav['sem'], 
    mode='lines', 
    line=dict(color='rgba(0, 130, 150, 0)'),  
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_behav.index, 
    y=lr_by_cp_avg_df_behav['mean'] - lr_by_cp_avg_df_behav['sem'], 
    mode='lines', 
    fill='tonexty', 
    fillcolor='rgba(0, 130, 150, 0.12)',  
    line=dict(color='rgba(0, 130, 150, 0)'),
    showlegend=False
))

# Vertical line at change point (0)
fig.add_vline(
    x=0, 
    line=dict(color='black', width=4, dash='dash'),
)

# Plot significant clusters on behavioral subplot
for _, row in dataframes['lr_around_cp_sig_clusters'].iterrows():
    if row['p_value'] < 0.05:
        style = {'color': 'rgba(0, 0, 0, 0.8)', 'yval': 0.045}
        cluster_range = range(row['cluster_start'], row['cluster_end'] + 1)
        fig.add_trace(go.Scatter(
            x=list(cluster_range),
            y=[style['yval']] * len(cluster_range),
           mode='lines',
            line=dict(color=style['color'], width=12),
            showlegend=False,
            hoverinfo='skip'
        ))

# --- Model Learning Rate Trace (second axis)
fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_model.index, 
    y=lr_by_cp_avg_df_model['mean'], 
    xaxis="x2", 
    yaxis="y2",
    mode='lines',
    line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
    name="Model Mean"
))

fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_model.index, 
    y=lr_by_cp_avg_df_model['mean'] + lr_by_cp_avg_df_model['sem'],
    xaxis="x2", 
    yaxis="y2", 
    mode='lines', 
    line=dict(color='rgba(255, 0, 0, 0)'),  
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=lr_by_cp_avg_df_model.index, 
    y=lr_by_cp_avg_df_model['mean'] - lr_by_cp_avg_df_model['sem'],
    xaxis="x2", 
    yaxis="y2", 
    mode='lines', 
    fill='tonexty', 
    fillcolor='rgba(255, 0, 0, 0.2)', 
    line=dict(color='rgba(255, 0, 0, 0)'),
    showlegend=False
))

# Vertical line at change point (0) in model subplot
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=lr_by_cp_avg_df_model['mean'].min() - 0.01,
    y1=lr_by_cp_avg_df_model['mean'].max() + 0.01,
    line=dict(color="black", width=4, dash="dash"),
    xref="x2", yref="y2"
)

# Layout
fig.update_layout(    
    xaxis=dict(
        range=[-3, 16],
        showline=True, 
        linewidth=3, 
        linecolor='black',
        title=dict(text="Outcome relative to change point"),
        tickvals=[i for i in range(0, 16, 3)],  # show only ticks >= 0
        ticktext=[str(i + 1) for i in range(0, 16, 3)], # shift each tick label by +1
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
    ),
    yaxis=dict(
        showline=True, 
        linewidth=3, 
        linecolor='black',
        title=dict(text="Apparent learning rate (participant)"),
        range=[0.041, max(0.025, lr_by_cp_avg_df_behav['mean'].max() + lr_by_cp_avg_df_behav['sem'].max())],
        dtick=0.025,
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
    ),
    xaxis2=dict(
        domain=[0.65, 1.0],
        anchor="y2",
        range=[-3, 16],
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=3,
        tickvals=[i for i in range(0, 16, 3)],  # show only ticks >= 0
        ticktext=[str(i + 1) for i in range(0, 16, 3)],  # shift each tick label by +1
        tickfont=dict(size=22),
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
    ),
    yaxis2=dict(
        domain=[0.65, 1.0],
        anchor="x2",
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=3,
        dtick=0.02,
        title=dict(text="Apparent learning rate (model)", font=dict(size=22)),
        tickfont=dict(size=22),
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
    ),
    font=dict(size=30, family="times new roman"),
    height=720,
    width=840,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lr_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


####################################################################### Uncertainty parameters on learning rate 

uncertainty_params = dataframes['uncertainty_params_on_lr_df']
uncertainty_params_ttest_df = dataframes['uncertainty_params_ttest_df']

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
    name='Change point probability (z-score)',
    x=[0], y=[behav_cpp_coef_mean],
    error_y=dict(type='data', array=[behav_cpp_coef_sem]),
    showlegend=False
))

fig.add_trace(go.Bar(
    name='Prior uncertainty (z-score)',
    x=[1], y=[behav_conf_coef_mean],
    error_y=dict(type='data', array=[behav_conf_coef_sem]),
    showlegend=False
))

fig.add_hline(y=0, line_dash='dot')

# Determine y positions for stars (slightly above error bar max)
y_offset = 0.35  # adjust as needed
conf_star_y = behav_conf_coef_mean + behav_conf_coef_sem + y_offset
cpp_star_y = behav_cpp_coef_mean + behav_cpp_coef_sem + y_offset
between_star_y = max(cpp_star_y, conf_star_y) + 0.1

# Add asterisks if significant
if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'cpp_zscore', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=0, y=conf_star_y,
        showarrow=False,
        font=dict(size=30)
    )

if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'conf_zscore', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=1, y=cpp_star_y,
        showarrow=False,
        font=dict(size=30)
    )

# Optional: Add a bar and star for between-group comparison
if uncertainty_params_ttest_df.loc[
    uncertainty_params_ttest_df['variable'] == 'between_coef', 'p_value'
].values[0] < 0.05:
    fig.add_shape(type='line',
                  x0=0, x1=1,
                  y0=between_star_y, y1=between_star_y,
                  line=dict(color='black', width=1.5))
    fig.add_annotation(
        text='*',
        x=0.5, y=between_star_y + 0.02,
        showarrow=False,
        font=dict(size=30)
    )

fig.update_traces(
    marker_color='rgba(0, 130, 150, 0.4)', 
    marker_line_color='rgba(0, 130, 150, 0.8)', 
    marker_line_width=1.0,
)

x_positions = {'Change point probability (z-score)': 0, 'Prior uncertainty (z-score)': 1}

# Jitter width
jitter_width = 0.08

# Jittered scatter x values
np.random.seed(42)

# Generate jittered x values for each group
x_jittered_cpp = np.random.normal(loc=x_positions['Change point probability (z-score)'], scale=jitter_width, size=len(behav_cpp_coef))
x_jittered_conf = np.random.normal(loc=x_positions['Prior uncertainty (z-score)'], scale=jitter_width, size=len(behav_conf_coef))

# Add jittered scatter points — use numeric jittered x, actual y
fig.add_trace(go.Scatter(
    x=x_jittered_cpp,
    y=behav_cpp_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size=20),
    showlegend=False,
))

fig.add_trace(go.Scatter(
    x=x_jittered_conf,
    y=behav_conf_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size=20),
    showlegend=False,
))

fig.update_xaxes(showline=True, linewidth=3, linecolor='black')
fig.update_yaxes(showline=True, linewidth=3, linecolor='black',  title_text="Regression weight (participant)")

fig.update_xaxes(
    tickvals=[0, 1],
    ticktext=['Change point probability <br>(z-score)', 'Prior uncertainty <br>(z-score)'],
    showline=True, linewidth=3, linecolor='black'
)

# Add inset bars (model) - assigned to secondary x2/y2 axes
fig.add_trace(go.Bar(
    name='Model change point probability',
    x=['CPP'], y=[model_cpp_coef_mean],
    error_y=dict(type='data', array=[model_cpp_coef_sem]),
    showlegend=False,
    xaxis='x2',
    yaxis='y2',
    marker_color='rgba(255, 0, 0, 0.6)',
    marker_line_color='rgba(255, 0, 0, 0.8)',
    marker_line_width=1.0
))

fig.add_trace(go.Bar(
    name='Model prior uncertainty',
    x=['Prior conf'], y=[model_conf_coef_mean],
    error_y=dict(type='data', array=[model_conf_coef_sem]),
    showlegend=False,
    xaxis='x2',
    yaxis='y2',
    marker_color='rgba(255, 0, 0, 0.6)',
    marker_line_color='rgba(255, 0, 0, 0.8)',
    marker_line_width=1.0
))

fig.update_layout(
    xaxis=dict(
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
    ),
    yaxis=dict(
        ticks="inside",          # <-- turn ticks on
        tickwidth=1.5,            # thickness of ticks
        tickcolor='black',
        range=[-0.6, 0.6],
    ),
    xaxis2=dict(
        domain=[0.69, 0.99],
        anchor='y2',
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
        tickvals=[0, 1],
        ticktext=['CPP', 'Prior uncert'],
        tickfont=dict(size=20),
    ),
    yaxis2=dict(
        domain=[0.1, 0.3],
        anchor='x2',
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
        tickfont=dict(size=20),
        title_text="Regression weight <br> (model)",
        title_font=dict(size=22)
    ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=30, family="Times New Roman"),
    height=720,
    width=840)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_uncertainty_params_on_lr_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

#######################################################Scatter plot of relationship between uncertainty params coef weights

fig = go.Figure()

# Add scatter plot for individual regression weights in the second subplot
fig.add_trace(go.Scatter(
    x=behav_conf_coef['coefficient'], 
    y=behav_cpp_coef['coefficient'], 
    mode='markers', 
    marker=dict(color='rgba(0, 130, 150, 0.6)', size=24),
    showlegend=False
))

# Add horizontal and vertical lines for scatter plot
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))

# Update x and y axes for the second subplot (right)
fig.update_xaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Prior uncertainty (beta)", 
    range=[-0.6, 0.6],
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Change point probability (beta)",  
    range=[-0.6, 0.6],
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=30, family="Times New Roman"),
    height=720,
    width=840)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_uncertainty_v_cpp_regression_weights"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


######################################################Binned relationship between confidence and learning rate

behav_uncertainty_df = dataframes['computational_var_df'].copy()

# Splitting by prior uncertainty (defining the number of bins)
nbins_uncert_lr = 6 

# Evaluating the relationship between learning rate and prior uncertainty based on behavioural responses
_, bin_edges_conf_lr = pd.qcut(behav_uncertainty_df["prior_conf"], nbins_uncert_lr, retbins=True)

behav_uncertainty_df["prior_conf_bin"] = pd.cut(behav_uncertainty_df["prior_conf"], bins=bin_edges_conf_lr, labels=False, include_lowest=True)

avg_prior_conf_per_bin = behav_uncertainty_df.groupby('prior_conf_bin')['prior_conf'].mean()

behav_uncertainty_df = behav_uncertainty_df.merge(avg_prior_conf_per_bin, on='prior_conf_bin', suffixes=('', '_bin_avg'))

# Storing the average learning rate at the participant level
binned_data =  {"prior_conf_bin": [], "prior_conf_bin_label": [], "avg_lr": [], "participant": []}

for participant in behav_uncertainty_df["participant"].unique():
    participant_data = behav_uncertainty_df[behav_uncertainty_df["participant"] == participant]
    
    for prior_conf_bin in sorted(participant_data["prior_conf_bin"].unique()):
            
        bin_data = participant_data[
            (participant_data["prior_conf_bin"] == prior_conf_bin)]
        
        avg_lr = bin_data["cleaned_learning_rate_estimate"].mean()
        bin_label = bin_data['prior_conf_bin_avg'].unique()[0]  

        binned_data['prior_conf_bin_label'].append(bin_label)
        binned_data["prior_conf_bin"].append(prior_conf_bin)
        binned_data["avg_lr"].append(avg_lr)
        binned_data["participant"].append(participant)

participant_conf_lr_df = pd.DataFrame(binned_data)

group_avg_data = {"prior_conf_bin": [], "prior_conf_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

for bin in sorted(participant_conf_lr_df['prior_conf_bin'].unique()):
    data = participant_conf_lr_df[participant_conf_lr_df['prior_conf_bin'] == bin]
    learning_rate = data['avg_lr'].mean()
    lr_sem = stats.sem(data['avg_lr'])
    bin_label = data['prior_conf_bin_label'].unique()[0] 

    group_avg_data["prior_conf_bin"].append(bin)
    group_avg_data["prior_conf_bin_label"].append(bin_label)
    group_avg_data["group_avg_lr"].append(learning_rate)
    group_avg_data["group_sem_lr"].append(lr_sem)

behav_group_conf_lr_df = pd.DataFrame(group_avg_data)


fig = go.Figure()

fig.add_trace(go.Scatter(
            x=behav_group_conf_lr_df['prior_conf_bin_label'],
            y=behav_group_conf_lr_df['group_avg_lr'], 
            mode='markers', 
            showlegend=False,
            marker=dict(size=24, color='rgba(0, 130, 150, 0.8)'),
            error_y=dict(
                type='data',
                array=behav_group_conf_lr_df['group_sem_lr'],
                visible=True ),
            ))
        
fig.update_xaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Prior uncertainty (SD)",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    range=[0.041, max(0.02, behav_group_conf_lr_df['group_avg_lr'].max() + behav_group_conf_lr_df['group_sem_lr'].max())],
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Apparent learning rate (participant)",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=30, family="Times New Roman"),
    height=720,
    width=840)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_uncertainty_v_lr_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


###################### Binned relationship between CPP and apparent learning rate

# Evaluating the relationship between learning rate and change point probability based on behavioural responses
_, bin_edges_cpp_lr = pd.qcut(behav_uncertainty_df["cpp"], nbins_uncert_lr, retbins=True)

behav_uncertainty_df["cpp_bin"] = pd.cut(behav_uncertainty_df["cpp"], bins=bin_edges_cpp_lr, labels=False, include_lowest=True)

avg_cpp_per_bin = behav_uncertainty_df.groupby('cpp_bin')['cpp'].mean()

behav_uncertainty_df = behav_uncertainty_df.merge(avg_cpp_per_bin, on='cpp_bin', suffixes=('', '_bin_avg'))

# Storing the average learning rate at the participant level
binned_data =  {"cpp_bin": [], "cpp_bin_label": [], "avg_lr": [], "participant": []}

for participant in behav_uncertainty_df["participant"].unique():
    participant_data = behav_uncertainty_df[behav_uncertainty_df["participant"] == participant]
    
    for cpp_bin in sorted(participant_data["cpp_bin"].unique()):
            
        bin_data = participant_data[
            (participant_data["cpp_bin"] == cpp_bin)]
        
        avg_lr = bin_data["cleaned_learning_rate_estimate"].mean()
        bin_label = bin_data['cpp_bin_avg'].unique()[0]  

        binned_data['cpp_bin_label'].append(bin_label)
        binned_data["cpp_bin"].append(cpp_bin)
        binned_data["avg_lr"].append(avg_lr)
        binned_data["participant"].append(participant)

participant_cpp_lr_df = pd.DataFrame(binned_data)

group_avg_data = {"cpp_bin": [], "cpp_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

for bin in sorted(participant_cpp_lr_df['cpp_bin'].unique()):
    data = participant_cpp_lr_df[participant_cpp_lr_df['cpp_bin'] == bin]
    learning_rate = data['avg_lr'].mean()
    lr_sem = stats.sem(data['avg_lr'])
    bin_label = data['cpp_bin_label'].unique()[0] 

    group_avg_data["cpp_bin"].append(bin)
    group_avg_data["cpp_bin_label"].append(bin_label)
    group_avg_data["group_avg_lr"].append(learning_rate)
    group_avg_data["group_sem_lr"].append(lr_sem)

behav_group_cpp_lr_df = pd.DataFrame(group_avg_data)


fig = go.Figure()

fig.add_trace(go.Scatter(
            x=behav_group_cpp_lr_df['cpp_bin_label'],
            y=behav_group_cpp_lr_df['group_avg_lr'], 
            mode='markers', 
            showlegend=False,
            marker=dict(size=24, color='rgba(0, 130, 150, 0.8)'),
            error_y=dict(
                type='data',
                array=behav_group_cpp_lr_df['group_sem_lr'],
                visible=True ),
            ))
        
fig.update_xaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Change point probability",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text="Apparent learning rate (participant)",
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=30, family="Times New Roman"),
    height=720,
    width=840)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_cpp_v_lr_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()



####################################################### Realtionship between CPP and prior uncertainty around change points

computational_var_df = dataframes['computational_var_df'].copy()
change_point_indices_df = dataframes['change_point_indices_df'].copy()


computational_var_df = computational_var_df.sort_values(by=['participant', 'cum_outcome_num']).reset_index(drop=True)
change_point_indices_df = change_point_indices_df.sort_values(by=['participant', 'cum_outcome_num']).reset_index(drop=True)


averaged_cp_df = []
averaged_conf_df = []

for participant in dataframes['computational_var_df']['participant'].unique():
    print('Processing participant:', participant)

    variable_data = dataframes['computational_var_df'][dataframes['computational_var_df']['participant'] == participant]
    change_points = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant]

    _, averaged_values_cpp = variables_by_change_point(participant_str=participant,
                                                                        condition_str='01',
                                                                        directory_path=BASE_DIR,
                                                                        variable_data=variable_data,
                                                                        change_point_indices=change_points,
                                                                        parameter='cpp',
                                                                        pre=2,
                                                                        post=15,
                                                                        exclude_zero=False)
    _, averaged_values_conf = variables_by_change_point(participant_str=participant,
                                                                        condition_str='01',
                                                                        directory_path=BASE_DIR,
                                                                        variable_data=variable_data,
                                                                        change_point_indices=change_points,
                                                                        parameter='prior_conf',
                                                                        pre=2,
                                                                        post=15,
                                                                        exclude_zero=False)
    
    averaged_values_cpp['participant'] = participant
    averaged_cp_df.append(averaged_values_cpp)

    averaged_values_conf['participant'] = participant
    averaged_conf_df.append(averaged_values_conf)   

round_cp_averaged_cpp_df = pd.concat(averaged_cp_df, ignore_index=True)
round_cp_averaged_conf_df = pd.concat(averaged_conf_df, ignore_index=True)

round_cp_data = {
    'cpp': round_cp_averaged_cpp_df,
    'prior_conf': round_cp_averaged_conf_df
}

cpp_data_dict = {}
conf_data_dict = {}

for variable, df in round_cp_data.items():
    for pos in df['relative_position'].unique():
        pos_data = df[df['relative_position'] == pos]

        pos_data_mean = pos_data['mean'].mean()
        pos_data_sem = stats.sem(pos_data['mean'])

        if variable == 'cpp':
            cpp_data_dict[pos] = {
                'mean': pos_data_mean,
                'sem': pos_data_sem
            }
        elif variable == 'prior_conf':
            conf_data_dict[pos] = {
                'mean': pos_data_mean,
                'sem': pos_data_sem
            }

cpp_round_cp_df = pd.DataFrame.from_dict(cpp_data_dict, orient='index').reset_index()
cpp_round_cp_df.columns = ['position', 'mean', 'sem']
cpp_round_cp_df = cpp_round_cp_df.sort_values('position').reset_index(drop=True)

conf_round_cp_df = pd.DataFrame.from_dict(conf_data_dict, orient='index').reset_index()
conf_round_cp_df.columns = ['position', 'mean', 'sem']
conf_round_cp_df = conf_round_cp_df.sort_values('position').reset_index(drop=True)

# Prior uncertainty around change points
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.concat([conf_round_cp_df['position'], conf_round_cp_df['position'][::-1]]),
    y=pd.concat([
        conf_round_cp_df['mean'] - conf_round_cp_df['sem'],
        (conf_round_cp_df['mean'] + conf_round_cp_df['sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), 
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=conf_round_cp_df['position'],
    y=conf_round_cp_df['mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4),
    name='Prior uncertainty',
    showlegend=False
))

fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                  line=dict(dash="dot", width =3, color="black"), 
                  xref="x", yref="y domain")


# Update x-axes
fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=3,
    linecolor='black',
    tickmode='array',
    range=[-3, 16],
    tickvals=[i for i in range(0, 16, 2)],  # show only ticks >= 0
    ticktext=[str(i + 1) for i in range(0, 16, 2)],  # shift each tick label by +1
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black', 
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text='Prior confidence (SD)',
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

# Update layout
fig.update_layout(
    height=720, 
    width=840,
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman"), 
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_uncertainty_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


#################################################### Change point probability around change points
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.concat([cpp_round_cp_df['position'], cpp_round_cp_df['position'][::-1]]),
    y=pd.concat([
        cpp_round_cp_df['mean'] - cpp_round_cp_df['sem'],
        (cpp_round_cp_df['mean'] + cpp_round_cp_df['sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), 
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=cpp_round_cp_df['position'],
    y=cpp_round_cp_df['mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4),
    name='Change point probability',
    showlegend=False
))

fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                  line=dict(dash="dot", width=3, color="black"), 
                  xref="x", yref="y domain")


# Update x-axes
fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=3,
    linecolor='black',
    tickmode='array',
    range=[-3, 16],
    tickvals=[i for i in range(0, 16, 2)],  # show only ticks >= 0
    ticktext=[str(i + 1) for i in range(0, 16, 2)],  # shift each tick label by +1 
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

fig.update_yaxes(
    showline=True, 
    linewidth=3, 
    linecolor='black', 
    title_text='Change point probability',
    ticks="inside",
    tickwidth=1.5,
    tickcolor='black',
)

# Update layout
fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman"), 
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_cpp_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()