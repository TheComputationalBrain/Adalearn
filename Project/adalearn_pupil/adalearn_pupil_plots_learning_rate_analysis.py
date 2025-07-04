"""
This script plots figures for analysis of learning rate for participants undertaking the Pupil task in the Adalearn project.  The following figures are generated:
- The relationship between participants' and model learning rates
- The dynamics of learning rate around change points (with normative predictions)
- Regression coefficients of prior confidence and change point probability on learning rate
- Scatterplot of individual regression coefficients for prior confidence and change point probability
- Relationship between (binned) prior confidence and learning rate
- Dynamics of prior confidence and change point probability around change points
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir



dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
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

for participant in participant_v_model_df['participant'].unique():
    data = participant_v_model_df[participant_v_model_df['participant']==participant]

    fig.add_trace(go.Scatter(
        x=data['model_lr_bin_label'], 
        y=data['avg_lr'], 
        mode='lines+markers',
        line=dict(color='rgba(0, 130, 150, 0.12)', width=1.5),
        marker=dict(color='rgba(0, 130, 150, 0.15)', size=12),
        showlegend=False))

fig.add_trace(go.Scatter(
    x=group_behav_v_model_lr_df['model_lr_bin_label'],
    y=group_behav_v_model_lr_df['group_avg_lr'], 
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.8)', size=19), 
    showlegend=False, 
    ))

fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Normative learning rate", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Participant learning rate",   
)

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
    xaxis2=dict(
        domain=[0.075, 0.275],  
        anchor='y2',
        range=[0, 1.0],      
        title='Correlation coefficient',
        showline=True,
        linecolor='black'
    ),
    yaxis2=dict(
        domain=[0.8, 1.0],  
        anchor='x2',
        title='Count',
        showline=True,
        linecolor='black'
    )
)

fig.update_layout(
    xaxis=dict(
        title=dict(text="Model learning rate", font=dict(size=42)),
    ),
    yaxis=dict(
        title=dict(text="Participant learning rate", font=dict(size=42))
    ),
    xaxis2=dict(
        title=dict(text='Correlation coefficient', font=dict(size=36)),
        tickfont=dict(size=32),
    ),
    yaxis2=dict(
        title=dict(text='Count', font=dict(size=36)),
        tickfont=dict(size=32),
    ),
    font=dict(size=42, family="Times New Roman"), 
    height=1200,
    width=1800,
    template='simple_white',
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_behav_v_model_lr_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

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
    line=dict(color='rgba(0, 130, 150, 0.8)'),
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
    line=dict(color='black', width=2, dash='dash'),
)

# Plot significant clusters on behavioral subplot
for _, row in dataframes['lr_around_cp_sig_clusters'].iterrows():
    if row['p_value'] < 0.05:
        style = {'color': 'rgba(0, 0, 0, 1.0)', 'yval': 0.045}
        cluster_range = range(row['cluster_start'], row['cluster_end'] + 1)
        fig.add_trace(go.Scatter(
            x=list(cluster_range),
            y=[style['yval']] * len(cluster_range),
            mode='markers',
            marker=dict(color=style['color'], size=28, symbol='asterisk-open'),
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
    line=dict(color='red'),
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
    line=dict(color="black", width=2, dash="dash"),
    xref="x2", yref="y2"
)

# Layout
fig.update_layout(    
    xaxis=dict(
        range=[-3, 16],
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        title=dict(text="Outcome relative to change point", font=dict(size=42)),
        tickvals=[i for i in range(0, 16, 2)],  # show only ticks >= 0
        ticktext=[str(i + 1) for i in range(0, 16, 2)],  # shift each tick label by +1
    ),
    yaxis=dict(
        showline=True, 
        linewidth=1.5, 
        linecolor='black',
        title=dict(text="Learning rate (participant)", font=dict(size=42)),
        range=[0.041, max(0.01, lr_by_cp_avg_df_behav['mean'].max() + lr_by_cp_avg_df_behav['sem'].max())]
    ),
    xaxis2=dict(
        domain=[0.7, 1.0],
        anchor="y2",
        range=[-3, 16],
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickvals=[i for i in range(0, 16, 2)],  # show only ticks >= 0
        ticktext=[str(i + 1) for i in range(0, 16, 2)],  # shift each tick label by +1
        title=dict(text="Outcome relative to change point", font=dict(size=36)),
        tickfont=dict(size=32),
    ),
    yaxis2=dict(
        domain=[0.7, 1.0],
        anchor="x2",
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        dtick=0.02,
        title=dict(text="Learning rate (model)", font=dict(size=36)),
        tickfont=dict(size=32),
    ),
    font=dict(size=42, family="times new roman"),
    height=1200,
    width=1800,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lr_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

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
between_star_y = max(conf_star_y, cpp_star_y) - 0.1

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
    fig.add_shape(type='line',
                  x0=0, x1=1,
                  y0=between_star_y, y1=between_star_y,
                  line=dict(color='black', width=1.5))
    fig.add_annotation(
        text='*',
        x=0.5, y=between_star_y - 0.02,
        showarrow=False,
        font=dict(size=60)
    )

fig.update_traces(
    marker_color='rgba(0, 130, 150, 0.4)', 
    marker_line_color='rgba(0, 130, 150, 0.8)', 
    marker_line_width=1.0,
)

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
    showlegend=False,
))

fig.add_trace(go.Scatter(
    x=x_jittered_cpp,
    y=behav_cpp_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size=16),
    showlegend=False,
))

fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black',  title_text="Regression weight (participant)")

fig.update_xaxes(
    tickvals=[0, 1],
    ticktext=['Prior confidence (z-score)', 'Change point probability (z-score)'],
    showline=True, linewidth=1.5, linecolor='black'
)

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
    layer="above"
)

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
        domain=[0.7, 0.95],  # inset y position (change to resize/relocate)
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
    width=1800)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_uncertainty_params_on_lr_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()

#######################################################Scatter plot of relationship between uncertainty params coef weights

fig = go.Figure()

# Add scatter plot for individual regression weights in the second subplot
fig.add_trace(go.Scatter(
    x=behav_conf_coef['coefficient'], 
    y=behav_cpp_coef['coefficient'], 
    mode='markers', 
    marker=dict(color='rgba(0, 130, 150, 0.6)', size=19),
    showlegend=False
))

# Add horizontal and vertical lines for scatter plot
fig.add_hline(y=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))

# Update x and y axes for the second subplot (right)
fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Prior confidence (beta)", 
    range=[-0.6, 0.6]
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Change point probability (beta)",  
    range=[-0.6, 0.6]
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=42, family="Times New Roman"),
    height=1200,
    width=1800)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_conf_v_cpp_regression_weights"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


######################################################Binned relationship between confidence and learning rate

behav_uncertainty_df = dataframes['computational_var_df'].copy()

# Splitting by prior confidence (defining the number of bins)
nbins_uncert_lr = 6 

# Evaluating the relationship between learning rate and prior confidence based on behavioural responses
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
            mode='lines+markers', 
            showlegend=False, 
            line = dict(color = 'teal', width=2),
            marker=dict(size=19),
            error_y=dict(
                type='data',
                array=behav_group_conf_lr_df['group_sem_lr'],
                visible=True ),
            ))
        
fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Prior confidence (-log(sd))"
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Learning rate (participant)"
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=42, family="Times New Roman"),
    height=1200,
    width=1800)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_conf_v_lr_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


####################################################### Realtionship between CPP and prior confidence around change points

computational_var_df = dataframes['computational_var_df'].copy()
change_point_indices_df = dataframes['change_point_indices_df'].copy()


computational_var_df = computational_var_df.sort_values(by=['participant', 'cum_outcome_num']).reset_index(drop=True)
change_point_indices_df = change_point_indices_df.sort_values(by=['participant', 'cum_outcome_num']).reset_index(drop=True)

before_window = 5
after_window = 5
window_len = before_window + after_window + 1

all_window_dfs = []

for participant in computational_var_df['participant'].unique():
    participant_data = computational_var_df[computational_var_df['participant'] == participant].reset_index(drop=True)

    cp_data = change_point_indices_df[change_point_indices_df['participant'] == participant]

    print(f'For participant {participant} there are {len(cp_data)} change points.')

    for _, cp_row in cp_data.iterrows():
        cp = cp_row['cum_outcome_num']
        session = cp_row['task_session']

        session_data = participant_data[participant_data['task_session'] == session].reset_index(drop=True)

        try:
            cp_index = session_data[session_data['cum_outcome_num'] == cp].index[0]
        except IndexError:
            print(f"Could not find cp {cp} in participant {participant}, session {session}")
            continue

        start_idx = max(0, cp_index - before_window)
        end_idx = min(len(session_data), cp_index + after_window + 1)

        window = session_data.iloc[start_idx:end_idx]

        pad_before = max(0, before_window - cp_index)
        pad_after = max(0, (cp_index + after_window + 1) - len(session_data))

        # Values to pad
        conf_vals = window['prior_conf'].values
        cpp_vals = window['cpp'].values

        conf_padded = np.pad(conf_vals, (pad_before, pad_after), mode='constant', constant_values=np.nan)
        cpp_padded = np.pad(cpp_vals, (pad_before, pad_after), mode='constant', constant_values=np.nan)

        relative_positions = list(range(-before_window, after_window + 1))

        # Build DataFrame
        window_df = pd.DataFrame({
            'participant': participant,
            'task_session': session,
            'cp_index': cp,
            'relative_position': relative_positions,
            'prior_conf': conf_padded,
            'cpp': cpp_padded
        })

        all_window_dfs.append(window_df)

# Combine all windows
all_windows_df = pd.concat(all_window_dfs, ignore_index=True)


rel_pos_dict = {'participant': [], 'rel_pos': [], 'mean_conf': [], 'mean_cpp': []}

for participant in all_windows_df['participant'].unique():
    data = all_windows_df[all_windows_df['participant'] == participant]
    for index in data['relative_position'].unique():
        rel_data = data[data['relative_position'] == index]

        mean_conf = rel_data['prior_conf'].mean()
        mean_cpp = rel_data['cpp'].mean()

        rel_pos_dict['participant'].append(participant)
        rel_pos_dict['rel_pos'].append(index)
        rel_pos_dict['mean_conf'].append(mean_conf)
        rel_pos_dict['mean_cpp'].append(mean_cpp)

rel_pos_df = pd.DataFrame(rel_pos_dict)


grouped_rel_pos_dict = {'rel_pos': [], 'conf_mean': [], 'conf_std': [], 'conf_sem': [],  'cpp_mean': [], 'cpp_std': [], 'cpp_sem': []} 

for i in rel_pos_df['rel_pos'].unique():
    data = rel_pos_df[rel_pos_df['rel_pos'] == i]
   
    conf_mean = data['mean_conf'].mean()
    conf_std = data['mean_conf'].std()
    conf_sem = stats.sem(data['mean_conf'])
    
    cpp_mean = data['mean_cpp'].mean()
    cpp_std = data['mean_cpp'].std()
    cpp_sem = stats.sem(data['mean_cpp'])

    grouped_rel_pos_dict['rel_pos'].append(i)
    grouped_rel_pos_dict['conf_mean'].append(conf_mean)
    grouped_rel_pos_dict['conf_std'].append(conf_std)
    grouped_rel_pos_dict['conf_sem'].append(conf_sem)
    grouped_rel_pos_dict['cpp_mean'].append(cpp_mean)
    grouped_rel_pos_dict['cpp_std'].append(cpp_std)
    grouped_rel_pos_dict['cpp_sem'].append(cpp_sem)


grouped_rel_pos_df = pd.DataFrame(grouped_rel_pos_dict)


fig = make_subplots(rows=2, cols=1, shared_xaxes=False)

# Prior confidence
fig.add_trace(go.Scatter(
    x=pd.concat([grouped_rel_pos_df['rel_pos'], grouped_rel_pos_df['rel_pos'][::-1]]),
    y=pd.concat([
        grouped_rel_pos_df['conf_mean'] - grouped_rel_pos_df['conf_sem'],
        (grouped_rel_pos_df['conf_mean'] + grouped_rel_pos_df['conf_sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), 
    hoverinfo="skip",
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=grouped_rel_pos_df['rel_pos'],
    y=grouped_rel_pos_df['conf_mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=3),
    name='Prior Confidence'
), row=1, col=1)


# Change point probability
fig.add_trace(go.Scatter(
    x=pd.concat([grouped_rel_pos_df['rel_pos'], grouped_rel_pos_df['rel_pos'][::-1]]),
    y=pd.concat([
        grouped_rel_pos_df['cpp_mean'] - grouped_rel_pos_df['cpp_sem'],
        (grouped_rel_pos_df['cpp_mean'] + grouped_rel_pos_df['cpp_sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=grouped_rel_pos_df['rel_pos'],
    y=grouped_rel_pos_df['cpp_mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=3),
    name='CPP'
), row=2, col=1)


# Add vertical and horizontal lines for all subplots
for i in range(1, 3):  # Rows 1 to 3
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                  line=dict(dash="dot", color="black"), 
                  xref="x", yref="y domain", row=i, col=1)
    

# Update x-axes
for i in range(1, 3):
    fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=1,
    linecolor='black',
    tickmode='array',
    tickvals=list(range(-5, 6)),
    ticktext=["", "", "", "", "", "(change point)", "", "", "", "", ""],
    tickangle=0,
    range=[-5.5, 6.5], 
    row=i, col=1
)

# Update y-axes
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Prior confidence', row=1, col=1)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Change point probability', row=2, col=1)


# Update layout
fig.update_layout(height=2*1200, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"), showlegend=False)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_prior_conf_and_cpp_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


fig.show()

   