"""
This script plots figures for the analysis of the pupil signal for participants undertaking the Pupil task of the Adalearn project. The following figures are generated:
- The average baseline corrected epoched pupil signal
- The average non-baseline corrected epoched pupil signal
- Perstimulus regression coeficients for prior uncertainty and change point probability on the baseline correct pupil signal
- Perstimulus regression coeficients for prior uncertainty and change point probability on the non-baseline correct pupil signal
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
        'outcome_epoched_pupil_df', 
        'outcome_epoched_pupil_nobc_df', 
        'outcome_epochs_time_df', 
        'outcome_epochs_time_nobc_df', 
        'coefs_df', 
        'coefs_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc',
        'between_coef_permutation_analysis',
        'between_coef_permutation_analysis_nobc'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')


example_participant =  11 # Change this to the participant you want to plot

############################Average pupil response (baseline-corrected) plus example individual participant
baseline_corrected_epochs = dataframes['outcome_epoched_pupil_df']

baseline_corrected_epoch_times = dataframes['outcome_epochs_time_df']

average_pupil_df_list = []

for participant in baseline_corrected_epochs['participant'].unique():
    data = baseline_corrected_epochs[baseline_corrected_epochs['participant'] == participant]
    
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)

    pupil_std = np.std(data, axis=0)

    pupil_sem =  stats.sem(data, axis=0)

    time = baseline_corrected_epoch_times[baseline_corrected_epoch_times['participant']==participant]['time'].values

    average_pupil_df = pd.DataFrame({
        'pupil_signal': pupil_mean,
        'pupil_signal_std': pupil_std,
        'pupil_signal_sem': pupil_sem, 
        'time': time,
        'count': range(len(pupil_mean)),
        'participant': participant
    })

    average_pupil_df_list.append(average_pupil_df)

bc_group_avg_pupil_df = pd.concat(average_pupil_df_list, ignore_index=True)

# Plot the average pupil response for a specific participant
example_participant_data = bc_group_avg_pupil_df[bc_group_avg_pupil_df['participant'] == example_participant]

fig = go.Figure()

fig.add_trace(go.Scatter(x=example_participant_data['time'], y=example_participant_data['pupil_signal'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width=4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(example_participant_data['time']) + list(example_participant_data['time'][::-1]),
    y=list(example_participant_data['pupil_signal'] + example_participant_data['pupil_signal_sem']) + list((example_participant_data['pupil_signal'] - example_participant_data['pupil_signal_sem'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vrect(
    x0=-0.25,
    x1=0,
    fillcolor='rgba(210, 0, 150, 0.3)',
    opacity=0.5,
    layer="above",
    line_width=0,
    row=1,
    col=2
)

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))

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
        title_text = 'Pupil diameter signal (z-score)',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black', 
    )

fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman")
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-average_baseline_corrected_pupil_signal_example_participant"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


######################################################### Plot the group-level average pupil response

group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in bc_group_avg_pupil_df['count'].unique():
    data = bc_group_avg_pupil_df[bc_group_avg_pupil_df['count'] == count]

    mean_pupil = data['pupil_signal'].mean()
    sem_pupil = stats.sem(data['pupil_signal'])
    time = data['time'].mean()

    group_pupil_dict['mean_pupil'].append(mean_pupil)
    group_pupil_dict['sem_pupil'].append(sem_pupil)
    group_pupil_dict['time'].append(time)

bc_group_pupil_df = pd.DataFrame(group_pupil_dict)


fig = go.Figure()

fig.add_trace(go.Scatter(x=bc_group_pupil_df['time'], y=bc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width=4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(bc_group_pupil_df['time']) + list(bc_group_pupil_df['time'][::-1]),
    y=list(bc_group_pupil_df['mean_pupil'] + bc_group_pupil_df['sem_pupil']) + list((bc_group_pupil_df['mean_pupil'] - bc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vrect(
    x0=-0.25,
    x1=0,
    fillcolor='rgba(210, 0, 150, 0.3)',
    opacity=0.5,
    layer="above",
    line_width=0,
    row=1,
    col=2
)


fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))

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
        title_text = 'Pupil diameter signal (z-score)',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black', 
    )

fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman")
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_average_baseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

############################Average pupil response (nonbaseline-corrected)
nonbaseline_corrected_epochs = dataframes['outcome_epoched_pupil_nobc_df']

nonbaseline_corrected_epoch_times = dataframes['outcome_epochs_time_nobc_df']

average_pupil_df_list = []

for participant in nonbaseline_corrected_epochs['participant'].unique():
    data = nonbaseline_corrected_epochs[nonbaseline_corrected_epochs['participant'] == participant]
    
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)

    pupil_std = np.std(data, axis=0)

    pupil_sem =  stats.sem(data, axis=0)

    time = nonbaseline_corrected_epoch_times[nonbaseline_corrected_epoch_times['participant']==participant]['time'].values

    average_pupil_df = pd.DataFrame({
        'pupil_signal': pupil_mean,
        'pupil_signal_std': pupil_std,
        'pupil_signal_sem': pupil_sem, 
        'time': time,
        'count': range(len(pupil_mean)),
        'participant': participant
    })

    average_pupil_df_list.append(average_pupil_df)

nobc_group_avg_pupil_df = pd.concat(average_pupil_df_list, ignore_index=True)

# Plot the average pupil response for a specific participant
example_participant_data = nobc_group_avg_pupil_df[nobc_group_avg_pupil_df['participant'] == example_participant]

fig = go.Figure()

fig.add_trace(go.Scatter(x=example_participant_data['time'], y=example_participant_data['pupil_signal'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width = 4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(example_participant_data['time']) + list(example_participant_data['time'][::-1]),
    y=list(example_participant_data['pupil_signal'] + example_participant_data['pupil_signal_sem']) + list((example_participant_data['pupil_signal'] - example_participant_data['pupil_signal_sem'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))

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
        title_text = 'Pupil diameter signal (z-score)',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    )

fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman")
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-average_nonbaseline_corrected_pupil_signal_example_participant"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

####################################################### Plot the group-level average pupil response

group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in nobc_group_avg_pupil_df['count'].unique():
    data = nobc_group_avg_pupil_df[nobc_group_avg_pupil_df['count'] == count]

    mean_pupil = data['pupil_signal'].mean()
    sem_pupil = stats.sem(data['pupil_signal'])
    time = data['time'].mean()

    group_pupil_dict['mean_pupil'].append(mean_pupil)
    group_pupil_dict['sem_pupil'].append(sem_pupil)
    group_pupil_dict['time'].append(time)

nobc_group_pupil_df = pd.DataFrame(group_pupil_dict)


fig = go.Figure()

fig.add_trace(go.Scatter(x=nobc_group_pupil_df['time'], y=nobc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width=4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(nobc_group_pupil_df['time']) + list(nobc_group_pupil_df['time'][::-1]),
    y=list(nobc_group_pupil_df['mean_pupil'] + nobc_group_pupil_df['sem_pupil']) + list((nobc_group_pupil_df['mean_pupil'] - nobc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))

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
        title_text = 'Pupil diameter signal (z-score)',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black', 
    )

fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman")
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_average_non_baseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 
 

fig.show()

############################################################Peristimulus regression (baseline corrected)
baseline_corrected_coefs = dataframes['coefs_df']
baseline_corrected_clusters = dataframes['cluster_permutation_analysis']
between_coefs_clusters = dataframes['between_coef_permutation_analysis']


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=baseline_corrected_coefs['time'], 
    y=baseline_corrected_coefs['cpp'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)', width=4),  
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
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4), 
    name='Prior uncertainty'))

fig.add_trace(go.Scatter(
    x=list(baseline_corrected_coefs['time']) + list(baseline_corrected_coefs['time'][::-1]),
    y=list(baseline_corrected_coefs['prior_conf'] + baseline_corrected_coefs['prior_conf_error']) + list((baseline_corrected_coefs['prior_conf'] - baseline_corrected_coefs['prior_conf_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vrect(
    x0=-0.25,
    x1=0,
    fillcolor='rgba(210, 0, 150, 0.3)',
    opacity=0.5,
    layer="above",
    line_width=0,
    row=1,
    col=2
)

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))


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
            y=[style['yval']] * len(cluster_range), 
            mode='lines',
            line=dict(color=style['color'], width=12),
            showlegend=False,
            hoverinfo='skip'
        ))

# Add between-variable significant clusters
for _, row in between_coefs_clusters.iterrows():
    if row['p_value'] < 0.05:
        cluster_range = baseline_corrected_coefs['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y= [style['yval']-0.005] * len(cluster_range), 
            mode='lines',
            line=dict(color='black', width=12),
            showlegend=False,
            hoverinfo='skip'
        ))

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
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

####################################################################Peristimulus regression (nonbaseline-corrected)
nonbaseline_corrected_coefs = dataframes['coefs_nobc_df']
nonbaseline_corrected_clusters = dataframes['cluster_permutation_analysis_nobc']
nonbaseline_between_coefs_clusters = dataframes['between_coef_permutation_analysis_nobc']

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=nonbaseline_corrected_coefs['time'], 
    y=nonbaseline_corrected_coefs['cpp'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)', width=4),  
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
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4), 
    name='Prior uncertainty'))

fig.add_trace(go.Scatter(
    x=list(nonbaseline_corrected_coefs['time']) + list(nonbaseline_corrected_coefs['time'][::-1]),
    y=list(nonbaseline_corrected_coefs['prior_conf'] + nonbaseline_corrected_coefs['prior_conf_error']) + list((nonbaseline_corrected_coefs['prior_conf'] - nonbaseline_corrected_coefs['prior_conf_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))


fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))


var_style_map = {
    'cpp': {'color': 'rgba(220, 120, 0, 0.8)', 'yval': -0.05},
    'prior_conf': {'color': 'rgba(0, 130, 150, 0.8)', 'yval': -0.07}
}

# Add significant clusters with separation by variable
for _, row in nonbaseline_corrected_clusters.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05}) 

        cluster_range = nonbaseline_corrected_coefs['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=[style['yval']] * len(cluster_range),  
            mode='lines',
            line=dict(color=style['color'], width=12),
            showlegend=False,
            hoverinfo='skip'
        ))

# Add between-variable significant clusters
for _, row in nonbaseline_between_coefs_clusters.iterrows():
    if row['p_value'] < 0.05:
        cluster_range = nonbaseline_corrected_coefs['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y= [style['yval']-0.010] * len(cluster_range), 
            mode='lines',
            line=dict(color='black', width=12),
            showlegend=False,
            hoverinfo='skip'
        ))        

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
        x=1.0,
        y=0.3,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_non_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


####################################################### Phasic- and tonic-associated effects
fig = go.Figure()

fig.add_trace(go.Scatter(x=bc_group_pupil_df['time'], y=bc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width=4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(bc_group_pupil_df['time']) + list(bc_group_pupil_df['time'][::-1]),
    y=list(bc_group_pupil_df['mean_pupil'] + bc_group_pupil_df['sem_pupil']) + list((bc_group_pupil_df['mean_pupil'] - bc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
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

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))

fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))

for _, row in baseline_corrected_clusters.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})

        # Get cluster time range
        cluster_range = bc_group_pupil_df['time'].iloc[row['cluster_start']:row['cluster_end']+1]
        cluster_start = cluster_range.min()
        cluster_end = cluster_range.max()

        # Add shaded region ("square") under pupil trace
        fig.add_vrect(
            x0=cluster_start, 
            x1=cluster_end,
            fillcolor='rgba(128, 0, 128, 0.8)', 
            opacity=0.4,               
            layer="above",            
            line_width=0
        )

fig.add_annotation(
    text="Phasic effect",
    x=0.25, xref="x domain",  
    y=0.95, yref="y domain",
    showarrow=False,
    font=dict(size=30, color='rgba(128, 0, 128, 0.8)', family="Times New Roman")
)

fig.update_xaxes(
        title_text = "Time relative to outcome presentation (s)",
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black'
    )

fig.update_yaxes(
        showline=True, 
        linewidth=3, 
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
        title_text = 'Baseline-corrected pupil signal (z-score)',
    )

fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=30, family="Times New Roman"))


# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_regression_pupil_bc_effects_display"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

########################################################################################

fig = go.Figure()

fig.add_trace(go.Scatter(x=nobc_group_pupil_df['time'], y=nobc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 0, 0, 0.8)', width=4), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(nobc_group_pupil_df['time']) + list(nobc_group_pupil_df['time'][::-1]),
    y=list(nobc_group_pupil_df['mean_pupil'] + nobc_group_pupil_df['sem_pupil']) + list((nobc_group_pupil_df['mean_pupil'] - nobc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(150, 150, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=3))

fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=3))


time_vals = nobc_group_pupil_df['time'].values
cross_idx = np.where(np.diff(np.sign(time_vals)) > 0)[0]

if len(cross_idx) > 0:
    idx_cross = cross_idx[0]
    t1, t2 = time_vals[idx_cross], time_vals[idx_cross + 1]
    zero_time = t1 + (0 - t1) * (t2 - t1) / (t2 - t1)
else:
    zero_time = 0

for _, row in nonbaseline_corrected_clusters.iterrows():
    if row['p_value'] < 0.05:
        var = row['variable']
        style = var_style_map.get(var, {'color': 'black', 'yval': -0.05})

        cluster_start = nobc_group_pupil_df['time'].iloc[row['cluster_start']]

        fig.add_vrect(
            x0=cluster_start,
            x1=zero_time,
            fillcolor='rgba(255, 165, 0, 0.8)',
            opacity=0.5,
            layer="above",
            line_width=0
        )

fig.add_annotation(
    text="Tonic effect",
    x=0.18, xref="x domain",
    y=0.95, yref="y domain",
    showarrow=False,
    font=dict(size=30, color='rgba(255, 165, 0, 0.8)', family="Times New Roman")
)

fig.update_xaxes(
        title_text = "Time relative to outcome presentation (s)",
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black'
    )

fig.update_yaxes(
        showline=True, 
        linewidth=3, 
        linecolor='black',
        title_text = 'Nonbaseline-corrected pupil signal (z-score)',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black'
    )


fig.update_layout(
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=28, family="Times New Roman"))

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_regression_pupil_nobc_effects_display"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

