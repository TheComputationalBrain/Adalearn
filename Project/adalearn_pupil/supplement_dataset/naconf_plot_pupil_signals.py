"""
This script plots the average pupil signals obtained from the NACONF study. The following plots are produced:
- Average baseline correct pupil signal
- Average non-baseline correct pupil signal
"""

import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
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
        'outcome_epoched_pupil_df', 
        'outcome_epoched_pupil_nobc_df', 
        'outcome_epochs_time_df', 
        'outcome_epochs_time_nobc_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil_NACONF-')

#####################################################################Baseline-corrected average

baseline_corrected_epochs = dataframes['outcome_epoched_pupil_df']

baseline_corrected_epoch_times = dataframes['outcome_epochs_time_df']

average_pupil_df_list = []

for participant in baseline_corrected_epochs['participant'].unique():
    print(f'analysing participant: {participant}')
    
    data = baseline_corrected_epochs[baseline_corrected_epochs['participant'] == participant]
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)
    pupil_std = np.std(data, axis=0)  
    pupil_sem = stats.sem(data, axis=0)  

    time = baseline_corrected_epoch_times[baseline_corrected_epoch_times['participant']==participant]['time'].values

    n_time = len(time)
    n_mean = len(pupil_mean)

    if n_mean > n_time:
        print(f' - Pupil signals longer than time for participant {participant}')
        print(f'   Last pupil_mean value: {pupil_mean[-1]}')
        print(f'   Last pupil_std value: {pupil_std[-1]}')
        print(f'   Last pupil_sem value: {pupil_sem[-1]}')
        pupil_mean = pupil_mean[:n_time]
        pupil_std = pupil_std[:n_time]
        pupil_sem = pupil_sem[:n_time]

    if n_time > n_mean:
        print(f' - Time longer than pupil signals for participant {participant}')
        print(f'   Last time value: {time[-1]}')
        time = time[:n_mean]

    # Now lengths match
    print(f'Lengths after trimming - pupil_mean: {len(pupil_mean)}, time: {len(time)}\n')

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

# All individual baseline corrected pupil responses on the same plot
fig = go.Figure()

for participant in bc_group_avg_pupil_df['participant'].unique():
    participant_data = bc_group_avg_pupil_df[bc_group_avg_pupil_df['participant'] == participant]

    fig.add_trace(go.Scatter(x=participant_data['time'], y=participant_data['pupil_signal'], mode='lines', line=dict(color='rgba(0, 130, 150, 0.4)'), showlegend=False))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))

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
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=1200, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"))

# Save the figure to the group-level folder
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-NACONF_baseline_corrected_pupil_signal_all_participant_traces"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show() 




fig = go.Figure()

fig.add_trace(go.Scatter(x=bc_group_pupil_df['time'], y=bc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 130, 150, 0.8)'), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(bc_group_pupil_df['time']) + list(bc_group_pupil_df['time'][::-1]),
    y=list(bc_group_pupil_df['mean_pupil'] + bc_group_pupil_df['sem_pupil']) + list((bc_group_pupil_df['mean_pupil'] - bc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))

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
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=1200, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"))

# Save the figure to the group-level folder
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-NACONF_average_baseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


#####################################################################nonbaseline-corrected average

nonbaseline_corrected_epochs = dataframes['outcome_epoched_pupil_nobc_df']

nonbaseline_corrected_epoch_times = dataframes['outcome_epochs_time_nobc_df']

nobc_average_pupil_df_list = []

for participant in nonbaseline_corrected_epochs['participant'].unique():
    print(f'analysing participant: {participant}')
    
    data = nonbaseline_corrected_epochs[nonbaseline_corrected_epochs['participant'] == participant]
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)
    pupil_std = np.std(data, axis=0)  
    pupil_sem = stats.sem(data, axis=0)  

    time = nonbaseline_corrected_epoch_times[nonbaseline_corrected_epoch_times['participant']==participant]['time'].values

    n_time = len(time)
    n_mean = len(pupil_mean)

    if n_mean > n_time:
        print(f' - Pupil signals longer than time for participant {participant}')
        print(f'   Last pupil_mean value: {pupil_mean[-1]}')
        print(f'   Last pupil_std value: {pupil_std[-1]}')
        print(f'   Last pupil_sem value: {pupil_sem[-1]}')
        pupil_mean = pupil_mean[:n_time]
        pupil_std = pupil_std[:n_time]
        pupil_sem = pupil_sem[:n_time]

    if n_time > n_mean:
        print(f' - Time longer than pupil signals for participant {participant}')
        print(f'   Last time value: {time[-1]}')
        time = time[:n_mean]

    # Now lengths match
    print(f'Lengths after trimming - pupil_mean: {len(pupil_mean)}, time: {len(time)}\n')

    average_pupil_df = pd.DataFrame({
        'pupil_signal': pupil_mean,
        'pupil_signal_std': pupil_std,
        'pupil_signal_sem': pupil_sem, 
        'time': time,
        'count': range(len(pupil_mean)),
        'participant': participant
    })

    nobc_average_pupil_df_list.append(average_pupil_df)

nobc_group_avg_pupil_df = pd.concat(average_pupil_df_list, ignore_index=True)


nobc_group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in nobc_group_avg_pupil_df['count'].unique():
    data = nobc_group_avg_pupil_df[nobc_group_avg_pupil_df['count'] == count]

    mean_pupil = data['pupil_signal'].mean()
    sem_pupil = stats.sem(data['pupil_signal'])
    time = data['time'].mean()

    nobc_group_pupil_dict['mean_pupil'].append(mean_pupil)
    nobc_group_pupil_dict['sem_pupil'].append(sem_pupil)
    nobc_group_pupil_dict['time'].append(time)

nobc_group_pupil_df = pd.DataFrame(group_pupil_dict)

# All individual baseline corrected pupil responses on the same plot
fig = go.Figure()

for participant in nobc_group_avg_pupil_df['participant'].unique():
    participant_data = nobc_group_avg_pupil_df[nobc_group_avg_pupil_df['participant'] == participant]

    fig.add_trace(go.Scatter(x=participant_data['time'], y=participant_data['pupil_signal'], mode='lines', line=dict(color='rgba(0, 130, 150, 0.4)'), showlegend=False))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))

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
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=1200, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"))

# Save the figure to the group-level folder
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-NACONF_nonbaseline_corrected_pupil_signal_all_participant_traces"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show() 




fig = go.Figure()

fig.add_trace(go.Scatter(x=nobc_group_pupil_df['time'], y=nobc_group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 130, 150, 0.8)'), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(nobc_group_pupil_df['time']) + list(nobc_group_pupil_df['time'][::-1]),
    y=list(nobc_group_pupil_df['mean_pupil'] + nobc_group_pupil_df['sem_pupil']) + list((nobc_group_pupil_df['mean_pupil'] - nobc_group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot', line=dict(color='black', width=1.5))
fig.add_hline(y=0, line_dash = 'dot', line=dict(color='black', width=1.5))

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
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=1200, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"))

# Save the figure to the group-level folder
plot_dir = os.path.join(NACONF_BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-NACONF_average_nonbaseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()