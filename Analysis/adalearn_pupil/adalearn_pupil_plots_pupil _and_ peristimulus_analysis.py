"""

"""
import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_group_level_functions_data_utilities import load_sample_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, base_dir, condition_str


dataframes = load_sample_csvs(directory_path=base_dir, substrings=['epoched_pupil_df', 'epoched_pupil_nobc_df', 'outcome_epochs_descriptive_stats_df', 'outcome_epochs_descriptive_stats_nobc_df', 'coefs_df', 'coefs_nobc_df', 'cluster_permutation_analysis', 'cluster_permutation_analysis_nobc'])


############################Average pupil response (baseline-corrected)
baseline_corrected_epochs = dataframes['epoched_pupil_df']

baseline_corrected_descriptives = dataframes['outcome_epochs_descriptive_stats_df']

average_pupil_df_list = []

for participant in baseline_corrected_epochs['participant'].unique():
    data = baseline_corrected_epochs[baseline_corrected_epochs['participant'] == participant]
    
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)

    pupil_std = np.std(data, axis=0)

    pupil_sem =  stats.sem(data, axis=0)

    time = baseline_corrected_descriptives[baseline_corrected_descriptives['participant']==participant]['time'].values

    average_pupil_df = pd.DataFrame({
        'pupil_signal': pupil_mean,
        'pupil_signal_std': pupil_std,
        'pupil_signal_sem': pupil_sem, 
        'time': time,
        'count': range(len(pupil_mean)),
        'participant': participant
    })

    average_pupil_df_list.append(average_pupil_df)

group_avg_pupil_df = pd.concat(average_pupil_df_list, ignore_index=True)


group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in group_avg_pupil_df['count'].unique():
    data = group_avg_pupil_df[group_avg_pupil_df['count'] == count]

    mean_pupil = data['pupil_signal'].mean()
    sem_pupil = stats.sem(data['pupil_signal'])
    time = data['time'].mean()

    group_pupil_dict['mean_pupil'].append(mean_pupil)
    group_pupil_dict['sem_pupil'].append(sem_pupil)
    group_pupil_dict['time'].append(time)

group_pupil_df = pd.DataFrame(group_pupil_dict)


fig = go.Figure()

fig.add_trace(go.Scatter(x=group_pupil_df['time'], y=group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 130, 150, 0.8)'), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(group_pupil_df['time']) + list(group_pupil_df['time'][::-1]),
    y=list(group_pupil_df['mean_pupil'] + group_pupil_df['sem_pupil']) + list((group_pupil_df['mean_pupil'] - group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')

fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
    )

fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black',
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=950, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=20))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_average_baseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()




############################Average pupil response (nonbaseline-corrected)
nonbaseline_corrected_epochs = dataframes['epoched_pupil_nobc_df']

nonbaseline_corrected_descriptives = dataframes['outcome_epochs_descriptive_stats_nobc_df']

average_pupil_df_list = []

for participant in nonbaseline_corrected_epochs['participant'].unique():
    data = nonbaseline_corrected_epochs[nonbaseline_corrected_epochs['participant'] == participant]
    
    data = data.drop(columns='participant')

    pupil_mean = np.mean(data, axis=0)

    pupil_std = np.std(data, axis=0)

    pupil_sem =  stats.sem(data, axis=0)

    time = nonbaseline_corrected_descriptives[nonbaseline_corrected_descriptives['participant']==participant]['time'].values

    average_pupil_df = pd.DataFrame({
        'pupil_signal': pupil_mean,
        'pupil_signal_std': pupil_std,
        'pupil_signal_sem': pupil_sem, 
        'time': time,
        'count': range(len(pupil_mean)),
        'participant': participant
    })

    average_pupil_df_list.append(average_pupil_df)

group_avg_pupil_df = pd.concat(average_pupil_df_list, ignore_index=True)


group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in group_avg_pupil_df['count'].unique():
    data = group_avg_pupil_df[group_avg_pupil_df['count'] == count]

    mean_pupil = data['pupil_signal'].mean()
    sem_pupil = stats.sem(data['pupil_signal'])
    time = data['time'].mean()

    group_pupil_dict['mean_pupil'].append(mean_pupil)
    group_pupil_dict['sem_pupil'].append(sem_pupil)
    group_pupil_dict['time'].append(time)

group_pupil_df = pd.DataFrame(group_pupil_dict)


fig = go.Figure()

fig.add_trace(go.Scatter(x=group_pupil_df['time'], y=group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 130, 150, 0.8)'), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(group_pupil_df['time']) + list(group_pupil_df['time'][::-1]),
    y=list(group_pupil_df['mean_pupil'] + group_pupil_df['sem_pupil']) + list((group_pupil_df['mean_pupil'] - group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')

fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
    )

fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black',
        title_text = 'Pupil diameter signal (z-score)' 
    )

fig.update_layout(height=950, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=20))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_average_non_baseline_corrected_pupil_signal"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 
 

fig.show()


############################################################Peristimulus regression (baseline corrected)

baseline_corrected_coefs = dataframes['coefs_df']
baseline_corrected_clusters = dataframes['cluster_permutation_analysis']


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


fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')


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
            mode='markers',
            marker=dict(color=style['color'], size=8, symbol='asterisk-open'),
            showlegend=False,
            hoverinfo='skip'
        ))

fig.update_xaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Regression weights",  
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800,
    font=dict(size=20),
    legend=dict(
        x=0.32,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


fig.show()



####################################################################Peristimulus regression (nonbaseline-corrected)

nonbaseline_corrected_coefs = dataframes['coefs_nobc_df']
nonbaseline_corrected_clusters = dataframes['cluster_permutation_analysis_nobc']



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


fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')


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
            mode='markers',
            marker=dict(color=style['color'], size=8, symbol='asterisk-open'),
            showlegend=False,
            hoverinfo='skip'
        ))
        

fig.update_xaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Regression weights",  
)

fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800,
    font=dict(size=20),
    legend=dict(
        x=0.18,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_non_baseline_corrected_peristimulus_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


fig.show()



#########################################################################

