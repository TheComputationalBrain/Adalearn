import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


from adalearn_pupil.supplement_dataset.naconf_ideal_observer_model_functions import load_task_behavioural_data, load_constants, change_point_analysis, compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations

from adalearn_pupil.supplement_dataset.naconf_pupil_functions import load_and_process_pupil_data, all_outcome_pupil_epochs

from adalearn_pupil.supplement_dataset.naconf_pupil_regression_functions import pupil_peristimulus_regression_analysis

from adalearn_pupil.supplement_dataset.naconf_group_level_analysis_functions import compute_group_level_permutation_cluster_test

from adalearn_pupil.supplement_dataset.naconf_participant_list import usable_participants

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import base_dir, list_subject_selected, task_sessions, from_session, to_session


task_session_df_list = []
regression_coefs_list = []
regression_nobc_coefs_list = []

for participant in usable_participants:
    print(f"\n\n\nProcessing participant {participant}...")
    # Define the directory path for the participant
    directory_path = Path(f'{base_dir}/sub-{participant}')

    print(f"\nLoading task data and computing model derived variables...")
    task_session_df = load_task_behavioural_data(directory_path=directory_path, selected_participant=participant, task_sessions=task_sessions)

    num_change_points, change_points_per_session, change_point_indices, task_session_df = change_point_analysis(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df)

    constants, P_C, P_MIN, P_MAX = load_constants(directory_path=directory_path, task_sessions=task_sessions)

    task_session_df = compute_surprise_from_observations(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df, change_point_indices_df=change_point_indices, p_c=P_C, p_min=P_MIN, p_max=P_MAX, plot_surprise = False)

    task_session_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df, change_point_indices_df=change_point_indices, p_c=P_C, p_min=P_MIN, p_max=P_MAX, estimate_type = 'mean', conf_type = 'prev-conf', unit = 'confidence', plot_prior_conf = False, plot_cpp = False)

    task_session_df_list.append(task_session_df)

    print(f"\nLoading, preprocessing and epoching pupil data...")
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant, list_subject_selected=list_subject_selected, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=2.0, baseline_cor=[-0.25, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 

    outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant, list_subject_selected=list_subject_selected, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.25, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc, outcome_epochs_to_remove_df_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant, list_subject_selected=list_subject_selected, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    print(f"\nRunning peristimulus regression analysis on baseline corrected pupil signal...")
    coefficients_df, pupil_regress_df, residuals_df = pupil_peristimulus_regression_analysis(participant_str = participant, directory_path = directory_path, task_session_df = task_session_df, outcome_epoched_pupil_df = outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove_df, variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = True, plot_regression_analysis=False)

    regression_coefs_list.append(coefficients_df)

    print(f"\nRunning peristimulus regression analysis on non-baseline corrected pupil signal...")
    coefficients_df_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_peristimulus_regression_analysis(participant_str = participant, directory_path = directory_path, task_session_df = task_session_df, outcome_epoched_pupil_df = outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_df_nobc, variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = False, plot_regression_analysis=False)

    regression_nobc_coefs_list.append(coefficients_df_nobc)

# Concatenate the dataframes in the lists into a single group-level dataframe
group_level_task_session_df = pd.concat(task_session_df_list, ignore_index=True)
group_level_regression_coefs_df = pd.concat(regression_coefs_list, ignore_index=True)
group_level_regression_nobc_coefs_df = pd.concat(regression_nobc_coefs_list, ignore_index=True)



# Analysis of baseline corrected pupil signal --------------------
# Create dictionary for plotting
baseline_corr_coefs_dict = {'cpp': [], 'cpp_error': [], 'conf': [], 'conf_error': [], 'time': [], 'count': []}

for count in group_level_regression_coefs_df['count'].unique():
    data = group_level_regression_coefs_df[group_level_regression_coefs_df['count'] == count].sort_values(by='participant')
    mean_cpp = data['cpp_zscore'].mean()
    sem_cpp = stats.sem(data['cpp_zscore'])
    mean_conf = data['prior_conf_zscore'].mean()
    sem_conf = stats.sem(data['prior_conf_zscore'])
    mean_time = data['time'].mean()
    i_count = count

    baseline_corr_coefs_dict['cpp'].append(mean_cpp)
    baseline_corr_coefs_dict['cpp_error'].append(sem_cpp)
    baseline_corr_coefs_dict['conf'].append(mean_conf)
    baseline_corr_coefs_dict['conf_error'].append(sem_conf)
    baseline_corr_coefs_dict['time'].append(mean_time)
    baseline_corr_coefs_dict['count'].append(i_count)

baseline_corr_coefs_df = pd.DataFrame(baseline_corr_coefs_dict)


baseline_corr_cluster_analysis_results = compute_group_level_permutation_cluster_test(group_level_regression_coefs_df, vars=['prior_conf_zscore', 'cpp_zscore'])


fig = go.Figure()

# Slice your data once to keep things consistent
time = baseline_corr_coefs_df['time'][:-1]
conf = baseline_corr_coefs_df['conf'][:-1]
conf_error = baseline_corr_coefs_df['conf_error'][:-1]
cpp = baseline_corr_coefs_df['cpp'][:-1]
cpp_error = baseline_corr_coefs_df['cpp_error'][:-1]

# Line trace
fig.add_trace(go.Scatter(
    x=time,
    y=cpp,
    mode='lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),
    name='Change point probability'))

# Error band trace
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(cpp + cpp_error) + list((cpp - cpp_error)[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False))

cpp_cluster_slice = baseline_corr_cluster_analysis_results['cpp_zscore']['clusters'][0][0]
cpp_cluster_indices = list(range(cpp_cluster_slice.start, cpp_cluster_slice.stop))  
cpp_significant_times = baseline_corr_coefs_df['time'].iloc[cpp_cluster_indices]

# Add dots for significant clusters (at the time points of the cluster)
fig.add_trace(go.Scatter(
    x=cpp_significant_times,  # Time points where the cluster is significant
    y=[-0.02] * len(cpp_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
    mode='markers',
    marker=dict(color='rgba(220, 120, 0, 0.6)', size=8, symbol='asterisk-open'),
    name='Significant Cluster - CPP',
    showlegend=False
))

# Line trace
fig.add_trace(go.Scatter(
    x=time,
    y=conf,
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'),
    name='Prior confidence'))

# Error band trace
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(conf +conf_error) + list((conf - conf_error)[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False))

conf_cluster_slice = baseline_corr_cluster_analysis_results['prior_conf_zscore']['clusters'][0][0]
conf_cluster_indices = list(range(conf_cluster_slice.start, conf_cluster_slice.stop))  
conf_significant_times = baseline_corr_coefs_df['time'].iloc[conf_cluster_indices]

# Add dots for significant clusters (at the time points of the cluster)
fig.add_trace(go.Scatter(
    x=conf_significant_times,  # Time points where the cluster is significant
    y=[-0.025] * len(conf_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.6)', size=6),
    name='Significant Cluster - Prior Confidence',
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
    title_text="Regression weights",  
)

fig.update_layout(
    title=dict(
            text="Peristimulus regression weights for prior confidence and change point probability on the baseline corrected pupil signal"
        ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800,
    font=dict(size=18),
    legend=dict(
        x=0.35,
        y=0.9,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))


fig.show()


# Analysis of non-baseline corrected pupil signal --------------------
# Create dictionary for plotting
nonbaseline_corr_coefs_dict = {'cpp': [], 'cpp_error': [], 'conf': [], 'conf_error': [], 'time': [], 'count': []}

for count in group_level_regression_nobc_coefs_df['count'].unique():
    data = group_level_regression_nobc_coefs_df[group_level_regression_nobc_coefs_df['count'] == count].sort_values(by='participant')
    mean_cpp = data['cpp_zscore'].mean()
    sem_cpp = stats.sem(data['cpp_zscore'])
    mean_conf = data['prior_conf_zscore'].mean()
    sem_conf = stats.sem(data['prior_conf_zscore'])
    mean_time = data['time'].mean()
    i_count = count

    nonbaseline_corr_coefs_dict['cpp'].append(mean_cpp)
    nonbaseline_corr_coefs_dict['cpp_error'].append(sem_cpp)
    nonbaseline_corr_coefs_dict['conf'].append(mean_conf)
    nonbaseline_corr_coefs_dict['conf_error'].append(sem_conf)
    nonbaseline_corr_coefs_dict['time'].append(mean_time)
    nonbaseline_corr_coefs_dict['count'].append(i_count)

nonbaseline_corr_coefs_df = pd.DataFrame(nonbaseline_corr_coefs_dict)


nonbaseline_corr_cluster_analysis_results = compute_group_level_permutation_cluster_test(group_level_regression_nobc_coefs_df, vars=['prior_conf_zscore', 'cpp_zscore'])


fig = go.Figure()

# Slice your data once to keep things consistent
time = nonbaseline_corr_coefs_df['time'][:-1]
conf = nonbaseline_corr_coefs_df['conf'][:-1]
conf_error = nonbaseline_corr_coefs_df['conf_error'][:-1]
cpp = nonbaseline_corr_coefs_df['cpp'][:-1]
cpp_error = nonbaseline_corr_coefs_df['cpp_error'][:-1]

# Line trace
fig.add_trace(go.Scatter(
    x=time,
    y=cpp,
    mode='lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),
    name='Change point probability'))

# Error band trace
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(cpp + cpp_error) + list((cpp - cpp_error)[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False))


cpp_nobc_cluster_slice = nonbaseline_corr_cluster_analysis_results['cpp_zscore']['clusters'][0][0]
cpp_nobc_cluster_indices = list(range(cpp_nobc_cluster_slice.start, cpp_nobc_cluster_slice.stop))  
cpp_nobc_significant_times = nonbaseline_corr_coefs_df['time'].iloc[cpp_nobc_cluster_indices]

# Add dots for significant clusters (at the time points of the cluster)
fig.add_trace(go.Scatter(
    x=cpp_nobc_significant_times,  # Time points where the cluster is significant
    y=[-0.1] * len(cpp_nobc_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
    mode='markers',
    marker=dict(color='rgba(220, 120, 0, 0.6)', size=8, symbol='asterisk-open'),
    name='Significant Cluster - CPP',
    showlegend=False
))

# Line trace
fig.add_trace(go.Scatter(
    x=time,
    y=conf,
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'),
    name='Prior confidence'))

# Error band trace
fig.add_trace(go.Scatter(
    x=list(time) + list(time[::-1]),
    y=list(conf +conf_error) + list((conf - conf_error)[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False))

conf_nobc_cluster_slice = nonbaseline_corr_cluster_analysis_results['prior_conf_zscore']['clusters'][0][0]
conf_nobc_cluster_indices = list(range(conf_nobc_cluster_slice.start, conf_nobc_cluster_slice.stop))  
conf_nobc_significant_times = nonbaseline_corr_coefs_df['time'].iloc[conf_nobc_cluster_indices]

# Add dots for significant clusters (at the time points of the cluster)
fig.add_trace(go.Scatter(
    x=conf_nobc_significant_times,  # Time points where the cluster is significant
    y=[-0.105] * len(conf_nobc_significant_times),  # Set a fixed low y-value for dots (adjust if necessary)
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.6)', size=6),
    name='Significant Cluster - Prior Confidence',
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
    title_text="Regression weights",  
)

fig.update_layout(
    title=dict(
            text="Peristimulus regression weights for prior confidence and change point probability on the non-baseline corrected pupil signal"
        ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800,
    font=dict(size=18),
    legend=dict(
        x=0.18,
        y=1.0,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.7)'))


fig.show()