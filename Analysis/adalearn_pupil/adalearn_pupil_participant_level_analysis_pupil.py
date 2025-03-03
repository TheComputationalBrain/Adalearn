import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_pupil import load_and_process_pupil_data, change_point_pupil_epochs, comp_var_pupil_quartile_epochs, all_outcome_pupil_epochs, outcome_colour_pupil_epochs

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import change_point_indices, behavioural_df

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_str, task_sessions, from_session, to_session



preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=1.2, baseline_cor=[-0.2, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 



outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_pupil_dict = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.2, 0], select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



change_point_pupil_dict = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, change_point_indices=change_point_indices, baseline_cor=[-0.2, 0],select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



surprise_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



prior_conf_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



cpp_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)




pupil_baseline_df = outcome_epoched_pupil_df_nobc.copy()
pupil_baseline_df = pupil_baseline_df.drop(columns=['mean', 'std', 'sem'])
pupil_baseline_df = pupil_baseline_df.set_index('time')

filtered_pupil_data = pupil_baseline_df.loc[(pupil_baseline_df.index >= -0.4) & (pupil_baseline_df.index <= 0)]

baseline_means = filtered_pupil_data.mean()

baseline_means_df = pd.DataFrame({
    'baseline_means': baseline_means, 
    'task_session': (baseline_means.index // 75) + 1
    })


unique_sessions = baseline_means_df['task_session'].unique()  
color_map = {session: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, session in enumerate(unique_sessions)}

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=baseline_means_df.index, 
    y=baseline_means_df['baseline_means'], 
    mode='markers', 
    marker=dict(color=[color_map[i] for i in baseline_means_df['task_session']]),  
    name='Baseline'
))

fig.update_layout(
    title=f'Pupil baseline means across trials (identified by task sessions) for particpant {participant_str}',
    xaxis_title='Trial number',
    yaxis_title='Pupil Size (mm)',
    showlegend=False
)

fig.show()


num_sessions = len(unique_sessions)

fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=False,
                    subplot_titles=[f'Mean baseline (non-corrected) pupil diameter in task session {session}' for session in unique_sessions])

# Add pupil traces to each subplot
for i, session in enumerate(unique_sessions, start=1):
    session_data = baseline_means_df[baseline_means_df['task_session'] == session]

    fig.add_trace(go.Scatter(x=session_data.index, y=session_data['baseline_means'], mode='lines+markers', name='Baseline Pupil Diameter (all data)', showlegend=False, line=dict(color='rgba(0, 0, 255, 0.2)')), row=i, col=1)

    fig.add_trace(go.Scatter(x=session_data.index[10:], y=session_data['baseline_means'].iloc[10:], mode='lines+markers', name='Baseline Pupil Diameter (reduced data)', showlegend=False, line=dict(color='rgba(255, 0, 0, 0.2)')), row=i, col=1)

fig.update_layout(
    height=900*num_sessions,
    width=1200,
    title = f'Pupil baseline means for trials within each task sessions for particpant {participant_str}',
    xaxis_title='Trial number',
    yaxis_title='Pupil Size (mm)',
    showlegend=False)

fig.show()

