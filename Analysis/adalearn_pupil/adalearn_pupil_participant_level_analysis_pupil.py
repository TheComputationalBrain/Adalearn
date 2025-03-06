import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_pupil import load_and_process_pupil_data, change_point_pupil_epochs, comp_var_pupil_quartile_epochs, all_outcome_pupil_epochs, outcome_colour_pupil_epochs, pupil_response_around_change_point

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import change_point_indices, behavioural_df

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_str, task_sessions, from_session, to_session



preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=1.2, baseline_cor=[-0.4, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 



outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.4, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_pupil_dict = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.4, 0], select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



pupil_amp_around_cp_df = pupil_response_around_change_point(participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=outcome_epoched_pupil_df, change_point_indices=change_point_indices, before=6, after=6, estimate_type='amplitude', plot_pupil_around_change_point=False)



pupil_avg_around_cp_df = pupil_response_around_change_point(participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=outcome_epoched_pupil_df, change_point_indices=change_point_indices, before=6, after=6, estimate_type='mean', plot_pupil_around_change_point=False)



pupil_baseline_around_cp_df = pupil_response_around_change_point(participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, change_point_indices=change_point_indices, before=6, after=6, estimate_type='baseline', plot_pupil_around_change_point=False)



change_point_pupil_dict = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, change_point_indices=change_point_indices, baseline_cor=[-0.4, 0], select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



surprise_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, baseline_cor=[-0.4, 0], variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



prior_conf_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, baseline_cor=[-0.4, 0],variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



cpp_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, baseline_cor=[-0.4, 0],variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)