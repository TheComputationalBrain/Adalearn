""" Main analysis script of Ada-learn VNS pupil data. Functions within the script loads and analyses the pupil data, primarily in relation to defined events of interest (i.e. outcomes that represent a change point or are high or low in surprise). ****TO UPDATE**** """


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

from adalearn_VNS.adalearn_VNS_participant_level_analysis_comp import c1_computational_var_df, c2_computational_var_df

from adalearn_VNS.adalearn_VNS_participant_level_analysis_behav import c1_change_point_indices,  c1_behavioural_df, c2_change_point_indices, c2_behavioural_df

from adalearn_VNS.adalearn_VNS_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_1_str, condition_2_str, c1_from_session, c1_to_session, c1_task_sessions, c2_from_session, c2_to_session, c2_task_sessions



c1_preprocessed_pupil_df, c1_pupil_trig_df, c1_pupil_ses_df, c1_blink_df, c1_epoched_pupil_df, c1_blink_ratio_epoch_df, c1_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=1.2, baseline_cor=[-0.2, 0], lpf=[5], select_eye=True, remove_epochs=True, z_score=True, plot_interpolation_process=False) 

c2_preprocessed_pupil_df, c2_pupil_trig_df, c2_pupil_ses_df, c2_blink_df, c2_epoched_pupil_df, c2_blink_ratio_epoch_df, c2_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=1.2,  baseline_cor=[-0.2, 0], lpf=[5], select_eye=True, remove_epochs=True, z_score=True, plot_interpolation_process=False) 



c1_outcome_onsets, c1_outcome_nums, c1_outcome_epoched_pupil_df, c1_outcome_blink_ratio_df, c1_outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

c2_outcome_onsets, c2_outcome_nums, c2_outcome_epoched_pupil_df, c2_outcome_blink_ratio_df, c2_outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



c1_outcome_onsets_nobc, c1_outcome_nums_nobc, c1_outcome_epoched_pupil_df_nobc, c1_outcome_blink_ratio_df_nobc, c1_outcome_epochs_to_remove_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, baseline_cor=[-0.2, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

c2_outcome_onsets_nobc, c2_outcome_nums_nobc, c2_outcome_epoched_pupil_df_nobc, c2_outcome_blink_ratio_df_nobc, c2_outcome_epochs_to_remove_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, baseline_cor=[-0.2, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



c1_outcome_pupil_dict = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)

c2_outcome_pupil_dict = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



c1_change_point_pupil_dict = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, change_point_indices=c1_change_point_indices, select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)

c2_change_point_pupil_dict = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, change_point_indices=c2_change_point_indices, select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



c1_surprise_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, computational_var_df=c1_computational_var_df, variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)

c2_surprise_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, computational_var_df=c2_computational_var_df, variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



c1_prior_conf_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, computational_var_df=c1_computational_var_df, variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)

c2_prior_conf_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, computational_var_df=c2_computational_var_df, variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



c1_cpp_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, comp_var_df=c1_computational_var_df, variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)

c2_cpp_pupil_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, comp_var_df=c2_computational_var_df, variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)
