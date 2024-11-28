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

from pupil_signal_functions import load_and_process_pupil_data, change_point_pupil_epochs, surprise_pupil_epochs, all_outcome_pupil_epochs, outcome_colour_pupil_epochs

from computational_analysis_test import c1_surprise, c2_surprise


from behavioural_analysis_test import c1_change_point_indices,  c1_behavioural_df, c2_change_point_indices,  c2_behavioural_df

from participant_task_information import directory_path, participant_str, list_subject_selected, condition_1_str, condition_2_str, c1_from_session, c1_to_session, c1_task_sessions, c2_from_session, c2_to_session, c2_task_sessions

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil



c1_preprocessed_pupil_df, c1_pupil_trig_df, c1_pupil_ses_df, c1_blink_df, c1_epoched_pupil_df, c1_blink_ratio_epoch_df, c1_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=3, select_eye=True, remove_epochs=True, z_score=True, plot_interpolation_process=False) 

c2_preprocessed_pupil_df, c2_pupil_trig_df, c2_pupil_ses_df, c2_blink_df, c2_epoched_pupil_df, c2_blink_ratio_epoch_df, c2_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=3, select_eye=True, remove_epochs=True, z_score=True, plot_interpolation_process=False) 



c1_outcome_onsets, c1_outcome_nums, c1_outcome_epoched_pupil_df, c1_outcome_blink_ratio_df, c1_outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

c2_outcome_onsets, c2_outcome_nums, c2_outcome_epoched_pupil_df, c2_outcome_blink_ratio_df, c2_outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



c1_outcome_1_onsets, c1_outcome_1_nums, c1_outcome_1_epoched_pupil_df, c1_outcome_1_blink_ratio_df, c1_outcome_1_epochs_to_remove, c1_outcome_0_onsets, c1_outcome_0_nums, c1_outcome_0_epoched_pupil_df, c1_outcome_0_blink_ratio_df,  c1_outcome_0_epochs_to_remove = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)

c2_outcome_1_onsets, c2_outcome_1_nums, c2_outcome_1_epoched_pupil_df, c2_outcome_1_blink_ratio_df, c2_outcome_1_epochs_to_remove, c2_outcome_0_onsets, c2_outcome_0_nums, c2_outcome_0_epoched_pupil_df, c2_outcome_0_blink_ratio_df, c2_outcome_0_epochs_to_remove = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



c1_change_point_onsets, c1_change_point_nums, c1_epoched_pupil_cp_df, c1_blink_ratio_cp_df, c1_epochs_to_remove = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, change_point_indices=c1_change_point_indices, select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)

c2_change_point_onsets, c2_change_point_nums, c2_epoched_pupil_cp_df, c2_blink_ratio_cp_df, c2_epochs_to_remove = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, change_point_indices=c2_change_point_indices, select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



c1_surprise_first_quart_onsets, c1_surprise_first_quart_nums, c1_epoched_pupil_first_surprise_df, c1_first_surprise_blink_ratio_df, c1_first_surprise_epochs_to_remove, c1_surprise_second_quart_onsets, c1_surprise_second_quart_nums, c1_epoched_pupil_second_surprise_df, c1_second_surprise_blink_ratio_df, c1_second_surprise_epochs_to_remove, c1_surprise_third_quart_onsets, c1_surprise_third_quart_nums, c1_epoched_pupil_third_surprise_df, c1_third_surprise_blink_ratio_df, c1_third_surprise_epochs_to_remove, c1_surprise_fourth_quart_onsets, c1_surprise_fourth_quart_nums, c1_epoched_pupil_fourth_surprise_df, c1_fourth_surprise_blink_ratio_df, c1_fourth_surprise_epochs_to_remove = surprise_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, surprise_df=c1_surprise, select_eye=True, remove_epochs=True, plot_pupil_by_surprise=False)

c2_surprise_first_quart_onsets, c2_surprise_first_quart_nums, c2_epoched_pupil_first_surprise_df, c2_first_surprise_blink_ratio_df, c2_first_surprise_epochs_to_remove, c2_surprise_second_quart_onsets, c2_surprise_second_quart_nums, c2_epoched_pupil_second_surprise_df, c2_second_surprise_blink_ratio_df, c2_second_surprise_epochs_to_remove, c2_surprise_third_quart_onsets, c2_surprise_third_quart_nums, c2_epoched_pupil_third_surprise_df, c2_third_surprise_blink_ratio_df, c2_third_surprise_epochs_to_remove, c2_surprise_fourth_quart_onsets, c2_surprise_fourth_quart_nums, c2_epoched_pupil_fourth_surprise_df, c2_fourth_surprise_blink_ratio_df, c2_fourth_surprise_epochs_to_remove = surprise_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, surprise_df=c2_surprise, select_eye=True, remove_epochs=True, plot_pupil_by_surprise=False)