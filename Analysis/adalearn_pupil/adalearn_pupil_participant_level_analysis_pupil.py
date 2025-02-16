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


#sys.path.append('/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Misc')
#from preprocess_eyedata import pupil



preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.2, window_end_after=1.2, baseline_cor=[-0.2, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 



outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_1_onsets, outcome_1_nums, outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df, outcome_1_epochs_to_remove, outcome_0_onsets, outcome_0_nums, outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df,  outcome_0_epochs_to_remove = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.2, 0], select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



change_point_onsets, change_point_nums, epoched_pupil_cp_df, blink_ratio_cp_df, epochs_to_remove = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, change_point_indices=change_point_indices, baseline_cor=[-0.2, 0],select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



surprise_first_quart_onsets, surprise_first_quart_nums, epoched_pupil_first_surprise_df, first_surprise_blink_ratio_df, first_surprise_epochs_to_remove, surprise_second_quart_onsets, surprise_second_quart_nums, epoched_pupil_second_surprise_df, second_surprise_blink_ratio_df, second_surprise_epochs_to_remove, surprise_third_quart_onsets, surprise_third_quart_nums, epoched_pupil_third_surprise_df, third_surprise_blink_ratio_df, third_surprise_epochs_to_remove, surprise_fourth_quart_onsets, surprise_fourth_quart_nums, epoched_pupil_fourth_surprise_df, fourth_surprise_blink_ratio_df, fourth_surprise_epochs_to_remove = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



conf_first_quart_onsets, conf_first_quart_nums, epoched_pupil_first_conf_df, first_conf_blink_ratio_df, first_conf_epochs_to_remove, conf_second_quart_onsets, conf_second_quart_nums, epoched_pupil_second_conf_df, second_conf_blink_ratio_df, second_conf_epochs_to_remove, conf_third_quart_onsets, conf_third_quart_nums, epoched_pupil_third_conf_df, third_conf_blink_ratio_df, third_conf_epochs_to_remove, conf_fourth_quart_onsets, conf_fourth_quart_nums, epoched_pupil_fourth_conf_df, fourth_conf_blink_ratio_df, fourth_conf_epochs_to_remove = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



cpp_first_quart_onsets, cpp_first_quart_nums, epoched_pupil_first_cpp_df, first_cpp_blink_ratio_df, first_cpp_epochs_to_remove, cpp_second_quart_onsets, cpp_second_quart_nums, epoched_pupil_second_cpp_df, second_cpp_blink_ratio_df, second_cpp_epochs_to_remove, cpp_third_quart_onsets, cpp_third_quart_nums, epoched_pupil_third_cpp_df, third_cpp_blink_ratio_df, third_cpp_epochs_to_remove, cpp_fourth_quart_onsets, cpp_fourth_quart_nums, epoched_pupil_fourth_cpp_df, fourth_cpp_blink_ratio_df, fourth_cpp_epochs_to_remove = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, computational_var_df=computational_var_df, variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)