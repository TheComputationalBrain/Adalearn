""" Main analysis script of Ada_learn VNS pupil data. Analysis perfomed for both conditions for each individual. Data for each individual saved to a spreadsheet with each run through."""


import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from pupil_signal_functions import load_pupil_data, convert_change_points_to_pupil, pupil_change_point_analysis

from behavioural_analysis_test import c1_change_point_indices, c2_change_point_indices

from participant_task_characteristics import participant, participant_str, list_subject_selected, condition_1_str, condition_2_str, c1_from_session, c1_to_session, c1_task_sessions, c2_from_session, c2_to_session, c2_task_sessions

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil
from pick_best_eye import pick_best_eye


# Loading pupil data using function in pupil_signal functions
c1_pupil_data, c1_trig_data = load_pupil_data(participant_str=participant_str, condition=condition_1_str, condition_task_sessions=c1_task_sessions, plot=True) 
c2_pupil_data, c2_trig_data = load_pupil_data(participant_str=participant_str, condition=condition_2_str, condition_task_sessions=c2_task_sessions, plot=True)


# Converting identified change point within behavioural data to pupil timestamp using function in pupil_signal_functions
c1_change_point_indices_pupil, c1_change_point_num_pupil = convert_change_points_to_pupil(pupil_trig_df=c1_trig_data, change_point_indices=c1_change_point_indices, condition=condition_1_str)
c2_change_point_indices_pupil, c2_change_point_num_pupil = convert_change_points_to_pupil(pupil_trig_df=c2_trig_data, change_point_indices=c2_change_point_indices, condition=condition_2_str)


# Epoching and plotting pupil response to change points using function in pupil_signal_functions 
c1_epoched_data = pupil_change_point_analysis(participant_str=participant_str, list_subject_selected=list_subject_selected, condition=condition_1_str, from_session=c1_from_session, to_session=c1_to_session, change_point_onsets=c1_change_point_indices_pupil, change_point_num=c1_change_point_num_pupil, outlier_limit_std=10, outlier_correct_average_amount=10, window_start_before=1, window_end_after=1.4, z_score=False, plot=True)
c2_epoched_data = pupil_change_point_analysis(participant_str=participant_str, list_subject_selected=list_subject_selected, condition=condition_2_str, from_session=c2_from_session, to_session=c2_to_session, change_point_onsets=c2_change_point_indices_pupil, change_point_num=c2_change_point_num_pupil, outlier_limit_std=10, outlier_correct_average_amount=10, window_start_before=1, window_end_after=1.4, z_score=False, plot=True)