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

from pupil_signal_functions import load_pupil_data, process_pupil_data, change_point_pupil_epochs, surprise_pupil_epochs

from computational_analysis_test import c1_surprise, c2_surprise

from behavioural_analysis_test import c1_change_point_indices, c2_change_point_indices

from participant_task_characteristics import participant_str, list_subject_selected, condition_1_str, condition_2_str, c1_from_session, c1_to_session, c1_task_sessions, c2_from_session, c2_to_session, c2_task_sessions

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil
from pick_best_eye import pick_best_eye


# Loading pupil data using function in pupil_signal functions
c1_pupil_data, c1_trig_data = load_pupil_data(participant_str, condition_1_str, c1_task_sessions, plot=True) 
c2_pupil_data, c2_trig_data = load_pupil_data(participant_str, condition_2_str, c2_task_sessions, plot=True)

# Epoching and plotting pupil response to change points using function in pupil_signal_functions 
c1_change_point_onsets, c1_change_point_nums, c1_epoched_pupil_cp_df, c1_blink_ratio_cp_df = change_point_pupil_epochs(participant_str, list_subject_selected, condition_1_str, c1_from_session +1, c1_to_session, c1_trig_data, c1_change_point_indices, session_list=None, plot=True)
c2_change_point_onsets, c2_change_point_nums, c2_epoched_pupil_cp_df, c2_blink_ratio_cp_df = change_point_pupil_epochs(participant_str, list_subject_selected, condition_2_str, c2_from_session +1, c2_to_session, c2_trig_data, c2_change_point_indices, session_list=None, plot=True)


# Epoching and plotting pupil response based on upper and lower quartiles for surprising observations using function in pupil_signal_functions 
c1_low_surp_ses_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '19', '20']

c1_low_surprise_dict, c1_low_surprise_count_dict, c1_high_surprise_dict, c1_high_surprise_count_dict, c1_low_surprise_epoch_df, c1_high_surprise_epoch_df, c1_high_surprise_blink_ratio, c1_low_surprise_blink_ratio = surprise_pupil_epochs(participant_str, list_subject_selected, condition_1_str, c1_from_session +1, c1_to_session, c1_surprise, c1_trig_data, high_surp_ses_list=None, low_surp_ses_list=c1_low_surp_ses_list, plot=True)
c2_low_surprise_dict, c2_low_surprise_count_dict, c2_high_surprise_dict, c2_high_surprise_count_dict, c2_low_surprise_epoch_df, c2_high_surprise_epoch_df, c2_high_surprise_blink_ratio, c2_low_surprise_blink_ratio = surprise_pupil_epochs(participant_str, list_subject_selected, condition_2_str, c2_from_session +1, c1_to_session, c2_surprise, c2_trig_data, high_surp_ses_list=None, low_surp_ses_list=None, plot=True)
