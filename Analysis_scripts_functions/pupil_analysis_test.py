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

from pupil_signal_functions import load_pupil_data, change_point_pupil_epochs, surprise_pupil_epochs, all_outcome_pupil_epochs, outcome_colour_pupil_epochs

from computational_analysis_test import c1_surprise, c2_surprise

from behavioural_analysis_test import c1_change_point_indices, c2_change_point_indices

from participant_task_characteristics import participant_str, list_subject_selected, condition_1_str, condition_2_str, c1_from_session, c1_to_session, c1_task_sessions, c2_from_session, c2_to_session, c2_task_sessions

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil
from pick_best_eye import pick_best_eye



c1_pupil_df, c1_pupil_trig_df, c1_pupil_sess_df, c1_blink_df = load_pupil_data(participant_str=participant_str, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, selected_eye='right', plot_pupil_by_task_session=False)

c2_pupil_df, c2_pupil_trig_df, c2_pupil_sess_df, c2_blink_df = load_pupil_data(participant_str=participant_str, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, selected_eye='right', plot_pupil_by_task_session=False)



c1_outcome_onsets, c1_outcome_nums, c1_outcome_epoched_pupil_df, c1_outcome_blink_ratio_df = all_outcome_pupil_epochs(participant_str=participant_str, condition_str=condition_1_str, list_subject_selected=list_subject_selected, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, session_list=None, plot_averaged_pupil_response=False, plot_raster_pupil_response=False) 

c2_outcome_onsets, c2_outcome_nums, c2_outcome_epoched_pupil_df, c2_outcome_blink_ratio_df = all_outcome_pupil_epochs(participant_str=participant_str, condition_str=condition_2_str, list_subject_selected=list_subject_selected, from_session=c2_from_session, to_session=c2_to_session, pupil_trig_df=c2_pupil_trig_df, session_list=None, plot_averaged_pupil_response=False, plot_raster_pupil_response=False) 



c1_outcome_1_onsets, c1_outcome_1_nums, c1_outcome_1_epoched_pupil_df, c1_outcome_1_blink_ratio_df, c1_outcome_0_onsets, c1_outcome_0_nums, c1_outcome_0_epoched_pupil_df, c1_outcome_0_blink_ratio_df_= outcome_colour_pupil_epochs(participant_str=participant_str, condition_str=condition_1_str, list_subject_selected=list_subject_selected, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c1_pupil_trig_df, session_list=None, plot_pupil_by_colour_response=False)

c2_outcome_1_onsets, c2_outcome_1_nums, c2_outcome_1_epoched_pupil_df, c2_outcome_1_blink_ratio_df, c2_outcome_0_onsets, c2_outcome_0_nums, c2_outcome_0_epoched_pupil_df, c2_outcome_0_blink_ratio_df_= outcome_colour_pupil_epochs(participant_str=participant_str, condition_str=condition_2_str,list_subject_selected=list_subject_selected, from_session=c1_from_session, to_session=c1_to_session, pupil_trig_df=c2_pupil_trig_df, session_list=None, plot_pupil_by_colour_response=False)



c1_change_point_onsets, c1_change_point_nums, c1_epoched_pupil_cp_df, c1_blink_ratio_cp_df = change_point_pupil_epochs(participant_str = participant_str, list_subject_selected = list_subject_selected, condition_str = condition_1_str, from_session = c1_from_session, to_session = c1_to_session, pupil_trig_df = c1_pupil_trig_df, change_point_indices = c1_change_point_indices, session_list = None, plot_pupil_by_cp=False)

c2_change_point_onsets, c2_change_point_nums, c2_epoched_pupil_cp_df, c2_blink_ratio_cp_df = change_point_pupil_epochs(participant_str = participant_str, list_subject_selected = list_subject_selected, condition_str = condition_2_str, from_session = c2_from_session, to_session = c2_to_session, pupil_trig_df = c2_pupil_trig_df, change_point_indices = c2_change_point_indices, session_list = None, plot_pupil_by_cp=False)



c1_low_surp_ses_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '18', '19', '20']

c1_low_surprise_dict, c1_low_surprise_count_dict, c1_high_surprise_dict, c1_high_surprise_count_dict, c1_low_surprise_epoch_df, c1_high_surprise_epoch_df, c1_low_surprise_blink_ratio, c1_high_surprise_blink_ratio = surprise_pupil_epochs(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_1_str, from_session=c1_from_session, to_session=c1_to_session, surprise_df=c1_surprise, pupil_trig_df=c1_pupil_trig_df, high_surp_ses_list=None, low_surp_ses_list=c1_low_surp_ses_list, plot_pupil_by_surprise=False)

c2_low_surprise_dict, c2_low_surprise_count_dict, c2_high_surprise_dict, c2_high_surprise_count_dict, c2_low_surprise_epoch_df, c2_high_surprise_epoch_df, c2_low_surprise_blink_ratio, c2_high_surprise_blink_ratio = surprise_pupil_epochs(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_2_str, from_session=c2_from_session, to_session=c2_to_session, surprise_df=c2_surprise, pupil_trig_df=c2_pupil_trig_df, high_surp_ses_list=None, low_surp_ses_list=None, plot_pupil_by_surprise=False)






c1_epoch_matrix = [c1_outcome_epoched_pupil_df.iloc[:, i].values for i in range(c1_outcome_epoched_pupil_df.shape[1] - 4)]

c2_epoch_matrix = [c2_outcome_epoched_pupil_df.iloc[:, i].values for i in range (c2_outcome_epoched_pupil_df.shape[1] - 4)]


c1_epoch_matrix_filtered = [col.values for i, (col_name, col) in enumerate(c1_outcome_epoched_pupil_df.iloc[:, :-4].items())
                   if (col <= 3).all() and (col >= -3).all()]

print(f"The number of rows removed in filtering was {len(c1_epoch_matrix) - len(c1_epoch_matrix_filtered)}")

fig = px.imshow(c1_epoch_matrix_filtered, 
                labels=dict(x="Time (s)", y="Outcome number", color="Pupil Diameter (z-score)"),
                x=c1_outcome_epoched_pupil_df['time'].values,  
                aspect="auto",  
                origin="lower"  
               )

fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

fig.update_layout(title=f'Filtered heatmap of epoched time series of pupil dimeter following a change point for participant 03 in condition 02')

fig.show()


c2_epoch_matrix_filtered = [col.values for i, (col_name, col) in enumerate(c2_outcome_epoched_pupil_df.iloc[:, :-4].items())
                   if (col <= 3).all() and (col >= -3).all()]

print(f"The number of rows removed in filtering was {len(c2_epoch_matrix) - len(c2_epoch_matrix_filtered)}")

fig = px.imshow(c2_epoch_matrix_filtered, 
                labels=dict(x="Time (s)", y="Outcome number", color="Pupil Diameter (z-score)"),
                x=c2_outcome_epoched_pupil_df['time'].values,  
                aspect="auto",  
                origin="lower"  
               )

fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

fig.update_layout(title=f'Heatmap of epoched time series of pupil dimeter following a change point for participant 03 in condition 02')

fig.show()