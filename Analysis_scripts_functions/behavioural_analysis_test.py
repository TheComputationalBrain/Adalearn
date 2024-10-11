""" Main analysis script of Ada_learn VNS behavioural data. Analysis perfomed for both conditions for each individual. Data for each individual saved to a spreadsheet with each run through."""


import pandas as pd
import numpy as np
import json
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from behavioural_task_functions import load_behavioural_data, behavioural_change_point_analysis, behavioural_learning_rate_analysis, learning_rate_by_change_point

from participant_task_characteristics import participant,participant_str, condition_1_str, condition_2_str, c1_task_sessions, c2_task_sessions


# Loading adalearn task behavioural data using function created in behavioural_task_functions
adalearn_data_c1_df = load_behavioural_data(participant_str, condition_1_str, c1_task_sessions, plot_behav_data=False, plot_estimate_v_ideal=False)
adalearn_data_c2_df = load_behavioural_data(participant_str, condition_2_str, c2_task_sessions, plot_behav_data=False, plot_estimate_v_ideal=False)


# Identifying change points within behavioural data for each task session using function in behavioural_task_functions
c1_change_point_num, c1_change_point_by_sess, c1_change_point_indices = behavioural_change_point_analysis(adalearn_data_c1_df)
c2_change_point_num, c2_change_point_by_sess, c2_change_point_indices = behavioural_change_point_analysis(adalearn_data_c2_df)


# Calculate behavioural learning rate using function in behavioural_task_functions
adalearn_data_c1_df = behavioural_learning_rate_analysis(adalearn_data_c1_df, c1_change_point_indices, participant_str, condition_1_str, plot_learning_rate_by_ses=False)
adalearn_data_c2_df = behavioural_learning_rate_analysis(adalearn_data_c2_df, c2_change_point_indices, participant_str, condition_2_str, plot_learning_rate_by_ses=False)


# Plotting average behavioural learning rate dynamics locked to change point using function in behavioural_task_functions
c1_learning_rate_by_cp, c1_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c1_df, c1_change_point_indices, participant, condition_1_str, pre=2, post=10, plot_learning_rate_by_cp=False)
c2_learning_rate_by_cp, c2_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c2_df, c2_change_point_indices, participant, condition_2_str, pre=2, post = 10, plot_learning_rate_by_cp=False)