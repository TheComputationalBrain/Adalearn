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

from behavioural_task_functions import load_behavioural_data, create_behavioural_dataframe, behavioural_data_diagnostics,behavioural_change_point_analysis, calculate_behavioural_learning_rate, plotting_behavioural_learning_rate, learning_rate_by_change_point, plot_behavioural_data

from participant_task_characteristics import participant,participant_str, condition_1_str, condition_2_str, c1_task_sessions, c2_task_sessions



# Loading adalearn task behavioural data using function created in behavioural_task_functions
adalearn_data_c1 = load_behavioural_data(participant_str, condition_1_str, c1_task_sessions)
adalearn_data_c2 = load_behavioural_data(participant_str, condition_2_str, c2_task_sessions)


# Creating and loading adalearn task beahvioural data into a DataFrame using function created in behavioural_task_functions
adalearn_data_c1_df = create_behavioural_dataframe(adalearn_data_c1)
adalearn_data_c2_df = create_behavioural_dataframe(adalearn_data_c2)


# Checking whether expected quantities are present in the data (e.g. number of unique task sessions, number of outcome presentations within each task session)
adalearn_c1_diagnostics = behavioural_data_diagnostics(adalearn_data_c1_df, c1_task_sessions)
adalearn_c2_diagnostics = behavioural_data_diagnostics(adalearn_data_c2_df, c2_task_sessions)


# Plotting the raw data obtained from the adalearn task using function in behavioural_task_functions
adalearn_c1_fig = plot_behavioural_data(adalearn_data_c1_df, participant, condition_1_str, estimate_v_io_plot=True)
adalearn_c2_fig = plot_behavioural_data(adalearn_data_c2_df, participant, condition_2_str, estimate_v_io_plot=True)


# Identifying change points within behavioural data for each task session using function in behavioural_task_functions
c1_change_point_num, c1_change_point_by_sess, c1_change_point_indices = behavioural_change_point_analysis(adalearn_data_c1_df)
c2_change_point_num, c2_change_point_by_sess, c2_change_point_indices = behavioural_change_point_analysis(adalearn_data_c2_df)


# Calculate behavioural learning rate using function in behavioural_task_functions
adalearn_data_c1_df = calculate_behavioural_learning_rate(adalearn_data_c1_df)
adalearn_data_c2_df = calculate_behavioural_learning_rate(adalearn_data_c2_df)


# Plotting behavioural learning rate using function in behavioural_task_functions
c1_learning_rate_fig = plotting_behavioural_learning_rate(adalearn_data_c1_df, c1_change_point_indices, participant_str, condition_1_str)
c2_learning_rate_fig = plotting_behavioural_learning_rate(adalearn_data_c2_df, c2_change_point_indices, participant_str, condition_2_str)  


# Plotting average behavioural learning rate dynamics locked to change point using function in behavioural_task_functions
c1_learning_rate_by_cp, c1_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c1_df, c1_change_point_indices, participant, condition_1_str, pre=2, post=10)
c2_learning_rate_by_cp, c2_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c2_df, c2_change_point_indices, participant, condition_2_str, pre=2, post = 10)