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


# Define participant, condition and task session characteristics
# Participants parameters
participant = 1
participant_str = str(participant).zfill(2)
list_subject_selected = [participant]

# Condition and task session parameters
# Condition 1 (c1) characteristics 
condition_1 = str(1).zfill(2)
c1_from_session, c1_to_session = 1, 20
c1_first_task_session_str = str(c1_from_session).zfill(2)
c1_last_task_session_str = str(c1_to_session).zfill(2)
c1_task_sessions = [str(session).zfill(2) for session in range(c1_from_session, c1_to_session + 1)]

# Condition 2 (c2) characteristics 
condition_2 = str(2).zfill(2)
c2_from_session, c2_to_session = 1, 16
c2_first_task_session_str = str(c2_from_session).zfill(2)
c2_last_task_session_str = str(c2_to_session).zfill(2)
c2_task_sessions = [str(session).zfill(2) for session in range(c2_from_session, c2_to_session + 1)]



# Loading adalearn task behavioural data using function created in behavioural_task_functions
adalearn_data_c1 = load_behavioural_data(participant_str, condition_1, c1_task_sessions)
adalearn_data_c2 = load_behavioural_data(participant_str, condition_2, c2_task_sessions)


# Creating and loading adalearn task beahvioural data into a DataFrame using function created in behavioural_task_functions
adalearn_data_c1_df = create_behavioural_dataframe(adalearn_data_c1)
adalearn_data_c2_df = create_behavioural_dataframe(adalearn_data_c2)


# Checking whether expected quantities are present in the data (e.g. number of unique task sessions, number of outcome presentations within each task session)
adalearn_c1_diagnostics = behavioural_data_diagnostics(adalearn_data_c1_df, c1_task_sessions)
adalearn_c2_diagnostics = behavioural_data_diagnostics(adalearn_data_c2_df, c2_task_sessions)


# Plotting the raw data obtained from the adalearn task using function in behavioural_task_functions
adalearn_c1_fig = plot_behavioural_data(adalearn_data_c1_df, participant, condition_1, estimate_v_io_plot=True)
adalearn_c2_fig = plot_behavioural_data(adalearn_data_c2_df, participant, condition_2, estimate_v_io_plot=True)


# Identifying change points within behavioural data for each task session using function in behavioural_task_functions
c1_change_point_num, c1_change_point_by_sess, c1_change_point_indices = behavioural_change_point_analysis(adalearn_data_c1_df)
c2_change_point_num, c2_change_point_by_sess, c2_change_point_indices = behavioural_change_point_analysis(adalearn_data_c2_df)


# Calculate behavioural learning rate using function in behavioural_task_functions
adalearn_data_c1_df = calculate_behavioural_learning_rate(adalearn_data_c1_df)
adalearn_data_c2_df = calculate_behavioural_learning_rate(adalearn_data_c2_df)


# Plotting behavioural learning rate using function in behavioural_task_functions
c1_learning_rate_fig = plotting_behavioural_learning_rate(adalearn_data_c1_df, condition_1)
c2_learning_rate_fig = plotting_behavioural_learning_rate(adalearn_data_c2_df, condition_2)  


# Plotting average behavioural learning rate dynamics locked to change point using function in behavioural_task_functions
c1_learning_rate_by_cp, c1_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c1_df, c1_change_point_indices, participant, condition_1, pre=2, post=10)
c2_learning_rate_by_cp, c2_avg_learning_rate_cp = learning_rate_by_change_point(adalearn_data_c2_df, c2_change_point_indices, participant, condition_2, pre=2, post = 10)