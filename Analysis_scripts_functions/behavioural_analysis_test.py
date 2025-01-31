""" Main analysis script of Ada-learn VNS behavioural data. Analysis perfomed for both conditions for each individual. Script runs methods to load the behavioural data, store it in a dataframe, generate a dataframe storing the outcome information at the point a change in the hidden task structure occurs (i.e. a change point), that can be passed to other methods in order to perform analysis around change points. Methods in this script also calculate the behavioural learning rate and computes the dynamics of this learning rate around change points.  ****Data for each individual saved to a spreadsheet with each run through.****"""


import pandas as pd
import numpy as np
import json
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import zscore
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from behavioural_task_functions import load_behavioural_data, behavioural_change_point_analysis, behavioural_performance_analysis

from participant_task_information import directory_path, participant_str, condition_1_str, condition_2_str, c1_task_sessions, c2_task_sessions



c1_behavioural_df, c1_computational_var_df = load_behavioural_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_1_str, condition_task_sessions=c1_task_sessions, plot_behav_data=False)

c2_behavioural_df, c2_computational_var_df = load_behavioural_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_2_str, condition_task_sessions=c2_task_sessions, plot_behav_data=False)



c1_change_point_num, c1_change_point_by_sess, c1_change_point_indices, c1_behavioural_df = behavioural_change_point_analysis(participant_str=participant_str, condition_str= condition_1_str, behavioural_df=c1_behavioural_df, plot_histogram_zscores=False)

c2_change_point_num, c2_change_point_by_sess, c2_change_point_indices, c2_dist_to_change_point = behavioural_change_point_analysis(participant_str=participant_str, condition_str= condition_2_str, behavioural_df=c2_behavioural_df, plot_histogram_zscores=False)



c1_mean_abs_error_df = behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_1_str, behavioural_df=c1_behavioural_df, plot_performance=False)

c2_mean_abs_error_df = behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_2_str, behavioural_df=c2_behavioural_df, plot_performance=False)