import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

from computational_analysis_functions import compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, compute_model_learning_rate, model_learning_rate_by_change_point
from participant_task_characteristics import participant_str, condition_1_str, condition_2_str
from behavioural_analysis_test import adalearn_data_c1_df, adalearn_data_c2_df, c1_change_point_indices, c2_change_point_indices

P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
LEARNING_RATE = 0.4



# Computing the surprise per observation within each task session using a function in computational_analysis_functions
c1_surprise, c1_post_dist_df, c1_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_1_str, adalearn_data_c1_df, c1_change_point_indices, plot_surprise=True, plot_posterior_conf=True, plot_posterior_dist_means=True)
c2_surprise, c2_post_dist_df, c2_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_2_str, adalearn_data_c2_df, c2_change_point_indices, plot_surprise=True, plot_posterior_conf=True, plot_posterior_dist_means=True)


# Computes the prior confidence and change point probability of each observation within each task session using a function in computational_analysis_functions
c1_prior_dist_df, c1_prior_conf_df, c1_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_1_str, adalearn_data_c1_df, c1_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=True, plot_prior_conf=True, plot_cpp=True)
c2_prior_dist_df, c2_prior_conf_df, c2_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_2_str, adalearn_data_c2_df, c2_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=True, plot_prior_conf=True, plot_cpp=True) 


# Computes model_derived learning rate in response to a sequence of outcomes using a function in computational_analysis_functions
c1_model_learning_rate = compute_model_learning_rate(participant_str, condition_1_str, adalearn_data_c1_df, c1_change_point_indices, lr = LEARNING_RATE, plot_model_learning_rate_no_noise= True, plot_model_learning_rate_noisy=False)
c2_model_learning_rate = compute_model_learning_rate(participant_str, condition_2_str, adalearn_data_c2_df, c2_change_point_indices, lr = LEARNING_RATE, plot_model_learning_rate_no_noise= True, plot_model_learning_rate_noisy=False)


# Calculates the mean model learning rate around identified change points using function in computational_analysis_functions
c1_learning_rate_by_cp, c1_average_model_learning_rate_by_cp = model_learning_rate_by_change_point(c1_model_learning_rate, c1_change_point_indices, participant_str, condition_1_str, pre = 2, post = 10, plot_model_learning_rate_by_cp=True)
c2_learning_rate_by_cp, c2_average_model_learning_rate_by_cp = model_learning_rate_by_change_point(c2_model_learning_rate, c2_change_point_indices, participant_str, condition_2_str, pre=2, post=10, plot_model_learning_rate_by_cp=True)
