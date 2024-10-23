""" Main analysis script of Ada-learn VNS computational model data. The model reuses the implementation of the Bayesian model for the probability learning task of Meyniel (2020) whose code is available online at https://github.com/florentmeyniel/TransitionProbModel.Analysis. Script runs methods to compute surprise around each outcome sequence, the prior confidence and change point probability. It also computes the model derived learning rate and its dynamics around change points. ****Data for each individual saved to a spreadsheet with each run through.****"""

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

from behavioural_analysis_test import c1_behavioural_df, c2_behavioural_df, c1_change_point_indices, c2_change_point_indices

P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
LEARNING_RATE = 0.4




c1_surprise, c1_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_1_str, c1_behavioural_df, c1_change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)

c2_surprise, c2_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_2_str, c2_behavioural_df, c2_change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



c1_prior_dist_df, c1_prior_conf_df, c1_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_1_str, c1_behavioural_df, c1_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)

c2_prior_dist_df, c2_prior_conf_df, c2_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_2_str, c2_behavioural_df, c2_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False) 



c1_model_learning_rate = compute_model_learning_rate(participant_str, condition_1_str, c1_behavioural_df, c1_change_point_indices, lr = LEARNING_RATE, plot_model_learning_rate_no_noise=False, plot_model_learning_rate_noisy=False)

c2_model_learning_rate = compute_model_learning_rate(participant_str, condition_2_str, c2_behavioural_df, c2_change_point_indices, lr = LEARNING_RATE, plot_model_learning_rate_no_noise=False, plot_model_learning_rate_noisy=False)



c1_learning_rate_by_cp, c1_average_model_learning_rate_by_cp = model_learning_rate_by_change_point(c1_model_learning_rate, c1_change_point_indices, participant_str, condition_1_str, pre = 2, post = 10, plot_model_learning_rate_by_cp=False)

c2_learning_rate_by_cp, c2_average_model_learning_rate_by_cp = model_learning_rate_by_change_point(c2_model_learning_rate, c2_change_point_indices, participant_str, condition_2_str, pre = 2, post = 10, plot_model_learning_rate_by_cp=False)
