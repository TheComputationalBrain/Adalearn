""" Main analysis script of Ada-learn VNS computational model data. The model reuses the implementation of the Bayesian model for the probability learning task of Meyniel (2020) whose code is available online at https://github.com/florentmeyniel/TransitionProbModel.Analysis. Script runs methods to compute surprise around each outcome sequence, the prior confidence and change point probability. It also computes the model derived learning rate and its dynamics around change points. ****Data for each individual saved to a spreadsheet with each run through.****"""

import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

from computational_analysis_functions import behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis,learning_rate_by_change_point, compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations

from participant_task_characteristics import participant_str, condition_1_str, condition_2_str

from behavioural_analysis_test import c1_behavioural_df, c1_change_point_indices, c2_behavioural_df, c2_change_point_indices

P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9


c1_behavioural_df = behavioural_learning_rate_analysis(behavioural_df=c1_behavioural_df, change_point_indices_df=c1_change_point_indices, participant_str=participant_str, condition_str=condition_1_str, plot_learning_rate_by_ses=True)

c2_behavioural_df = behavioural_learning_rate_analysis(behavioural_df=c2_behavioural_df, change_point_indices_df=c2_change_point_indices, participant_str=participant_str, condition_str=condition_2_str, plot_learning_rate_by_ses=True)



c1_behavioural_df = ideal_observer_learning_rate_analysis(behavioural_df=c1_behavioural_df, change_point_indices_df=c1_change_point_indices, participant_str=participant_str, condition_str=condition_1_str, plot_learning_rate_by_ses=True)

c2_behavioural_df = ideal_observer_learning_rate_analysis(behavioural_df=c2_behavioural_df, change_point_indices_df=c2_change_point_indices, participant_str=participant_str, condition_str=condition_2_str, plot_learning_rate_by_ses=True)



c1_filtered_estimate_learning_rate_by_cp, c1_averaged_estimate_learning_rate_by_cp = learning_rate_by_change_point(behavioural_df = c1_behavioural_df, change_point_indices = c1_change_point_indices, participant_str= participant_str, condition_str = condition_1_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = True)

c2_filtered_estimate_learning_rate_by_cp, c2_averaged_estimate_learning_rate_by_cp = learning_rate_by_change_point(behavioural_df = c2_behavioural_df, change_point_indices = c2_change_point_indices, participant_str= participant_str, condition_str = condition_2_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = True)



c1_model_estimate_learning_rate_by_cp, c1_averaged_model_learning_rate_by_cp = learning_rate_by_change_point(behavioural_df = c1_behavioural_df, change_point_indices = c1_change_point_indices, participant_str= participant_str, condition_str = condition_1_str, learning_rate_parameter = 'ideal_observer', pre=2, post=10, plot_learning_rate_by_cp = True)

c2_model_estimate_learning_rate_by_cp, c2_model_estimate_learning_rate_by_cp = learning_rate_by_change_point(behavioural_df = c2_behavioural_df, change_point_indices = c2_change_point_indices, participant_str= participant_str, condition_str = condition_2_str, learning_rate_parameter = 'ideal_observer', pre=2, post=10, plot_learning_rate_by_cp = True)



c1_surprise, c1_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_1_str, c1_behavioural_df, c1_change_point_indices, plot_surprise=True, plot_posterior_conf=True, plot_posterior_dist_means=True)

c2_surprise, c2_surprise_cat_df = compute_surprise_from_observations(participant_str, condition_2_str, c2_behavioural_df, c2_change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



c1_prior_dist_df, c1_prior_conf_df, c1_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_1_str, c1_behavioural_df, c1_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=True, plot_prior_conf=True, plot_cpp=True)

c2_prior_dist_df, c2_prior_conf_df, c2_cpp_df = compute_prior_conf_and_cpp_from_observations(participant_str, condition_2_str, c2_behavioural_df, c2_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False) 
