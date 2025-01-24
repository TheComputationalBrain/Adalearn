""" Main analysis script of Ada-learn VNS computational model data. The model reuses an implementation of the Bayesian model for the probability learning task of Meyniel (2020), whose code is available online at https://github.com/florentmeyniel/TransitionProbModel.Analysis. Script runs methods to compute surprise around each outcome sequence, the prior confidence and change point probability. It also computes the model derived learning rate and its dynamics around change points. ****Data for each individual saved to a spreadsheet with each run through.****"""

import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

from computational_analysis_functions import behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis,learning_rate_by_change_point, compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, behav_v_model_linear_analysis

from data_utility_functions import plot_session_behav_comp

from participant_task_information import participant_str, condition_1_str, condition_2_str, P_C, P1_MIN, P1_MAX

from behavioural_analysis_test import c1_behavioural_df,c1_computational_var_df, c1_change_point_indices, c2_behavioural_df, c2_computational_var_df, c2_change_point_indices



c1_computational_var_df = behavioural_learning_rate_analysis(computational_var_df=c1_computational_var_df, change_point_indices_df=c1_change_point_indices, participant_str=participant_str, condition_str=condition_1_str, plot_learning_rate_by_ses=False)

c2_computational_var_df = behavioural_learning_rate_analysis(computational_var_df=c2_computational_var_df, change_point_indices_df=c2_change_point_indices, participant_str=participant_str, condition_str=condition_2_str, plot_learning_rate_by_ses=False)



c1_computational_var_df = ideal_observer_learning_rate_analysis(computational_var_df=c1_computational_var_df, change_point_indices_df=c1_change_point_indices, participant_str=participant_str, condition_str=condition_1_str, plot_learning_rate_by_ses=False)

c2_computational_var_df = ideal_observer_learning_rate_analysis(computational_var_df=c2_computational_var_df, change_point_indices_df=c2_change_point_indices, participant_str=participant_str, condition_str=condition_2_str, plot_learning_rate_by_ses=False)



c1_filtered_estimate_learning_rate_by_cp, c1_averaged_estimate_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df = c1_computational_var_df, change_point_indices = c1_change_point_indices, participant_str= participant_str, condition_str = condition_1_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = False)

c2_filtered_estimate_learning_rate_by_cp, c2_averaged_estimate_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df = c2_computational_var_df, change_point_indices = c2_change_point_indices, participant_str= participant_str, condition_str = condition_2_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = False)



c1_model_estimate_learning_rate_by_cp, c1_averaged_model_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=c1_computational_var_df, change_point_indices=c1_change_point_indices, participant_str=participant_str, condition_str=condition_1_str, learning_rate_parameter='ideal_observer', pre=2, post=10, plot_learning_rate_by_cp=False)

c2_model_estimate_learning_rate_by_cp, c2_model_estimate_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=c2_computational_var_df, change_point_indices=c2_change_point_indices, participant_str=participant_str, condition_str=condition_2_str, learning_rate_parameter='ideal_observer', pre=2, post=10, plot_learning_rate_by_cp=False)



c1_computational_var_df, c1_surprise_cat_df = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_1_str, computational_var_df=c1_computational_var_df, change_point_indices_df=c1_change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)

c2_computational_var_df, c2_surprise_cat_df = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_2_str, computational_var_df=c2_computational_var_df, change_point_indices_df=c2_change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



c1_computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_1_str, computational_var_df=c1_computational_var_df, change_point_indices_df=c1_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)

c2_computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_2_str, computational_var_df=c2_computational_var_df, change_point_indices_df=c2_change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf-pred', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)




c1_behav_v_model_estimate_corr, c1_behav_v_model_estimate_regress = behav_v_model_linear_analysis(computational_var_df=c1_computational_var_df, participant_str=participant_str, condition_str=condition_1_str, model_var='ideal_observer', behav_var='estimate', exclude_zero=False, plot_behav_v_model_scatter=True, add_regression=True)


c2_behav_v_model_estimate_corr, c2_behav_v_model_estimate_regress = behav_v_model_linear_analysis(computational_var_df=c2_computational_var_df, participant_str=participant_str, condition_str=condition_2_str, model_var='ideal_observer', behav_var='estimate', exclude_zero=False, plot_behav_v_model_scatter=True, add_regression=True)



c1_behav_v_model_lr_corr, c1_behav_v_model_lr_regress = behav_v_model_linear_analysis(computational_var_df=c1_computational_var_df, participant_str=participant_str, condition_str=condition_1_str, model_var='cleaned_learning_rate_ideal_observer', behav_var='cleaned_learning_rate_estimate', exclude_zero=False, plot_behav_v_model_scatter=True, add_regression=True)

c2_behav_v_model_lr_corr, c2_behav_v_model_lr_regress = behav_v_model_linear_analysis(computational_var_df=c2_computational_var_df, participant_str=participant_str, condition_str=condition_2_str, model_var='cleaned_learning_rate_ideal_observer', behav_var='cleaned_learning_rate_estimate', exclude_zero=False, plot_behav_v_model_scatter=True, add_regression=True)



plot_session_behav_comp(participant_str=participant_str, condition_str=condition_1_str, behavioural_df=c1_behavioural_df, computational_df=c1_computational_var_df, change_point_indices=c1_change_point_indices, task_session=1)

plot_session_behav_comp(participant_str=participant_str, condition_str=condition_2_str, behavioural_df=c2_behavioural_df, computational_df=c2_computational_var_df, change_point_indices=c2_change_point_indices, task_session=1)