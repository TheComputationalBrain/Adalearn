import plotly.io as pio
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_comp import compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, compute_information_gain, behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis, learning_rate_by_change_point, behav_v_model_linear_analysis, uncertainty_on_learning_rate_participant_level, comp_parameters_on_info_gain

from adalearn_analysis_participant_level_functions_data_utilities import plot_session_behav_comp, evaluation_of_learning_rate

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import behavioural_df, computational_var_df, change_point_indices

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, condition_str, from_session, to_session, task_sessions, P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR



computational_var_df = behavioural_learning_rate_analysis(computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, participant_str=participant_str, condition_str=condition_str, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



computational_var_df = ideal_observer_learning_rate_analysis(computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, participant_str=participant_str, condition_str=condition_str, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=computational_var_df, change_point_indices=change_point_indices, participant_str=participant_str, condition_str=condition_str, learning_rate_parameter='estimate', pre=2, post=10, plot_learning_rate_by_cp=False)



filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=computational_var_df, change_point_indices=change_point_indices, participant_str=participant_str, condition_str=condition_str, learning_rate_parameter='ideal_observer', pre=2, post=10, plot_learning_rate_by_cp=False)



computational_var_df, surprise_cat_df = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)



computational_var_df = compute_information_gain(participant_str=participant_str, condition_str=condition_str,computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,  p_c=P_C, resol=20, p1_min=P1_MIN, p1_max=P1_MAX, plot_dkl=False, add_surprise_and_conf=True)



behav_v_model_estimate_corr, behav_v_model_estimate_regress = behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, model_var='ideal_observer', behav_var='estimate', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



behav_v_model_lr_corr, behav_v_model_lr_regress = behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, model_var='cleaned_learning_rate_ideal_observer', behav_var='cleaned_learning_rate_estimate', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



plot_session_behav_comp(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, computational_df=computational_var_df, change_point_indices=change_point_indices, task_session=9, turn_on_plotting=False)



lr_evaluation_dict = evaluation_of_learning_rate(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, computational_var_df=computational_var_df, change_point_indices=change_point_indices, quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=False)



learning_rate_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='estimate', exclude_zero=False, plot_corr_subplots=False)



computational_var_df, dkl_regression_df = comp_parameters_on_info_gain(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, plot_dkl_regression=False)

