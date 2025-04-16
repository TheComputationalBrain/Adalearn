"""
This scripts calls functions to runs group level analysis relating to participants' performance, the association between participants' and normative estimates and relationships between normative parameters themseleves. Running this script will plot and save the final figures relating to these analyses. The group level analysis is run on dataframes created and saved as .csv files in the participant level analysis (it therefore does not run if these files have not been created). 
"""

from adalearn_pupil.adalearn_pupil_participant_characteristics import base_dir, condition_str
from adalearn_pupil.adalearn_pupil_group_level_participant_demographics import excluded_participants

from adalearn_analysis_group_level_functions_behav_comp import sample_absolute_error_cv_and_score, sample_behav_uncertainty_on_learning_rate, sample_behav_v_model_estimate, sample_behav_v_model_learning_rate, sample_learning_rates_by_cp, model_uncertainty_on_learning_rate



mean_abs_error_avg_df, mean_score_avg_df = sample_absolute_error_cv_and_score(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], save_plot=True)



behav_model_estimate_regress_df, behav_model_estimate_corr_df, bin_average_data = sample_behav_v_model_estimate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], save_plot=True, plot_correlation_histogram=True)



behav_model_lr_regress_df, behav_model_lr_corr_df, bin_average_data = sample_behav_v_model_learning_rate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=False, save_plot=True)



behav_model_lr_regress_df_zero_excluded, behav_model_lr_corr_df_zero_excluded, bin_average_data_zero_excluded = sample_behav_v_model_learning_rate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=True, save_plot=True)



lr_by_cp_avg_df_behav, lr_by_cp_avg_df_model, behav_model_lr_round_cp_corr_df = sample_learning_rates_by_cp(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=False, save_plot=True)



lr_by_cp_avg_df_behav_zero_excluded, lr_by_cp_avg_df_model_zero_excluded, behav_model_lr_round_cp_corr_df_zero_excluded = sample_learning_rates_by_cp(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=True, save_plot=True)



model_uncert_on_lr_df, model_group_avg_df = model_uncertainty_on_learning_rate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], save_plot=True)
                                                                               
                                                                               
                                                                               
behav_uncert_on_lr_df, behav_group_conf_lr_df, behav_group_conf_cpp_df = sample_behav_uncertainty_on_learning_rate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=False, save_plot=True)



behav_uncert_on_lr_df_zero_excluded, behav_group_conf_lr_df_zero_excluded, behav_group_conf_cpp_df_zero_excluded = sample_behav_uncertainty_on_learning_rate(condition_str=condition_str, base_dir=base_dir, excluded_participants=[], exclude_zero=True, save_plot=True)