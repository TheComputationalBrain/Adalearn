"""
This script calls and runs functions related to the participant level regression analysis between pupil epochs and latent parameters of uncertainty, performed at a participant level. The analysis runs the analysis for both baseline corrected and non-baseline corrected pupil data (defined by separable calls of the function; the non-baseline corrected analysis is typically specified by an ending of "_nobc"). The analysis also runs plot checks to examine pupil responses around change points before running mediation analysis for both baseline corrected and non-baseline corrected pupil data, with and without zero exclusion.
"""


from adalearn_analysis_participant_level_functions_regression_pupil import pupil_peristimulus_regression_analysis, pupil_on_learning_rate_regression_analysis, pupil_uncertainty_lr_mediation_analysis

from adalearn_analysis_participant_level_functions_pupil import pupil_response_around_change_point

from adalearn_analysis_participant_level_functions_data_utilities import save_dataframes_to_csv, load_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, participant_str, condition_str



dataframes = load_csvs(directory_path=directory_path, condition_str=condition_str, substrings=['computational_var_df', 'change_point_indices', 'outcome_epoched_pupil_df', 'outcome_epochs_to_remove_df', 'outcome_epoched_pupil_nobc_df', 'outcome_epochs_to_remove_nobc_df'])



conf_cpp_coef, pupil_regress_df, residuals_df = pupil_peristimulus_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=True,  plot_regression_analysis=False)



conf_cpp_coef_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_peristimulus_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_nobc_df'], variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=False, plot_regression_analysis=False)




pupil_amp_effect_around_cp_df = pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'], change_point_indices=dataframes['change_point_indices'], before=5, after=5, estimate_type='baseline_corrected_effect', plot_pupil_around_change_point=False)



pupil_baseline_around_cp_df = pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_nobc_df'], change_point_indices=dataframes['change_point_indices'], before=5, after=5, estimate_type='baseline', plot_pupil_around_change_point=False)



pupil_v_lr_regress_df = pupil_on_learning_rate_regression_analysis(directory_path=directory_path,participant_str=participant_str, condition_str=condition_str, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'], outcome_epoched_pupil_df_nobc=dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], parameter_estimate='estimate', exclude_zero=False, plot_pupil_lr_regress=False)



mediation_analysis_conf = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'],  outcome_epoched_pupil_df_nobc=dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], uncertainty_param = 'conf_zscore', variables=['conf_zscore', 'cpp_zscore'], exclude_zero=False)



mediation_analysis_cpp = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'],  outcome_epoched_pupil_df_nobc=dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], uncertainty_param = 'cpp_zscore', variables=['conf_zscore', 'cpp_zscore'], exclude_zero=False)



save_dataframes_to_csv(
    directory_path=directory_path, 
    participant_str=participant_str, 
    condition_str=condition_str,
    task='Pupil', 
    dataframes={
        'pupil_regress_df': pupil_regress_df, 
        'pupil_regress_df_nobc': pupil_regress_df_nobc, 
        'pupil_amp_effect_around_cp_df': pupil_amp_effect_around_cp_df, 'pupil_baseline_around_cp_df': pupil_baseline_around_cp_df, 
        'pupil_v_lr_regress_df': pupil_v_lr_regress_df, 
        'mediation_analysis_conf': mediation_analysis_conf, 
        'mediation_analysis_cpp': mediation_analysis_cpp}
)