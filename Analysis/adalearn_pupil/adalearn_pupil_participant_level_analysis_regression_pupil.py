"""
This script calls and runs functions related to the participant level regression analysis between pupil epochs and latent parameters of uncertainty, performed at a participant level. The analysis runs the analysis for both baseline corrected and non-baseline corrected pupil data (defined by separable calls of the function; the non-baseline corrected analysis is typically specified by an ending of "_nobc"). The analysis also runs the mediation analysis for both baseline corrected and non-baseline corrected pupil data, with and without zero exclusion.
"""


from adalearn_analysis_participant_level_functions_regression_pupil import pupil_regression_analysis, pupil_uncertainty_lr_mediation_analysis

from adalearn_pupil.adalearn_pupil_participant_level_analysis_pupil import  outcome_epoched_pupil_df,  outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove_nobc

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str,  condition_str


conf_cpp_coef, pupil_regress_df, residuals_df = pupil_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals'], covariates = ['outcome', 'outcome_num', 'task_session'], baseline_corrected=True,  plot_regression_analysis=False)



conf_cpp_coef_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals'], covariates = ['outcome', 'outcome_num', 'task_session'], baseline_corrected=False, plot_regression_analysis=False)



mediation_output = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=True, exclude_zero=False, plot_mediation_analysis=False)



mediation_output_nobc = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=False, exclude_zero=False, plot_mediation_analysis=False)



mediation_output_zero_excluded = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=True, exclude_zero=True, plot_mediation_analysis=False)



mediation_output_nobc_zero_excluded = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=False, exclude_zero=True, plot_mediation_analysis=False)