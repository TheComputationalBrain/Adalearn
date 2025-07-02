"""
This script calls and runs functions for participant level regression analysis between pupil epochs and latent parameters of uncertainty. The script runs the analysis for both baseline corrected and non-baseline corrected pupil data (defined by separable calls of the function; the non-baseline corrected analysis is typically specified by an ending of "_nobc"). The analysis also runs plot checks to examine pupil responses around change points before running mediation analysis for both baseline corrected and non-baseline corrected pupil data, with and without zero exclusion.
"""
import adalearn_analysis_participant_level_functions_regression_pupil as pupil
import adalearn_analysis_participant_level_functions_data_utilities as data_utils 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, participant_str, condition_str



dataframes = data_utils.load_csvs(directory_path=directory_path, condition_str=condition_str, substrings=['computational_var_df', 'change_point_indices', 'outcome_epoched_pupil_df', 'outcome_epochs_to_remove_df', 'outcome_epoched_pupil_nobc_df', 'outcome_epochs_to_remove_nobc_df'])

conf_cpp_coef, pupil_regress_df, residuals_df = pupil.pupil_peristimulus_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=True,  plot_regression_analysis=False)

conf_cpp_coef_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil.pupil_peristimulus_regression_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_epoched_pupil_df=dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_nobc_df'], variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=False, plot_regression_analysis=False)

data_utils.save_dataframes_to_csv(
    directory_path=directory_path, 
    participant_str=participant_str, 
    condition_str=condition_str,
    task='Pupil', 
    dataframes={
        'pupil_regress_df': pupil_regress_df, 
        'pupil_regress_df_nobc': pupil_regress_df_nobc
})