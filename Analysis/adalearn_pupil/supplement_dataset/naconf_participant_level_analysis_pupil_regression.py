"""
This scripts loads pupil data obtained during the task at a participant level. Called scripts then process the data. This process includes: the preprocessing of the pupil signal (e.g. interpolation of blinks and removal of artefacts) before epoching the data around each sequence outcome. The called functions are adaptable. For example, the baseline correction can be set to a specific time window or no baseline correction can be applied at all (separate calls of the function are already set up in differentiate use of baseline correction versus no baseline correction as it is an important distinction in later analysis; the non-baseline corrected analysis is typically specified by an ending of "_nobc"). The level of the low-pass filter can be determined, as well as the eye to extract the data from. The script also allows the user to specify the time window around the outcome presentation that they want to epoch the data (e.g. -0.4 to 1.4s). For more specific details on the parameters of the functions, please refer to the docstrings of the functions called. Alongside the pupil epochs, a list of the epochs that are excluded for an individual is also generated, that is an important element used in future analysis (e.g. pupil by uncertainty parameter peristimulus regressions). In addition to the loading, preprocessing and epoching of the pupil response to sequence outcomes, pupil parameters (relating to amplitude and baseline, for example), are ploted in a window around task change points. Plots can be generated related to the interpolation procedure and epoching if the relevant option is activated. The script is designed to be run for a single participant at a time.   
"""

import adalearn_pupil.supplement_dataset.naconf_pupil_regression_functions as pupil_regress
import adalearn_analysis_participant_level_functions_data_utilities as data_utils

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import directory_path, selected_participant



dataframes = data_utils.load_csvs(directory_path=directory_path, condition_str='01', substrings=['task_session_df',  'outcome_epoched_pupil_df', 'outcome_epochs_to_remove_df', 'outcome_epoched_pupil_nobc_df',  'outcome_epochs_to_remove_nobc_df'])

coefficients_df, pupil_regress_df, residuals_df = pupil_regress.pupil_peristimulus_regression_analysis(participant_str = selected_participant, directory_path = directory_path, task_session_df = dataframes['task_session_df'], outcome_epoched_pupil_df = dataframes['outcome_epoched_pupil_df'], outcome_epochs_to_remove=dataframes['outcome_epochs_to_remove_df'], variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = True, plot_regression_analysis=False)

coefficients_df_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_regress.pupil_peristimulus_regression_analysis(participant_str = selected_participant, directory_path = directory_path, task_session_df = dataframes['task_session_df'], outcome_epoched_pupil_df = dataframes['outcome_epoched_pupil_nobc_df'], outcome_epochs_to_remove = dataframes['outcome_epochs_to_remove_nobc_df'], variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = False, plot_regression_analysis=False)

data_utils.save_dataframes_to_csv(dataframes={
    'pupil_regress_df': pupil_regress_df, 
    'pupil_regress_df_nobc': pupil_regress_df_nobc, 
    'coefficients_df': coefficients_df, 
    'coefficients_df_nobc': coefficients_df_nobc, 
    'residuals_df': residuals_df, 
    'residuals_df_nobc': residuals_df_nobc
}, 
directory_path=directory_path,
participant_str=selected_participant,
condition_str='01',
task='NACONF-Supplement'
)