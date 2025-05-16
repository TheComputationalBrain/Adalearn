"""
This scripts loads pupil data obtained during the task at a participant level. Called scripts then process the data. This process includes: the preprocessing of the pupil signal (e.g. interpolation of blinks and removal of artefacts) before epoching the data around each sequence outcome. The called functions are adaptable. For example, the baseline correction can be set to a specific time window or no baseline correction can be applied at all (separate calls of the function are already set up in differentiate use of baseline correction versus no baseline correction as it is an important distinction in later analysis; the non-baseline corrected analysis is typically specified by an ending of "_nobc"). The level of the low-pass filter can be determined, as well as the eye to extract the data from. The script also allows the user to specify the time window around the outcome presentation that they want to epoch the data (e.g. -0.4 to 1.4s). For more specific details on the parameters of the functions, please refer to the docstrings of the functions called. Alongside the pupil epochs, a list of the epochs that are excluded for an individual is also generated, that is an important element used in future analysis (e.g. pupil by uncertainty parameter peristimulus regressions). In addition to the loading, preprocessing and epoching of the pupil response to sequence outcomes, pupil parameters (relating to amplitude and baseline, for example), are ploted in a window around task change points. Plots can be generated related to the interpolation procedure and epoching if the relevant option is activated. The script is designed to be run for a single participant at a time.To load the data from the relevant csv files, the following scripts should be run before running this script:
- adalearn_pupil_participant_level_analysis_behav.py
"""

from adalearn_analysis_participant_level_functions_pupil import load_and_process_pupil_data, all_outcome_pupil_epochs

from adalearn_analysis_participant_level_functions_data_utilities import save_dataframes_to_csv, load_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, participant_str, list_subject_selected, condition_str, task_sessions, from_session, to_session



dataframes = load_csvs(directory_path=directory_path, condition_str=condition_str, substrings=[ 'change_point_indices'])



preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=[-0.25, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 



outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.25, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)



outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc, outcome_epochs_to_remove_nobc_df = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)




save_dataframes_to_csv(dataframes={
    'preprocessed_pupil_df': preprocessed_pupil_df, 
    'pupil_trig_df': pupil_trig_df,
    'outcome_epoched_pupil_df': outcome_epoched_pupil_df,
    'outcome_blink_ratio_df': outcome_blink_ratio_df,
    'outcome_epochs_to_remove_df': outcome_epochs_to_remove_df,
    'outcome_epoched_pupil_nobc_df': outcome_epoched_pupil_df_nobc,
    'outcome_blink_ratio_nobc_df': outcome_blink_ratio_df_nobc,
    'outcome_epochs_to_remove_nobc_df': outcome_epochs_to_remove_nobc_df 
    }, 
    directory_path=directory_path, 
    participant_str=participant_str, 
    condition_str=condition_str, 
    task='Pupil'
    )