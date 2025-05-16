"""
This script calls and runs functions used to check the pupil response to certain parameters (both task features and computational parameters). Plots can be generated to see how the pupil responds to different outcomes, the presence of a change point or latent uncertainty parameters. This script is not a part of the main analysis, but is used just to check pupil responses. This script is designed to run for a single participant at a time.
"""

from adalearn_analysis_participant_level_functions_pupil import change_point_pupil_epochs, comp_var_pupil_quartile_epochs,  outcome_colour_pupil_epochs

from adalearn_analysis_participant_level_functions_data_utilities import load_csvs, save_dataframes_to_csv


from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_str, task_sessions, from_session, to_session


dataframes = load_csvs(directory_path=directory_path, condition_str=condition_str, substrings=['change_point_indices', 'pupil_trig_df', 'computational_var_df'])



outcome_pupil_dict = outcome_colour_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=dataframes['pupil_trig_df'], baseline_cor=[-0.25, 0], select_eye=True, remove_epochs=True, plot_pupil_by_colour_response=False)



change_point_pupil_dict = change_point_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=dataframes['pupil_trig_df'], change_point_indices=dataframes['change_point_indices'], baseline_cor=[-0.25, 0], select_eye=True, remove_epochs=True, plot_pupil_by_cp=False)



surprise_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=dataframes['pupil_trig_df'], computational_var_df=dataframes['computational_var_df'], baseline_cor=[-0.4, 0], variable_cat='surprise_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



prior_conf_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=dataframes['pupil_trig_df'], computational_var_df=dataframes['computational_var_df'], baseline_cor=[-0.4, 0],variable_cat='prior_conf_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)



cpp_pupil_quartile_dict = comp_var_pupil_quartile_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=dataframes['pupil_trig_df'], computational_var_df=dataframes['computational_var_df'], baseline_cor=[-0.4, 0],variable_cat='cpp_cat', select_eye=True, remove_epochs=True, plot_pupil_by_quartile=False)
