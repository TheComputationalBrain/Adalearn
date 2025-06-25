"""
This script loads the data extracted from the probability learning task (Adalearn) for a individual participant and analyses the participant's performance. The behavioural data is first extracted, before task features (i.e. change points) are identified and extracted into their own dataframe. Some analysis is performed on change points relating to previous literature (i.e. identification of 'obvious' and 'non-obvious' change points as performed by Jepma et al. (2016) - PLos Comp Biol). Performance level metrics relating primarily to absolute error are also calculated. The script offers the user the ability to plot the individual participant's data for inspection (the generated plot will be saved in a newly created directory within the specified participant's data directory). The script is designed to be run on a single participant at a time, mainly to inspect the data and obtain the data for further collation in later analysis. The user should ensure that the relevant directories are set up correctly before running the script.  
"""

import adalearn_analysis_participant_level_functions_behav as behav
import adalearn_analysis_participant_level_functions_data_utilities as data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import participant_str, directory_path, condition_str, task_sessions



behavioural_df, computational_var_df = behav.load_behavioural_data( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

num_change_points, change_points_per_session, change_point_indices, behavioural_df = behav.behavioural_change_point_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_histogram_zscores=False)

mean_abs_error_df = behav.behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_performance=False)

df_dict = {
    'behavioural_df': behavioural_df,
    'mean_abs_error_df': mean_abs_error_df,
    'change_point_indices': change_point_indices
}

data_utils.save_dataframes_to_csv(dataframes=df_dict, directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, task='Pupil')

