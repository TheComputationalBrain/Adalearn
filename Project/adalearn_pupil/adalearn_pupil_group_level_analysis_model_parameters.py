"""
This script computes ideal observer model parameters for each participant in the group-level analysis of the Pupil task in the Adalearn project. It loads individual participant data, computes various computational variables such as learning rates, surprise, prior confidence, change point probability, and information gain, and saves the results to a group-level .csv file. The following files need to be run prior to running this script to load the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
"""
import pandas as pd
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_participant_level_functions_behav as behav
import adalearn_analysis_participant_level_functions_comp as comp

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR, TASK_SESSIONS, P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR, RESOL



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR,
    substrings=[
        'exclude_participants_behav_cor_df',
        'change_point_indices_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(Path(f'{BASE_DIR}/data'))

excluded_participants = list(dataframes['exclude_participants_behav_cor_df']['participant'])

group_computational_var_df_list = []

# For loop, extracting data from behavioural files for each participant within data list and then computing the model parameters for the sequences within each observed sequence
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{BASE_DIR}/data/sub-{participant_str}')
    else:
        continue

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant].copy()

    _ , computational_var_df = behav.load_behavioural_data( participant_str=participant_str, condition_str=CONDITION_STR, directory_path=directory_path, condition_task_sessions=TASK_SESSIONS, plot_behav_data=False)

    print(f"\nComputing model parameters...")
    # Computing participants' estimate-derived learning rate
    computational_var_df = comp.behavioural_learning_rate_analysis(participant_str=participant_str, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,  estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)

    # Computing model estimate-derived learning rate
    computational_var_df = comp.ideal_observer_learning_rate_analysis( participant_str=participant_str, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)

    # Computing surprise of each observation
    computational_var_df,_ = comp.run_inference_from_observations(participant_str=participant_str, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, resol=RESOL, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)

    # Computing prior confidence and change point probability of each observation
    computational_var_df = comp.compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=CONDITION_STR,directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, resol=RESOL, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, conf_type='prev-conf-pred', unit='uncertainty_sd', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)

    # Computing the information gain of each observation
    computational_var_df = comp.compute_information_gain(participant_str=participant_str, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, resol=20, p1_min=P1_MIN, p1_max=P1_MAX, plot_dkl=False, add_surprise_and_conf=True)

    computational_var_df['participant'] = participant_str

    group_computational_var_df_list.append(computational_var_df)

# Concatenate list into group-level dataframe
group_level_computational_var_df = pd.concat(group_computational_var_df_list, ignore_index=True)

# Save group-level csv
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'computational_var_df': group_level_computational_var_df
},
directory_path=BASE_DIR,
condition_str=CONDITION_STR,
task='Pupil'
)
