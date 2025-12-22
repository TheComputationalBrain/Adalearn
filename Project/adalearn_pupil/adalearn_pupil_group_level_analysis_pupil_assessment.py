"""
This script processes pupil data for each participant for the Pupil task of the Adalearn project. The script extracts pupil diameter epochs around outcome presentations, checking for excessive interpolation, before saving the results to .csv files. The script also extracts pupil position data (x and y coordinates) and creates a scalar value based on the euclidean distance from the observation origin, which is then used in further analysis. The following scripts need to be run to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil_group_level_analysis_behav_task_performance.py
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_participant_level_functions_pupil as pupil
import adalearn_analysis_group_level_functions_pupil_analysis as pupil_analysis

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR, FROM_SESSION, TO_SESSION, TASK_SESSIONS



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'change_point_indices_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(directory=Path(f'{BASE_DIR}/data'))

excluded_participants = list(dataframes['exclude_participants_behav_cor_df']['participant'])

exclude_participants = []
preprocessed_pupil_df_list, pupil_trig_df_list = [], [] 
outcome_epoched_pupil_df_list, outcome_blink_ratio_df_list, outcome_epochs_to_remove_df_list, outcome_epochs_time_df_list, outcome_epochs_columns_df_list = [], [], [], [], []
outcome_epoched_pupil_nobc_df_list, outcome_blink_ratio_nobc_df_list, outcome_epochs_to_remove_nobc_df_list, outcome_epochs_time_nobc_df_list, outcome_epochs_columns_nobc_df_list = [], [], [], [], []
missing_epochs_check = []
preprocessed_pupil_pos_df_list = []
outcome_epoched_x_pupil_df_list, outcome_epoched_y_pupil_df_list = [], []
outcome_pos_epochs_columns_df_list, outcome_pos_blink_ratio_df_list, outcome_pos_epochs_to_remove_df_list, outcome_pos_epochs_time_df_list = [], [], [], []



for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{BASE_DIR}/data/sub-{participant_str}')
    else:
        continue

    # Checking whether pupil data files are correct and can be processed
    print(f"\nIntial check on file quality before analysing pupil epochs...")
    sessions_to_process = [str(session).zfill(2) for session in range(FROM_SESSION, TO_SESSION + 1)]
    
    problem_files = 0
    total_files = 0

    for session in sessions_to_process:
        file_dir = f"{directory_path}/ses-{CONDITION_STR}/eye"
        file_name = f"sub-{participant_str}_ses-{CONDITION_STR}_task-AdalearnPupil_run-{session}_eyetracker.txt"
        file_path = f"{file_dir}/{file_name}"

        if os.path.exists(file_path):
            total_files += 1
            too_many_msgs = pupil.check_pupil_file_triggers(file_path=file_path, threshold=5)
            if too_many_msgs:
                print(f"Too many consecutive MSGs in {file_name}")
                problem_files += 1

    if total_files > 0 and problem_files / total_files >= 0.5:
        print(f"\nMore than 50% of triggers in the pupil file appear problematic for participant {participant_str} — excluding from further pupil analysis.")
        percent_epochs_removed = None
        exclude_participants_df = pd.DataFrame([{
            'exclude_participant': participant,
            'percent_epochs_removed': percent_epochs_removed
        }]) 
        exclude_participants.append(exclude_participants_df)
        continue

    # Loading and preprocessing pupil diameter data
    print(f"\nLoading pupil diameter data...")
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = pupil.load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=CONDITION_STR, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=[-0.25, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 

    preprocessed_pupil_df['participant'] = participant_str
    pupil_trig_df['participant'] = participant_str

    preprocessed_pupil_df_list.append(preprocessed_pupil_df)
    pupil_trig_df_list.append(pupil_trig_df)

    # Loading and preprocessing pupil position data (for future analyses)
    print(f"\nLoading pupil position data...")
    preprocessed_pupil_pos_df, pupil_trig_pos_df, pupil_pos_ses_df, blink_pos_df, epoched_x_pupil_pos_df, epoched_y_pupil_pos_df, blink_pos_ratio_epoch_df, epochs_pos_to_remove = pupil.load_and_process_pupil_position_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=CONDITION_STR, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=[], lpf=[5], select_eye=True, remove_epochs=False, z_score=False) 

    preprocessed_pupil_pos_df['participant'] = participant_str

    preprocessed_pupil_pos_df_list.append(preprocessed_pupil_pos_df)

    # Extracting pupil diameter epochs around outcome presentations
    print(f"\nExtracting epochs around outcome presentation...") 
    outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df = pupil.all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=CONDITION_STR, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.25, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_nobc_df, outcome_blink_ratio_nobc_df, outcome_epochs_to_remove_nobc, outcome_epochs_to_remove_nobc_df = pupil.all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=CONDITION_STR, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    # Extracting pupil position epochs around outcome presentations (for future analyses)
    print(f"\nExtracting pupil position epochs around outcome presentation...")
    pos_outcome_onsets, pos_outcome_nums, outcome_epoched_x_pupil_df, outcome_epoched_y_pupil_df, pos_outcome_blink_ratio_df, pos_outcome_epochs_to_remove, pos_outcome_epochs_to_remove_df = pupil.all_outcome_pupil_position_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=CONDITION_STR, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, pupil_trig_df=pupil_trig_df, baseline_cor=[], select_eye=True, remove_epochs=True)

    # Excluding participant if number of epochs removed due to excessive interpolation exceeds 55% (Was initially defined as 50% [which is used in the supplement dataset], but 2 participants were very close to this threshold. Prioritised maximizing participant data retention, so increased threshold to 55%).
    missing_epcochs = pd.DataFrame([{
        'participant': participant_str,
        'expected_epochs': len(outcome_blink_ratio_df),
        'removed_epochs': outcome_blink_ratio_df['above_threshold'].sum(),
        'percent_epochs_retained': (len(outcome_blink_ratio_df) - outcome_blink_ratio_df['above_threshold'].sum()) / len(outcome_blink_ratio_df) * 100
    }])

    missing_epochs_check.append(missing_epcochs)
    
    percent_epochs_removed = outcome_blink_ratio_df['above_threshold'].sum() / len(outcome_blink_ratio_df) * 100
    
    if percent_epochs_removed >= 55:
        print(f"\nMore than 55% epochs removed due to excessive interpolation for {participant_str} — excluding from further pupil analysis.")
        exclude_participants_df = pd.DataFrame([{
            'exclude_participant': participant,
            'percent_epochs_removed': percent_epochs_removed
        }])
        exclude_participants.append(exclude_participants_df)
        continue

    # Processing epoch dataframes to make them aligned across participants (transposing them so time x epoch, after removing statistical)
    epoch_time_df, csv_pupil_epoch_df, columns_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_pupil_df, participant_str=participant_str)

    epoch_time_nobc_df, csv_pupil_epoch_nobc_df, columns_nobc_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_pupil_nobc_df, participant_str=participant_str)

    epoch_x_time_df, csv_pupil_x_epoch_df, columns_x_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_x_pupil_df, participant_str=participant_str)

    epoch_y_time_df, csv_pupil_y_epoch_df, columns_y_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_y_pupil_df, participant_str=participant_str)

    # Appending participant column to each dataframe and adding to lists
    outcome_epochs_time_df_list.append(epoch_time_df)
    outcome_epoched_pupil_df_list.append(csv_pupil_epoch_df)
    outcome_epochs_columns_df_list.append(columns_df)
    outcome_blink_ratio_df_list.append(outcome_blink_ratio_df)
    outcome_epochs_to_remove_df_list.append(outcome_epochs_to_remove_df)
    outcome_epochs_time_nobc_df_list.append(epoch_time_nobc_df)
    outcome_epoched_pupil_nobc_df_list.append(csv_pupil_epoch_nobc_df)
    outcome_epochs_columns_nobc_df_list.append(columns_nobc_df)
    outcome_blink_ratio_nobc_df_list.append(outcome_blink_ratio_nobc_df)
    outcome_epochs_to_remove_nobc_df_list.append(outcome_epochs_to_remove_nobc_df)
    outcome_epoched_x_pupil_df_list.append(csv_pupil_x_epoch_df)
    outcome_epoched_y_pupil_df_list.append(csv_pupil_y_epoch_df)
    outcome_pos_epochs_columns_df_list.append(columns_x_df)
    outcome_pos_blink_ratio_df_list.append(pos_outcome_blink_ratio_df)
    outcome_pos_epochs_to_remove_df_list.append(pos_outcome_epochs_to_remove_df)
    outcome_pos_epochs_time_df_list.append(epoch_x_time_df)

# Collating lists to group-level dataframes
group_level_preprocessed_pupil_df = pd.concat(preprocessed_pupil_df_list, ignore_index=True)
group_level_preprocessed_pos_pupil_df = pd.concat(preprocessed_pupil_pos_df_list, ignore_index=True)
group_level_outcome_epochs_time_df = pd.concat(outcome_epochs_time_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_df = pd.concat(outcome_epoched_pupil_df_list, ignore_index=True)
group_level_outcome_epochs_columns_df = pd.concat(outcome_epochs_columns_df_list, ignore_index=True)
group_level_outcome_blink_ratio_df = pd.concat(outcome_blink_ratio_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_df = pd.concat(outcome_epochs_to_remove_df_list, ignore_index=True)
group_level_outcome_epochs_time_nobc_df = pd.concat(outcome_epochs_time_nobc_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_nobc_df = pd.concat(outcome_epoched_pupil_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_columns_nobc_df = pd.concat(outcome_epochs_columns_nobc_df_list, ignore_index=True)
group_level_outcome_blink_ratio_nobc_df = pd.concat(outcome_blink_ratio_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_nobc_df = pd.concat(outcome_epochs_to_remove_nobc_df_list, ignore_index=True)
group_level_outcome_epoched_x_pupil_df = pd.concat(outcome_epoched_x_pupil_df_list, ignore_index=True)
group_level_outcome_epoched_y_pupil_df = pd.concat(outcome_epoched_y_pupil_df_list, ignore_index=True)
group_level_outcome_pos_epochs_columns_df = pd.concat(outcome_pos_epochs_columns_df_list, ignore_index=True)
group_level_outcome_pos_blink_ratio_df = pd.concat(outcome_pos_blink_ratio_df_list, ignore_index=True)
group_level_outcome_pos_epochs_to_remove_df = pd.concat(outcome_pos_epochs_to_remove_df_list, ignore_index=True)
group_level_outcome_pos_epochs_time_df = pd.concat(outcome_pos_epochs_time_df_list, ignore_index=True)
group_level_missing_epochs_check_df = pd.concat(missing_epochs_check, ignore_index=True)
exclude_participants_pupil_analysis_df = pd.concat(exclude_participants, ignore_index=True)

# Check correspondence between diameter and position pupil dataframes
print(f"\nChecking correspondence between diameter and position pupil dataframes...")
for participant in participant_list:
    if participant not in excluded_participants:
        print(f"\nParticipant {str(participant).zfill(2)}...")
        print("Diameter v x position data")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epoched_pupil_df, df_2=group_level_outcome_epoched_x_pupil_df, participant_id=participant, check_index=True, check_columns=True)
        print("Diameter v y position data")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epoched_pupil_df, df_2=group_level_outcome_epoched_y_pupil_df, participant_id=participant, check_index=True, check_columns=True)
        print("X position v y position data")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epoched_x_pupil_df, df_2=group_level_outcome_epoched_y_pupil_df, participant_id=participant, check_index=True, check_columns=True)
        print("Extracted epoch times")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epochs_time_df, df_2=group_level_outcome_pos_epochs_time_df, participant_id=participant, check_index=True, check_columns=True)
        print("Columns info")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epochs_columns_df, df_2=group_level_outcome_pos_epochs_columns_df, participant_id=participant, check_index=True, check_columns=True)
        print("Blink ratio info")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_blink_ratio_df, df_2=group_level_outcome_pos_blink_ratio_df, participant_id=participant, check_index=True, check_columns=True)
        print("Epochs to remove info")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epochs_to_remove_df, df_2=group_level_outcome_pos_epochs_to_remove_df, participant_id=participant, check_index=True, check_columns=True)


# Creating euclidean distance pupil position dataframe (for future analyses)
print(f"\nCreating euclidean distance pupil position dataframe...")

# Using means across whole data as individual origins
whole_data_pos_means = []

for participant in group_level_preprocessed_pupil_df['participant'].unique():
    data_x = group_level_preprocessed_pupil_df[
        group_level_preprocessed_pupil_df['participant'] == participant
    ]
    data_y = group_level_preprocessed_pupil_df[
        group_level_preprocessed_pupil_df['participant'] == participant
    ]

    mean_x = data_x['x_pos_int'].mean()
    mean_y = data_y['y_pos_int'].mean()

    whole_data_pos_means.append({
        'participant': participant,
        'mean_x': mean_x,
        'mean_y': mean_y
    })

whole_data_pos_means_df = (
    group_level_preprocessed_pupil_df
        .groupby('participant', as_index=False)
        .agg(
            mean_x=('x_pos_int', 'mean'),
            mean_y=('y_pos_int', 'mean')
        )
)        

# Merging whole data means with epoched position data to calculate deviation from mean
epoch_x = group_level_outcome_epoched_x_pupil_df.merge(
    whole_data_pos_means_df[['participant', 'mean_x']],
    on='participant',
    how='left'
)

epoch_y = group_level_outcome_epoched_y_pupil_df.merge(
    whole_data_pos_means_df[['participant', 'mean_y']],
    on='participant',
    how='left'
)

time_cols = [c for c in epoch_x.columns if c not in ['participant', 'mean_x']]

x_dev = epoch_x[time_cols].to_numpy() - epoch_x['mean_x'].to_numpy()[:, None]
y_dev = epoch_y[time_cols].to_numpy() - epoch_y['mean_y'].to_numpy()[:, None]

dist = np.sqrt(x_dev**2 + y_dev**2)

euclidean_dist_df = pd.DataFrame(dist, columns=time_cols)
euclidean_dist_df['participant'] = epoch_x['participant'].values

# Checking correspondence between diameter and euclidean distance pupil dataframes
print(f"\nChecking correspondence between diameter and euclidean distance pupil dataframes...")
for participant in participant_list:
    if participant not in excluded_participants:
        print(f"\nParticipant {str(participant).zfill(2)}...")
        print("Diameter v euclidean distance data")
        pupil_analysis.check_pupil_dfs_compatible(df_1=group_level_outcome_epoched_pupil_df, df_2=euclidean_dist_df, participant_id=participant, check_index=True, check_columns=True)


# Saving files to .csv
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'exclude_participants_pupil_analysis_df': exclude_participants_pupil_analysis_df,
    'missing_epochs_check_df': group_level_missing_epochs_check_df,
    'outcome_epochs_time_df': group_level_outcome_epochs_time_df,
    'outcome_epoched_pupil_df': group_level_outcome_epoched_pupil_df,
    'outcome_epochs_columns_df': group_level_outcome_epochs_columns_df,
    'outcome_blink_ratio_df': group_level_outcome_blink_ratio_df,
    'outcome_epochs_to_remove_df': group_level_outcome_epochs_to_remove_df,
    'outcome_epochs_time_nobc_df': group_level_outcome_epochs_time_nobc_df,
    'outcome_epoched_pupil_nobc_df': group_level_outcome_epoched_pupil_nobc_df,
    'outcome_epochs_columns_nobc_df': group_level_outcome_epochs_columns_nobc_df,
    'outcome_blink_ratio_nobc_df': group_level_outcome_blink_ratio_nobc_df,
    'outcome_epochs_to_remove_nobc_df': group_level_outcome_epochs_to_remove_nobc_df, 
    'outcome_epochs_euclidean_distance_df': euclidean_dist_df,
    },
    directory_path=BASE_DIR,
    condition_str=CONDITION_STR,
    task='Pupil')