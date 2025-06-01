"""

"""
import pandas as pd
import os
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_pupil import load_and_process_pupil_data, all_outcome_pupil_epochs, check_pupil_file_triggers

from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, process_epoched_df, load_sample_csvs, save_sample_dataframes_to_csv 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, from_session, to_session, task_sessions



dataframes = load_sample_csvs(directory_path=base_dir, substrings=[
    'change_point_indices_df'
])

participant_list = get_participant_numbers(directory=base_dir)

exclude_participants = []
preprocessed_pupil_df_list = [] 
pupil_trig_df_list = []
outcome_epoched_pupil_df_list = []
outcome_blink_ratio_df_list = []
outcome_epochs_to_remove_df_list = []
outcome_epochs_descriptive_stats_df_list = []
outcome_epochs_columns_df_list = []
outcome_epoched_pupil_nobc_df_list = []
outcome_blink_ratio_nobc_df_list = []
outcome_epochs_to_remove_nobc_df_list = []
outcome_epochs_descriptive_stats_nobc_df_list = []
outcome_epochs_columns_nobc_df_list = []


for participant in participant_list:
    participant_str = str(participant).zfill(2)
    print(f"\n\n\nProcessing participant {participant_str}...")
    directory_path = Path(f'{base_dir}/sub-{participant_str}')

    # Checking whether pupil data files are correct and can be processed
    print(f"\nIntial check on file quality before analysing pupil epochs...")
    sessions_to_process = [str(session).zfill(2) for session in range(from_session, to_session + 1)]
    
    problem_files = 0
    total_files = 0

    for session in sessions_to_process:
        file_dir = f"{directory_path}/ses-{condition_str}/eye"
        file_name = f"sub-{participant_str}_ses-{condition_str}_task-AdalearnPupil_run-{session}_eyetracker.txt"
        file_path = f"{file_dir}/{file_name}"

        if os.path.exists(file_path):
            total_files += 1
            too_many_msgs = check_pupil_file_triggers(file_path=file_path, threshold=5)
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

    # Loading pupil files
    print(f"\nLoading pupil data...")
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=[-0.25, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 

    preprocessed_pupil_df['participant'] = participant_str
    pupil_trig_df['participant'] = participant_str

    preprocessed_pupil_df_list.append(preprocessed_pupil_df)
    pupil_trig_df_list.append(pupil_trig_df)

    # Extracting epochs around outcome presentations
    print(f"\nExtracting epochs around outcome presentation...") 
    outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.25, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_nobc_df, outcome_blink_ratio_nobc_df, outcome_epochs_to_remove_nobc, outcome_epochs_to_remove_nobc_df = all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant_str, list_subject_selected=[participant], condition_str=condition_str, condition_task_sessions=task_sessions, from_session=from_session, to_session=to_session, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    # Excluding participant if number of epochs removed due to excessive interpolation exceeds 50%
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
    descriptive_stats_df, csv_pupil_epoch_df, columns_df = process_epoched_df(epoched_df=outcome_epoched_pupil_df, participant_str=participant_str)

    descriptive_stats_nobc_df, csv_pupil_epoch_nobc_df, columns_nobc_df = process_epoched_df(epoched_df=outcome_epoched_pupil_nobc_df, participant_str=participant_str)


    dfs = [
    descriptive_stats_df,
    csv_pupil_epoch_df,
    columns_df,
    outcome_blink_ratio_df,
    outcome_epochs_to_remove_df,
    descriptive_stats_nobc_df,
    csv_pupil_epoch_nobc_df,
    columns_nobc_df,
    outcome_blink_ratio_nobc_df,
    outcome_epochs_to_remove_nobc_df
    ]

    df_lists = [
        outcome_epochs_descriptive_stats_df_list,
        outcome_epoched_pupil_df_list,
        outcome_epochs_columns_df_list,
        outcome_blink_ratio_df_list,
        outcome_epochs_to_remove_df_list,
        outcome_epochs_descriptive_stats_nobc_df_list,
        outcome_epoched_pupil_nobc_df_list,
        outcome_epochs_columns_nobc_df_list,
        outcome_blink_ratio_nobc_df_list,
        outcome_epochs_to_remove_nobc_df_list,
    ]

    for df, df_list in zip(dfs, df_lists):
        df['participant'] = participant_str
        df_list.append(df)


# Collating lists to group-level dataframes
group_level_outcome_epochs_descriptive_stats_df = pd.concat(outcome_epochs_descriptive_stats_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_df = pd.concat(outcome_epoched_pupil_df_list, ignore_index=True)
group_level_outcome_epochs_columns_df = pd.concat(outcome_epochs_columns_df_list, ignore_index=True)
group_level_outcome_blink_ratio_df = pd.concat(outcome_blink_ratio_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_df = pd.concat(outcome_epochs_to_remove_df_list, ignore_index=True)
group_level_outcome_epochs_descriptive_stats_nobc_df = pd.concat(outcome_epochs_descriptive_stats_nobc_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_nobc_df = pd.concat(outcome_epoched_pupil_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_columns_nobc_df = pd.concat(outcome_epochs_columns_nobc_df_list, ignore_index=True)
group_level_outcome_blink_ratio_nobc_df = pd.concat(outcome_blink_ratio_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_nobc_df = pd.concat(outcome_epochs_to_remove_nobc_df_list, ignore_index=True)
exclude_participants_pupil_analysis_df = pd.concat(exclude_participants, ignore_index=True)

# Saving files to .csv
save_sample_dataframes_to_csv(dataframes={
    'exclude_participants_pupil_analysis_df': exclude_participants_pupil_analysis_df,
    'outcome_epochs_descriptive_stats_df': group_level_outcome_epochs_descriptive_stats_df,
    'outcome_epoched_pupil_df': group_level_outcome_epoched_pupil_df,
    'outcome_epochs_columns_df': group_level_outcome_epochs_columns_df,
    'outcome_blink_ratio_df': group_level_outcome_blink_ratio_df,
    'outcome_epochs_to_remove_df': group_level_outcome_epochs_to_remove_df,
    'outcome_epochs_descriptive_stats_nobc_df': group_level_outcome_epochs_descriptive_stats_nobc_df,
    'outcome_epoched_pupil_nobc_df': group_level_outcome_epoched_pupil_nobc_df,
    'outcome_epochs_columns_nobc_df': group_level_outcome_epochs_columns_nobc_df,
    'outcome_blink_ratio_nobc_df': group_level_outcome_blink_ratio_nobc_df,
    'outcome_epochs_to_remove_nobc_df': group_level_outcome_epochs_to_remove_nobc_df
    },
    directory_path=base_dir,
    condition_str=condition_str,
    task='Pupil')