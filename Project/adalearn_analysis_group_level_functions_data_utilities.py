import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


import adalearn_analysis_participant_level_functions_behav as behav



def get_participant_numbers(directory):
    """
    Function to obtain the participant identifiers within the directory containing the data. 

    Parameters:
    - directory: Path or string, defining the directory containing the individual participant data folders.

    Returns:
    - participant_numbers: list, contains the particpant identifiers extracted from the particpant level folders in the data directory.
    """
    participant_numbers = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            match = re.match(r"sub-(\d+)", name)
            if match:
                participant_numbers.append(int(match.group(1)))

    return sorted(participant_numbers)



def check_behavioural_responses(participant_list, base_dir, condition_str, task_sessions, update_percent = 10, update_change = 0.01, estimate_range = 0.5, exclusion_cor_threshold = 0.1):
    """
    Function to quickly examing participant performance. Will exclude participants if correlation between estimates and model posterior probability is too low (indicating that there was some struggle to understand the task correctly). Participants are identified to be examined if they meet some criteria (thresholds for these criteria can be adjusted):
    - low update percentage across task sessions
    - low update change across task sessions
    - low range of values used in estimation

    After identifying potential problematic participants, correlation is performed. If coefficient is below a set threshold (can be adjusted), participant will be identified to be excluded. 

    Parameters:
    - participant_list: list, contains the identifiers of participants in the dataset
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - base_path: Path, path to root level directory
    - task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in the condition
    - update_percent: int, average percentage of updates per task session deemed too low
    - update_change: int, average outcome_by_outcome chnage in participants' estimates that is deemed too low
    - estimate_range: int, the range of estimation space used by participants deemed too low
    - exclusion_cor_threshold: int, threshold defining whether the correlation between participants' estimates and model posterior probability is too low, indicating poor understanding of the task.

    Returns:
    - exluded_participants: list, identifiers of participants demonstrating poor performance
    - identified_participants_model_est_cor_df: pd.DataFrame, contains the correlation coefficients for identified participants
    - update_df: pd.DataFrame, contains criteria data used to initially identify participants
    """
    update_eval_list = []
    identified_participants_model_est_cor_list = []

    for participant in participant_list:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{base_dir}/data/sub-{participant_str}')

        print(f"\nLoading task data and computing task performance...")

        behavioural_df, _ = behav.load_behavioural_data( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

        # 1. check the range of estimates
        behavioural_min = behavioural_df['estimate'].min()
        behavioural_max = behavioural_df['estimate'].max()
        behavioural_range = behavioural_max - behavioural_min

        # 2. Check the number of updates/update frequency and 3. the amount of update
        # Extracting the update proportion for each task session
        participant_update_counts = []
        participant_update_percent =[]
        participant_avg_update = []

        for session in behavioural_df['task_session'].unique():
            update_data = behavioural_df[behavioural_df['task_session'] == session].copy()
            update_data['estimate_update'] = update_data['estimate'].diff()
            update_data['was_updated_estimate'] = update_data['estimate_update'] > 0

            sum_update = update_data['was_updated_estimate'].sum()
            avg_update = abs(update_data['estimate_update']).mean()

            participant_update_counts.append(sum_update)
            participant_update_percent.append(sum_update / len(update_data) * 100)
            participant_avg_update.append(avg_update)

        update_eval = pd.DataFrame({
            'participant': participant,
            'avg_update_count': [np.mean(participant_update_counts)],
            'avg_update_percent': [np.mean(participant_update_percent)],
            'avg_update_change': [np.mean(participant_avg_update)],
            'estimate_min': [behavioural_min],
            'estimate_max': [behavioural_max],
            'estimate_range': [behavioural_range]
        })

        update_eval_list.append(update_eval)

    update_df = pd.concat(update_eval_list, ignore_index=True)

    # Identify low-quality participants
    low_quality_mask = (
        (update_df['avg_update_percent'] < update_percent) |
        (update_df['avg_update_change'] < update_change) |
        (update_df['estimate_range'] < estimate_range)
    )

    # Extract participant IDs
    identified_participants = update_df.loc[low_quality_mask, 'participant'].tolist()

    # Run correlation between identified participants' estimates and the model 
    for participant in identified_participants:
        directory_path = Path(f'{base_dir}/data/sub-{participant}')

        behavioural_df, _ = behav.load_behavioural_data( participant_str=participant, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

        behav_v_model_corr = behavioural_df['estimate'].corr(behavioural_df['ideal_observer'])

        behav_v_model_estimate_df = pd.DataFrame({
            'participant': [participant],
            'correlation': [behav_v_model_corr]
        })

        identified_participants_model_est_cor_list.append(behav_v_model_estimate_df)

    # Only compute correlation and concat if any participants were identified
    if identified_participants_model_est_cor_list:
        identified_participants_model_est_cor_df = pd.concat(
            identified_participants_model_est_cor_list, ignore_index=True
        )

        # Define correlation threshold for participant exclusion
        low_correlation_mask = identified_participants_model_est_cor_df['correlation'] < exclusion_cor_threshold
        excluded_participants = identified_participants_model_est_cor_df.loc[low_correlation_mask, 'participant'].tolist()

        print(f"\nExcluded participants (low quality behavioural data leading to low correlation): {excluded_participants}")

    else:
        # If no low-quality participants were found, return empty results
        identified_participants_model_est_cor_df = pd.DataFrame(columns=['participant', 'correlation'])
        excluded_participants = []

        print(f'No participants excluded based on behavioural performance')

    return excluded_participants, identified_participants_model_est_cor_df, update_df



def load_sample_csvs(directory_path, substrings, prefix):
    """
    Load .csv files whose filename stem *after* the given prefix exactly matches
    one of the substrings.

    Parameters:
    - directory_path: str or Path to directory containing files
    - substrings: list of target substrings to match (the part after prefix)
    - prefix: string prefix all filenames share before the substring

    Returns:
    - dict of DataFrames keyed by substring
    """
    load_dir = os.path.join(directory_path, "group_csv_files")
    loaded_dataframes = {}
    match_counts = defaultdict(int)

    if not os.path.exists(load_dir):
        print(f"Directory does not exist: {load_dir}")
        return loaded_dataframes

    for file in os.listdir(load_dir):
        if not file.endswith('.csv'):
            continue

        if not file.startswith(prefix):
            continue

        # Extract substring between prefix and '.csv'
        substring_part = file[len(prefix):-4]  # remove prefix and '.csv'

        if substring_part in substrings:
            file_path = os.path.join(load_dir, file)
            df = pd.read_csv(file_path)

            # Handle duplicate substring keys (optional)
            match_counts[substring_part] += 1
            suffix = f"_{match_counts[substring_part]}" if match_counts[substring_part] > 1 else ""
            key = f"{substring_part}{suffix}"

            loaded_dataframes[key] = df
            print(f"Loaded {key} from {file_path}")

    return loaded_dataframes




def save_sample_dataframes_to_csv(dataframes, directory_path, condition_str, task):
    """
    Saves each DataFrame in `dataframes` to a CSV file in the base directory.

    Parameters:
    - dataframes: dictionary, where keys are filenames and values are pandas DataFrames.
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - task: string, defines the task being analysed
    """
    save_dir = os.path.join(directory_path, "group_csv_files")
    os.makedirs(save_dir, exist_ok=True)

    for name, df in dataframes.items():
        if isinstance(df, dict):
            print(f"Skipping {name} (dict object)")
            continue
        if not isinstance(df, pd.DataFrame):
            print(f"Skipping {name} (not a DataFrame)")
            continue
        file_path = os.path.join(save_dir, f"sample_level_analysis-ses_{condition_str}-task_Adalearn_{task}-{name}.csv")
        df.to_csv(file_path, index=False)

        print(f"Saved {name} to {file_path}")



def process_epoched_df(epoched_df, participant_str):
    """
    Function to process the created epoched dataframe to make it ready for saving as a .csv file.

    Parameters:
    - epoched_df: pd.DataFrame, contains the epoched pupil data with epochs as columns.
    - participant_str: string, identifier of participant number

    Returns:
    - descriptive_stats_df: pd.DataFrame, contains the columns containing the descriptve statistics (e.g. mean, std, sem, time) for each point within the epoch
    - csv_pupil_epoch_df: pd.DataFrame, contains just the pupil epochs, with the dataframe transposed so that time is on the x-axis (and therefor common across all participants) and epochs as each row.
    - columns_df: pd.DataFrame, contains the columm headers and therefore the epoch numbers of the surviving epochs to be analysed for each participant
    """
    # Create descriptive stats DataFrame
    epoch_time_df = pd.DataFrame({
        'time': epoched_df['time'],
        'participant': participant_str
    }) 

    # Drop statistical columns and transpose
    csv_pupil_epoch_df = epoched_df.drop(columns=['time']).copy()
    columns_df = pd.DataFrame({
        'columns': csv_pupil_epoch_df.columns,
        'participant': participant_str
    })

    csv_pupil_epoch_df = csv_pupil_epoch_df.T
    csv_pupil_epoch_df['participant'] = participant_str

    return epoch_time_df, csv_pupil_epoch_df, columns_df



def results_dict_to_df(results_dict):
    records = []
    for (x_key, y_key), (coef, raw_p, fdr_p, reject) in results_dict.items():
        records.append({
            "x_var": x_key,
            "y_var": y_key,
            "correlation": coef,
            "pval_raw": raw_p,
            "pval_fdr": fdr_p,
            "significant_fdr": reject
        })
    return pd.DataFrame(records)