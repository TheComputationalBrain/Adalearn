import os
import re
import pandas as pd
from collections import defaultdict



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




def load_sample_csvs(directory_path, substrings):
    """
    Loads sample level .csv files whose filenames contain any of the given substrings.

    Parameters:
    - directory_path: string/Path, base directory where the files are stored.
    - substrings (list): List of substrings to search for in filenames.

    Returns:
    - loaded_dataframes: Dictionary, contains the loaded DataFrames with keys matching the matched filenames (without .csv).
    """
    load_dir = os.path.join(directory_path, "group_csv_files")
    loaded_dataframes = {}
    match_counts = defaultdict(int)

    if not os.path.exists(load_dir):
        print(f"Directory does not exist: {load_dir}")
        return loaded_dataframes

    substrings = sorted(substrings, key=len, reverse=True)  # sort by descending length

    for file in os.listdir(load_dir):
        if file.endswith('.csv'):
            matched = False
            for sub in substrings:
                if sub in file:
                    file_path = os.path.join(load_dir, file)
                    df = pd.read_csv(file_path)

                    # Track multiple matches for same substring
                    match_counts[sub] += 1
                    suffix = f"_{match_counts[sub]}" if match_counts[sub] > 1 else ""
                    key = f"{sub}{suffix}"

                    loaded_dataframes[key] = df
                    print(f"Loaded {key} from {file_path}")
                    matched = True
                    break  # stop after the first matching (longest) substring


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
    descriptive_stats_df = pd.DataFrame({
        'mean': epoched_df['mean'],
        'std': epoched_df['std'],
        'sem': epoched_df['sem'],
        'time': epoched_df['time'],
        'participant': participant_str
    }) 

    # Drop statistical columns and transpose
    csv_pupil_epoch_df = epoched_df.drop(columns=['mean', 'std', 'sem', 'time']).copy()
    columns_df = pd.DataFrame({
        'columns': csv_pupil_epoch_df.columns,
        'participant': participant_str
    })

    csv_pupil_epoch_df = csv_pupil_epoch_df.T
    csv_pupil_epoch_df['participant'] = participant_str

    return descriptive_stats_df, csv_pupil_epoch_df, columns_df