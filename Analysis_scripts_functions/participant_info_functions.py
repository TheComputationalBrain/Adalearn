import csv
import pandas as pd
from pathlib import Path
import re



def participant_characteristics(participant, base_dir, c1_vns_task_sessions=None, c2_vns_task_sessions=None):
    """
    Function to load and extract session information for a given participant. Presents stimulator information, and file numbers that can then be passed to other functions to load the data. 

    Parameters:
    - participant: float, participant identification number
    - base_dir: string, top-level directory path.
    - c1_vns_task_sessions, c2_vns_task_sessions: list, contains numbers of task session in which VNS was active during. 

    Returns:
    - participant_str: string, zero-padded participant ID
    - list_subject_selected: list, contains participant ID
    - directory_path: Path, path to the participant level directory
    - tsv_df: dict, contains DataFrames from .tsv files
    - high_vns_condition, low_vns_condition: string, zero-padded condition identifier, specifying which condition contained the high and low VNS condition
    - condition_1_str, condition_2_str: string, zero-padded condition identifier
    - c1_from_session, c1_to_session: range of task sessions for condition 1
    - c2_from_session, c2_to_session: range of task sessions for condition 2
    - c1_run_ranges, c2_run_ranges: extracted run ranges for physiological files    
    """
    base_dir = Path(base_dir)

    participant_str = str(participant).zfill(2)
    list_subject_selected = [participant]

    directory_path = Path(f'{base_dir}/sub-{participant_str}')

    # Initializing parameters with None to ensure they exist before use
    c1_from_session, c1_to_session = None, None
    c2_from_session, c2_to_session = None, None
    c1_run_ranges, c2_run_ranges = None, None
    high_vns_condition, low_vns_condition = None, None 

    tsv_df = {}

    print(f"Participant: {participant_str}")

    # Read .tsv files
    if directory_path.exists():
        print("Checking for .tsv files...")
        for idx, tsv_file in enumerate(directory_path.rglob('*.tsv'), start=1): 
            df = pd.read_csv(tsv_file, sep='\t')
            tsv_df[idx] = df
        print(f"Found files. Reading files....")
        print(f"The number of .tsv files found was: {len(tsv_df)}")
    else:
        print(f"Path {directory_path} does not exist.")

    # Initialize condition strings
    condition_1_str, condition_2_str = None, None

    # Create condition strings
    if tsv_df:
        for key, df in tsv_df.items():
            variable_name = f"condition_{key}_str"
            globals()[variable_name] = f"{int(key):02d}"
            if key == 1 and condition_1_str is None:
                condition_1_str = globals()[variable_name]
            elif key == 2 and condition_2_str is None:
                condition_2_str = globals()[variable_name]
            print(f"Condition string created: {variable_name}")

        # Print key information about VNS parameters
        for key, df in tsv_df.items():
            print(f"For session {df['SessionNumber'].iloc[0]}, the VNS parameters were:")
            print(f"Stimulation intensity: {df['StimulationIntensity'].iloc[0]} mA")
            print(f"Stimulation frequency: {df['StimulationFrequency'].iloc[0]} Hz")
            print(f"Stimulation pulse width: {df['PulseWidth'].iloc[0]} µs")
            print(f"ON duration: {df['DurationON'].iloc[0]} s")
            print(f"OFF duration: {df['DurationOFF'].iloc[0]} s")

        # Define high vs low VNS conditions
        if tsv_df[1]['StimulationIntensity'].iloc[0] > tsv_df[2]['StimulationIntensity'].iloc[0]:
            print("The high VNS intensity condition is condition 1 and the low VNS intensity condition is condition 2")
            high_vns_condition = f"{1:02d}"
            low_vns_condition = f"{2:02d}"
        elif tsv_df[1]['StimulationIntensity'].iloc[0] < tsv_df[2]['StimulationIntensity'].iloc[0]:
            print("The low VNS intensity condition is condition 1 and the low VNS intensity condition is condition 2")
            high_vns_condition = f"{2:02d}"
            low_vns_condition = f"{1:02d}"

    # Extract task session information for condition 1
    if condition_1_str is not None:
        print("Task sessions extracted from file names in condition 1")
        print("Filenames extracted from behavioural data files:")
        c1_from_session_behav, c1_to_session_behav, c1_task_sessions_behav, c1_is_continuous_behav = extract_max_min(directory_path=directory_path, condition_str=condition_1_str, file_folder='behav', file_extension='.json')

        if c1_is_continuous_behav is False:
            print("ERROR: Continuous range of task sessions not extracted from behavioural file names")

        print("Filenames extracted from eyetracker data files:")
        c1_from_session_eye, c1_to_session_eye, c1_task_sessions_eye, c1_is_continuous_eye = extract_max_min(directory_path=directory_path, condition_str=condition_1_str, file_folder='eye', file_extension='.txt')

        if c1_is_continuous_eye is False:
            print("ERROR: Continuous range of task sessions not extracted from eyetracker file names")
        
        print("Filenames extracted from physiological data files:")
        c1_run_ranges = extract_phys_run_ranges(directory_path=directory_path, condition_str=condition_1_str)

        print(f"Run ranges in physiological files: {c1_run_ranges}")

        # Check if task sessions match and set the session range
        print("Checking if behavioural and eyetracker files contain the same task sessions...")
        if c1_task_sessions_behav == c1_task_sessions_eye:
            c1_task_sessions = c1_task_sessions_behav
            c1_from_session = c1_from_session_behav
            c1_to_session = c1_to_session_behav
            print("MATCHED: behavioural and eyetracker files have the same task sessions")
            print(f"First task session: {c1_from_session}")
            print(f"Last task session: {c1_to_session}")
        else:
            print(f"ERROR: Mismatch in task sessions extracted from files.")
            c1_from_session, c1_to_session = None, None
    
    else:
        print("No .tsv file found for condition 1")

    # Extract task session information for condition 2
    if condition_2_str is not None:
        print("Task sessions extracted from file names in condition 2")
        print("Filenames extracted from behavioural data files:")
        c2_from_session_behav, c2_to_session_behav, c2_task_sessions_behav, c2_is_continuous_behav = extract_max_min(directory_path=directory_path, condition_str=condition_2_str, file_folder='behav', file_extension='.json')

        if c2_is_continuous_behav is False:
            print("ERROR: Continuous range of task sessions not extracted from behavioural file names")

        print("Filenames extracted from eyetracker data files:")
        c2_from_session_eye, c2_to_session_eye, c2_task_sessions_eye, c2_is_continuous_eye = extract_max_min(directory_path=directory_path, condition_str=condition_2_str, file_folder='eye', file_extension='.txt')

        if c2_is_continuous_eye is False:
            print("ERROR: Continuous range of task sessions not extracted from eyetracker file names")

        print("Filenames extracted from physiological data files:")
        c2_run_ranges = extract_phys_run_ranges(directory_path=directory_path, condition_str=condition_2_str)

        print(f"Run ranges in physiological files: {c2_run_ranges}")

        # Check if task sessions match and set the session range
        print("Checking if behavioural and eyetracker files contain the same task sessions...")
        if c2_task_sessions_behav == c2_task_sessions_eye:
            c2_task_sessions = c2_task_sessions_behav
            c2_from_session = c2_from_session_behav
            c2_to_session = c2_to_session_behav
            print("MATCHED: behavioural and eyetracker files have the same task sessions")
            print(f"First task session: {c2_from_session}")
            print(f"Last task session: {c2_to_session}")
        else:
            print(f"ERROR: Mismatch in task sessions extracted from files.")
            c2_from_session, c2_to_session  = None, None
    
    else:
        print("No .tsv file found for condition 2")

    # List of task sessions containing VNS
    if c1_vns_task_sessions:
        c1_vns_task_sessions = [f"{num:02}" for num in c1_vns_task_sessions]

    if c2_vns_task_sessions:
        c2_vns_task_sessions = [f"{num:02}" for num in c2_vns_task_sessions]


    return participant_str, list_subject_selected, directory_path, tsv_df, high_vns_condition, low_vns_condition, condition_1_str, condition_2_str, c1_task_sessions, c1_from_session, c1_to_session, c2_task_sessions, c2_from_session, c2_to_session, c1_run_ranges, c2_run_ranges, c1_vns_task_sessions, c2_vns_task_sessions 




def extract_max_min(directory_path, condition_str, file_folder, file_extension):
    """
    Function to extract task session numbers from a range of file names. Checks if there is a continuous range in the file names, providing a sanity check that no files are obviously missing or been misnamed. 

    Parameters: 
    - directory_path: string, participant level directory name. 
    - condition_str: string, condition identifier (zero-padded number to two numbers)
    - file_folder: string, folder name containing files of interest
    - file_extension: string, extension  of files of interest

    Returns:
    - min_session: string, the minimum task session.
    - max_session: string, the maximum task session.
    - task_sessions: list of strings, all task sessions sorted.
    - is_continuous: boolean, identifies is the range of task sessions is a continuous sequence of numbers. 
    """
    task_sessions = []

    # Loop through each file name and extract the run range
    file_path = Path(f"{directory_path}/ses-{condition_str}/{file_folder}")
    file_extension = file_extension

    # Get a list of all files with the specified extension in the directory
    file_names = [f.name for f in file_path.iterdir() if f.is_file() and f.suffix == file_extension and not f.name.startswith("._")]
    print(file_names)

    for file_name in file_names:
        match = re.search(r"run-(\d+)", file_name)
        if match:
            task_session = match.group(1)
            task_sessions.append(task_session)

    task_sessions = sorted(task_sessions, key=int)

     # Check if the sessions are continuous by comparing as integers
    is_continuous = all(int(task_sessions[i]) + 1 == int(task_sessions[i + 1]) for i in range(len(task_sessions) - 1))

    # Return the min and max task session numbers as strings
    min_session = int(task_sessions[0]) if task_sessions else None
    max_session = int(task_sessions[-1]) if task_sessions else None

    return min_session, max_session, task_sessions, is_continuous



def extract_phys_run_ranges(directory_path, condition_str):
    """
    Function to extract run ranges from .edf file names.

    Parameters: 
    - directory_path: string, participant level directory name. 
    - condition_str: string, condition identifier (zero-padded number to two numbers)

    Returns:
    - run_ranges: list of dictionaries, with keys 'run_start' and 'run_end' containing values relating tho the first and last task session contained within that file. 
    """
    run_ranges = []

    # Loop through each file name and extract the run range
    physio_path = Path(f"{directory_path}/ses-{condition_str}/physio")
    file_extension = '.edf'

    # Get a list of all files with the specified extension in the directory
    file_names = [f.name for f in physio_path.iterdir() if f.is_file() and f.suffix == file_extension and not f.name.startswith("._")]
    print(file_names)

    for file_name in file_names:
        match = re.search(r"run-(\d+)to(\d+)", file_name)
        if match:
            run_start = match.group(1)
            run_end = match.group(2)
            run_ranges.append({'run_start': run_start, 'run_end': run_end})

    run_ranges = sorted(run_ranges, key=lambda x: int(x['run_start']))

    return run_ranges