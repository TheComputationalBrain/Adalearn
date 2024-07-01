# Module imports
import pandas as pd
import numpy as np
import mne
import os
import scipy as sc
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'


def load_biopac_data(participant_str, condition, run_ranges):
    """
    Function to load biopac .edf files. Sets the working directory to the relevant file, 
    before loading and storing the data for each task session for a given condition. 
    
    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition: string, identifier of condition order (set to 2 numbers)
    - run_ranges: list of dicts, each dict contains 'run_start' and 'run_end' denoting the range of task sessions in each file

    Returns:
    - biopac_data: list, contains the dictionaries storing the data from the adalearn files for each run of task sessions. 
    - stim_chan: string, the stimulation (emg) channel name
    - trig_chan: string, the adalearn outcome trigger channel names
    """
    # Set the working directory of the files
    base_dir = f'/home/aaron/Documents/VNS Ada_Learn/data_VNS/sub-{participant_str}/ses-{condition}'
    os.chdir(base_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Create a list to store biopac data files
    biopac_data = []

    # Create a set to store unique stim channel names
    stim_chan_set = set()

    # Create a set to store unique trig channel names
    trig_chan_set = set()

    # Iterate over each run range
    for run_range in run_ranges:
        run_start = run_range['run_start']
        run_end = run_range['run_end']
        filename = f'sub-{participant_str}_ses-{condition}_task-AdalearnVNS_run-{run_start}to{run_end}_physio.edf'
        
        if os.path.exists(filename):
            print(f"Loading file: {filename}")
            raw_biopac = mne.io.read_raw_edf(filename, preload=True)
            biopac_data.append(raw_biopac)

            # Check channel names
            print(f"Channel names in {filename}: {raw_biopac.ch_names}")
            stim_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('STIM')]
            print(f"Stim channels found: {stim_channels_in_file}")
            trig_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('Digital')]
            print(f"Trig channels found: {trig_channels_in_file}")
            stim_chan_set.update(stim_channels_in_file)
            trig_chan_set.update(trig_channels_in_file)

        else:
            print(f"File not found: {filename}")

    # Convert stim channel set to a list and take the first element
    stim_chan = list(stim_chan_set)[0] if stim_chan_set else None
    print(f"Final stim channel: {stim_chan}")

    # Convert trigger set to a list and take the third element (need to make sure that combined channel is always the third element)
    trig_set_to_list = list(trig_chan_set)
    trig_set_to_list = sorted(trig_set_to_list)
    print(trig_set_to_list)
    trig_chan = trig_set_to_list[2]
    print(f"Final trig channel: {trig_chan}")

    return biopac_data, stim_chan, trig_chan


def process_biopac_data(biopac_data, stim_channels):
    """ 
    Function to process the loaded biopac data. It creates and returns a dataframe with the emg data and (relative) timestamp (from beginning of file recording). EMG signal is rectified and enveloped. Created a variable to denote the data file that data is extracted from (data files vertically stacked in dataframe). Plots of the emg data generated from each data file to enable quality check of data. 

    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data from the adalearn files for each run of task sessions. 
    - stim_channels: string, the stim channel name (used to extract the EMG data).

    Returns:
    - emg_data_df: pd.DataFrame, contains raw EMG data, associated (relative) timestamp and data file number.  
    """
    # Objects to (1) store dataframes for each data file and (2) a counter to populate data_file variable 
    all_emg_data = [] 
    data_file_counter = 1 

    # For loop to extract EMG and time data from loaded data files
    for raw_biopac in biopac_data:
        stimdata = raw_biopac.copy().pick(picks=stim_channels)
        
        # Filter the stimulation data and extract times
        emg_extracted, times = stimdata.get_data(return_times=True)
        emg_extracted = np.squeeze(emg_extracted) # reduces data to a one dimensional array 
        
        # Creating dataframe for storage of EMG data from each data file
        emg_data = pd.DataFrame({
            'emg': emg_extracted, 
            'time': times,
            'data_file': data_file_counter  # Add the data_file variable
        })
        
        # Processing the EMG signal: rectifying
        emg_data['rectified_emg'] = abs(emg_data['emg'])

        # Processing the EMG signal: low-pass filtering (4th order Butterworth filter) and enveloping
        lowp = 5 # 5 Hz cut-off
        samp_freq = 2000
        low_pass = lowp/samp_freq
        b, a = sc.signal.butter(4, low_pass, btype='lowpass')
        emg_data['lowpass_filtered'] = filtfilt(b, a, emg_data['rectified_emg'])

        # Append the dataframe to the list (one for each datafile)
        all_emg_data.append(emg_data)
        
        # Increment the data_file counter for the next file
        data_file_counter += 1
    
    # Concatenating all dataframes into a single dataframe
    emg_data_df = pd.concat(all_emg_data, ignore_index=True)
    
    return emg_data_df


def extract_adalearn_triggers_biopac(biopac_data, trig_channel, plot=False):
    """ 
    Function to extract the adalearn outcome presentation triggers inserted into the biopac file. Extracts and creates a dataframe to store the trigger information (i.e. 0 or 1) as well as the respective timestamp. Extracts the triggers from the combined channel in the biopac file (i.e. the channel that contains triggers for both yellow and blue). Also generates a dataframe with summary statistics for each identified task session (I.e. number of triggers, duration of task session, time at first stimulation, etc)

    Parameters:
    - biopac_data: list, contains the dictionaries storing the data from the adalearn files for each run of task sessions. 
    - trig_channel: string, the combined trigger channel name (used to extract the trigger information) (*Note: need to make sure that this combined is set up. If it isn't will need to evaluate the individual channels (those set up for yellow and blue outcomes) separately and then combine).
    - plot: boolean, if True, creates subplots of identified triggers using Plotly.

    Returns:
    - identified_triggers_df: pd.DataFrame, contains trigger data, associated timestamp, and data file number. 
    - task_session_summary: pd.DataFrame, contains broad summary statistics of the characteristics of task sessions identified through triggers. 
    """
    # Objects to store dataframes for each data file and a counter to populate the data_file variable
    all_trig_data = []
    data_file_counter = 1

    # For loop to extract triggers and time data from loaded data files
    for raw_biopac in biopac_data:
        trigdata = raw_biopac.copy().pick(picks=trig_channel)
        
        # Filter the trigger data and extract times
        trig_extracted, times = trigdata.get_data(return_times=True)
        trig_extracted = np.squeeze(trig_extracted)  # reduces data to a one dimensional array
        
        # Creating dataframe for storage of trigger data from each data file
        trig_data = pd.DataFrame({
            'triggers': trig_extracted, 
            'time': times,
            'data_file': data_file_counter  # Add the data_file variable
        })

        # Append the dataframe to the list (one for each datafile)
        all_trig_data.append(trig_data)
        
        # Increment the data_file counter for the next file
        data_file_counter += 1
    
    # Concatenating all dataframes into a single dataframe
    trig_data_df = pd.concat(all_trig_data, ignore_index=True)

    # Determine the threshold for identifying triggers
    trig_thres = trig_data_df['triggers'].max() * 0.5

    # DataFrame to store identified triggers
    identified_triggers = []

    # Process each data_file separately
    for file in trig_data_df['data_file'].unique():
        file_data = trig_data_df[trig_data_df['data_file'] == file]

        # Identify initial triggers above the threshold
        initial_triggers = file_data[file_data['triggers'] >= trig_thres]

        if initial_triggers.empty:
            continue

        # Initialize the list of valid triggers with the first detected trigger
        valid_triggers = [initial_triggers.iloc[0]]

        # Iterate over the rest of the triggers to ensure they are spaced sufficiently apart
        for idx in range(1, len(initial_triggers)):
            time_diff = initial_triggers.iloc[idx]['time'] - valid_triggers[-1]['time']
            if time_diff >= 1.0:  # Trigger interval in seconds
                valid_triggers.append(initial_triggers.iloc[idx])

        # Append valid triggers to the identified_triggers list
        identified_triggers.extend(valid_triggers)

    # Convert the list of identified triggers to a DataFrame
    identified_triggers_df = pd.DataFrame(identified_triggers)
    print(f"Number of valid triggers found: {len(identified_triggers_df)}")

    # Calculate time differences between successive triggers
    identified_triggers_df['time_diff'] = identified_triggers_df['time'].diff()
    identified_triggers_df['time_diff'].iloc[0] = 0  # Set the first time difference to 0

    # Create task session variable based on multiple criteria (i.e. threshold in time difference between successive triggers and the start of a new data file.)
    task_session = 1
    task_sessions = []
    prev_data_file = None
    for index, row in identified_triggers_df.iterrows():
        if index == 0 or row['time_diff'] > 3 or row['data_file'] != prev_data_file:
            task_session += 1
        task_sessions.append(task_session)
        prev_data_file = row['data_file']
    identified_triggers_df['task_session'] = task_sessions
    identified_triggers_df['task_session'] = identified_triggers_df['task_session'] -1

    # Count the number of triggers in each task session
    task_session_counts = identified_triggers_df['task_session'].value_counts().sort_index()
    print(f"Number of triggers in each task session:\n{task_session_counts}")

    # Plotting the triggers is required for visualisation
    if plot:
        num_files = len(trig_data_df['data_file'].unique())
        fig = make_subplots(rows=num_files, cols=1, shared_xaxes=True, 
                            subplot_titles=[f"Data File {i+1}" for i in range(num_files)])
        
        for i, file in enumerate(trig_data_df['data_file'].unique()):
            file_data = trig_data_df[trig_data_df['data_file'] == file]
            identified_triggers_file = identified_triggers_df[identified_triggers_df['data_file'] == file]
            
            # Add the trace for all triggers
            fig.add_trace(go.Scatter(x=file_data['time'], y=file_data['triggers'], mode='lines', 
                                     name=f'All Triggers File {file}', line=dict(color='blue')), 
                          row=i+1, col=1)
            
            # Add the trace for identified triggers
            fig.add_trace(go.Scatter(x=identified_triggers_file['time'], y=identified_triggers_file['triggers'], 
                                     mode='markers', name=f'Identified Triggers File {file}', 
                                     marker=dict(color='red', size=10, symbol='x')), 
                          row=i+1, col=1)

        fig.update_layout(height=300*num_files, width=1000, title_text="Identified Triggers Across Data Files")
        fig.show()
    
    # Calculate descriptive statistics for each identified task session and enter within a data frame. First group data by task session 
    task_session_grouped = identified_triggers_df.groupby('task_session')


    task_session_summary = task_session_grouped.agg(
        data_file =('data_file', 'first'),
        num_triggers=('triggers', 'count'),
        first_trig=('time', 'first'),
        last_trig=('time', 'last')
    )

    task_session_summary['session_duration'] = task_session_summary['last_trig'] - task_session_summary['first_trig']

    return identified_triggers_df, task_session_summary


def extract_task_session_emg_data(emg_data_df, task_session_summary, plot=False):
    """ 
    Function to reduce the size of the data to only that within a task session. Creates a dictionary of stored dataframes housing the EMG data for each task session.
    Additionally, can plot rectified EMG against time for each task session for visualization.

    Parameters:
    - emg_data_df: pd.DataFrame, contains EMG data, associated (relative) timestamp and data file number.  
    - task_session_summary: pd.DataFrame, contains broad summary statistics of the characteristics of task sessions identified through triggers.
    - plot: boolean, if True, creates subplots for each task session of rectified EMG signal across time using Plotly. 

    Returns:
    - task_session_dict: dict, contains EMG dataframe (value) for each task session (key).
    """
    # Initialize an empty dictionary to store the dataframes
    task_session_dict = {}

    # Iterate over each row in the task_session_summary dataframe
    for index, row in task_session_summary.iterrows():
        data_file = row['data_file']
        start_time = row['first_trig']
        end_time = row['last_trig'] 
        
        # Filter the EMG dataframe to get the relevant rows
        task_data = emg_data_df[
            (emg_data_df['data_file'] == data_file) &
            (emg_data_df['time'] >= start_time) &
            (emg_data_df['time'] <= end_time)]
        
        # Store the filtered dataframe in the dictionary with the index as the key
        task_session_dict[index] = task_data

    # Plotting the EMG data stored within the dictionary if plot=True
    if plot:
        # Calculate the total number of subplots
        num_subplots = len(task_session_dict)

        # Create a figure with subplots and specify row heights
        fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=[f"Task Session {key}" for key in task_session_dict.keys()],
                            row_heights=[0.6] * num_subplots)

        # Iterate over each key (task session) and its corresponding dataframe
        for i, (key, df) in enumerate(task_session_dict.items(), start=1):
            # Add trace for rectified_emg against time
            fig.add_trace(go.Scatter(x=df['time'], y=df['rectified_emg'], mode='lines', name=f'Task Session {key}'),
                          row=i, col=1)

            # Update x-axis and y-axis titles for each subplot
            fig.update_xaxes(title_text="Time", row=i, col=1)
            fig.update_yaxes(title_text="EMG", row=i, col=1)

            # Hide the legend for each subplot
            fig.update_traces(showlegend=False)

        # Update layout
        fig.update_layout(
            title="Rectified EMG vs Time for Each Task Session",
            height=4000  # Adjust the overall height of the figure if needed
        )

        # Show the plot
        fig.show()

    return task_session_dict