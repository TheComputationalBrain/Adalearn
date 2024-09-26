# Module imports
import pandas as pd
import numpy as np
import mne
import os
from itertools import combinations
import scipy as sc
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks
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
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - stim_chan: string, the stimulation (emg) channel name
    - trig_chan: string, the adalearn outcome trigger channel names
    """
    # Set the working directory of the files
    base_dir = f'/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition}/physio'
    os.chdir(base_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Create a list to store biopac data files
    biopac_data = []

    # Create a set to store unique stim channel names
    stim_chan_set = set()

    # Create a set to store unique ECG channel names
    ecg_chan_set = set()

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
            stim_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('EMG') or ch.startswith('STIM')]
            print(f"Stim channels found: {stim_channels_in_file}")
            ecg_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('ECG')]
            print(f"ECG channels found: {ecg_channels_in_file}")
            trig_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('Digital')]
            print(f"Trig channels found: {trig_channels_in_file}")
            stim_chan_set.update(stim_channels_in_file)
            ecg_chan_set.update(ecg_channels_in_file)
            trig_chan_set.update(trig_channels_in_file)

        else:
            print(f"File not found: {filename}")

    # Convert stim (EMG) channel set to a list and take the first element
    stim_chan = list(stim_chan_set)[0] if stim_chan_set else None
    print(f"Final stim channel: {stim_chan}")

    # Convert ECG channel set to a list and take the first element
    ecg_chan = list(ecg_chan_set)[0] if ecg_chan_set else None
    print(f"Final ECG channel: {ecg_chan}")

    # Convert trigger set to a list and take the third element (need to make sure that combined channel is always the third element)
    trig_set_to_list = list(trig_chan_set)
    trig_set_to_list = sorted(trig_set_to_list)
    print(trig_set_to_list)
    trig_chan = trig_set_to_list[2]
    print(f"Final trig channel: {trig_chan}")

    return biopac_data, stim_chan, ecg_chan, trig_chan


def process_emg_data(biopac_data, stim_channels, emg_by_file_plot = False):
    """ 
    Function to process the loaded emg data. It creates and returns a dataframe with the emg data and (relative) timestamp (from beginning of file recording). EMG signal is rectified and enveloped. Plots of the emg data are generated from each data file to enable quality check of data.

    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - stim_channels: string, the stim channel name (used to extract the EMG data).
    - emg_by_file_Plot: boolean, if True, plots the emg signal for each file as subplots (default is set to False as uses lots of memory.)

    Returns:
    - emg_data_df: pd.DataFrame, contains EMG data, associated (relative) timestamp and data file number.  
    """
    # Objects to (1) store dataframes for each data file and (2) a counter to populate data_file variable 
    all_emg_data = [] 
    data_file_counter = 1 

    # For loop to extract EMG and time data from loaded data files
    for raw_biopac in biopac_data:
        stimdata = raw_biopac.copy().pick(picks=stim_channels)
        
        # Filter the stimulation data and extract times
        emg_extracted, times = stimdata.get_data(return_times=True)
        emg_extracted = np.squeeze(emg_extracted)  # Reduces data to a one-dimensional array 
        
        # Creating dataframe for storage of EMG data from each data file
        emg_data = pd.DataFrame({
            'emg': emg_extracted, 
            'time': times,
            'data_file': data_file_counter  # Add the data_file variable
        })
        
        # Processing the EMG signal: rectifying
        emg_data['rectified_emg'] = abs(emg_data['emg'])

        # Processing the EMG signal: low-pass filtering (4th order Butterworth filter) and enveloping
        lowp = 5  # 5 Hz cut-off
        samp_freq = 2000
        low_pass = lowp / samp_freq
        b, a = sc.signal.butter(4, low_pass, btype='lowpass')
        emg_data['lowpass_filtered'] = filtfilt(b, a, emg_data['rectified_emg'])

        # Append the dataframe to the list (one for each datafile)
        all_emg_data.append(emg_data)

        # Increment the data_file counter for the next file
        data_file_counter += 1

    # Concatenating all dataframes into a single dataframe
    emg_data_df = pd.concat(all_emg_data, ignore_index=True)
    
    # Plotting with Plotly: one subplot per data file
    if emg_by_file_plot:
        fig = make_subplots(rows=len(all_emg_data), cols=1, shared_xaxes=True, 
                        subplot_titles=[f'File {i+1}' for i in range(len(all_emg_data))])

        # Add traces (one for each file)
        for idx, emg_data in enumerate(all_emg_data):
            fig.add_trace(
                go.Scatter(x=emg_data['time'], y=emg_data['emg'], mode='lines',
                       name=f'File {idx + 1}'),
                row=idx + 1, col=1
            )

        # Update layout to show the plots stacked
        fig.update_layout(height=300 * len(all_emg_data), title_text='EMG Data Across Files',
                      xaxis_title='Time (s)', yaxis_title='EMG')

        # Display the plot in the browser
        fig.show()

    return emg_data_df


def integrate_separate_trigger_channels(biopac_data, run_ranges):
    """
    Function to assess and extract the trigger data from the bipoac file when one trigger channel is not integrating the two different types of trigger (0 or 1; blue or yellow) and so the triggers have to be extracted from two separate channels. Takes in the raw biopac files. Identifies each channel associated with triggers in each file. Derives the number of triggers in each file and compares this number to the number of triggers that would be expected based on the number of triggers within each task session multiplied by the number of task sessions that should be in that file (i.e. run range). If a single channel does not contain the ciorrect number of triggers, it looks for two separate channels which when summed does equate to the the expected number of triggers. Once defined, collates triggers across the two channels within each data file into a single variable, before separating and characterising task sessions within each file. 
    
    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - run_ranges: list of dicts, each dictionary contains 'run_start' and 'run_end' denoting the range of task sessions in each file

    Returns:
    - combined_chan_triggers: pd.Dataframe, contains the integrated triggers derived from the two channels (or single channel if found). Contains the source channel and file, plus derived task session based on time difference between triggers. 
    - task_session_summary_df: pd.Dataframe, contains summary statistics of each calculated task session (time of first trigger, time of last trigger, duration of task session, number of triggers within each session). 
    -  """
    
    # Store results for each file within biopac_data
    integrated_triggers_df = []  

    for data_file_count, (raw_biopac, run_range) in enumerate(zip(biopac_data, run_ranges), start=1):
        print(f"Channel names in file {data_file_count}: {raw_biopac.ch_names}")

        # Identify the names of the trigger channels
        trig_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('Digital')]
        print(f"Trig channels found: {trig_channels_in_file}")

        # Get filename or manually pass it along (not really used anymore, but could be added to trig_df)
        filename = raw_biopac.filenames[0] if hasattr(raw_biopac, 'filenames') else f'unknown_file_{data_file_count}'
        print(f"Processing file: {filename}")

        # Process and store trigger data (amplitude) and associated times for each channel in the datafile. 
        trigger_channel_data = []
        for channel in trig_channels_in_file:
            trigdata = raw_biopac.copy().pick(picks=channel)
            trig_extracted, times = trigdata.get_data(return_times=True)
            trig_extracted = np.squeeze(trig_extracted)

            trig_df = pd.DataFrame({
                'triggers': trig_extracted, 
                'time': times,
                'channel': channel, 
                'file': data_file_count
            })
            print(f"Created trigger data for channel {channel} with shape {trig_df.shape}")
            trigger_channel_data.append(trig_df)

        # Concatenate the trigger data for all channels in the current file
        all_trig_chan_df = pd.concat(trigger_channel_data, ignore_index=True)

        # Identifying triggers and their timestamps in each channel per file
        trig_thres = all_trig_chan_df['triggers'].max() * 0.5
        identified_triggers = []
        trigger_count_per_channel = {}

        for channel in all_trig_chan_df['channel'].unique():
            channel_data = all_trig_chan_df[all_trig_chan_df['channel'] == channel]
            initial_triggers = channel_data[channel_data['triggers'] >= trig_thres]

            if initial_triggers.empty:
                trigger_count_per_channel[channel] = 0
                continue

            valid_triggers = [initial_triggers.iloc[0]]
            for idx in range(1, len(initial_triggers)):
                time_diff = initial_triggers.iloc[idx]['time'] - valid_triggers[-1]['time']
                if time_diff >= 1.0: # This time difference can be changed if necessary. Difference between triggers should be ~1.5s, but may be lower for start/end triggers.
                    valid_triggers.append(initial_triggers.iloc[idx])

            identified_triggers.extend(valid_triggers)
            trigger_count_per_channel[channel] = len(valid_triggers)

        identified_triggers_df = pd.DataFrame(identified_triggers)

        if identified_triggers_df.empty:
            print(f"No valid triggers identified for file: {filename}")
        else:
            print(f"Identified triggers shape: {identified_triggers_df.shape}")
            print(f"Columns in identified triggers dataframe: {identified_triggers_df.columns}")

            # Calculate expected triggers based on run_range (should be 77 time the number of task sessions within the file)
            expected_trigs = 77 * (int(run_range['run_end']) - int(run_range['run_start']) + 1)
            print(f"The number of triggers expected for {raw_biopac} is: {expected_trigs}")

            # Check for a single channel with the correct number of triggers. This channel should then be used to extract trigger information. 
            single_channel_match_found = False

            channel_pairs = []

            pair_match_found = False

            for channel, count in trigger_count_per_channel.items():
                if count == expected_trigs:
                    print(f"{channel}: Correct number of triggers found. Use {channel} for trig_channel.")
                    integrated_triggers_df.append(identified_triggers_df[identified_triggers_df['channel'] == channel])
                    single_channel_match_found = True
                    continue

            # If a single channel with the correct number of expected triggers is not found, look for a combination of two separate channels that do (the two separate channels represent the two colours respectively)
            if not single_channel_match_found:
                channel_pairs = list(combinations(trigger_count_per_channel.keys(), 2))
                pair_match_found = False

            # Check all pairs of channels
            for pair in channel_pairs:
                sum_triggers = trigger_count_per_channel[pair[0]] + trigger_count_per_channel[pair[1]]
                print(f"Sum of triggers for channels {pair[0]} and {pair[1]}: {sum_triggers}")
        
                # Check if the sum matches the expected number of triggers
                if sum_triggers == expected_trigs:
                    print(f"Identified channels that need to be merged: {pair[0]} and {pair[1]}")
                    integrated_triggers_df.append(
                        identified_triggers_df[identified_triggers_df['channel'].isin([pair[0], pair[1]])]
                    )
                    pair_match_found = True

                    # Integrate the triggers from both of the identified channels
                    channel_1 = identified_triggers_df[identified_triggers_df['channel'] == pair[0]]
                    channel_2 = identified_triggers_df[identified_triggers_df['channel'] == pair[1]]

                    integrated_trigger_channel = pd.merge(channel_1, channel_2, on='time', how='outer')
                    integrated_trigger_channel['channel'] = integrated_trigger_channel['channel_x'].combine_first(integrated_trigger_channel['channel_y'])
                    integrated_trigger_channel['triggers'] = integrated_trigger_channel['triggers_x'].combine_first(integrated_trigger_channel['triggers_y'])
                    integrated_trigger_channel['file'] = integrated_trigger_channel['file_x'].combine_first(integrated_trigger_channel['file_y'])
                    integrated_trigger_channel = integrated_trigger_channel.drop(columns=['channel_x', 'channel_y', 'triggers_x', 'triggers_y', 'file_x', 'file_y'])

                    integrated_triggers_df.append(integrated_trigger_channel)
   
            # If no matching pair is found, raise an error
            if not pair_match_found:
                print(f"No combination of channels matches the expected number of triggers ({expected_trigs}) for {raw_biopac}.")

   
    # Concatenate all triggers across data files
    combined_chan_triggers = pd.concat(integrated_triggers_df, ignore_index=True)

    # Sort by time and reset index
    combined_chan_triggers.sort_values(by=['file', 'time'], inplace=True)
    combined_chan_triggers.reset_index(drop=True, inplace=True)

    # Create time_diff to act as criterion for separating task sessions. Again, difference greater of 1.5 s (here 3 s used) is the criterion as that is the time interval between stimuli. 
    combined_chan_triggers['time_diff'] = combined_chan_triggers['time'].diff().fillna(0)

    # Create task session variable based on criteria
    task_session = 0 
    task_sessions = []
    prev_data_file = None
    for index, row in combined_chan_triggers.iterrows():
        if row['time_diff'] > 3 or row['file'] != prev_data_file:
            task_session += 1
        task_sessions.append(task_session)
        prev_data_file = row['file']  # Ensure this column exists before using it

    combined_chan_triggers['task_session'] = task_sessions

    combined_chan_triggers.drop_duplicates(subset=['time'], inplace=True)
    combined_chan_triggers.reset_index()

    # Create a list to store task session information
    task_session_summary = []

    # Loop through each unique task session
    for session in combined_chan_triggers['task_session'].unique():
        # Filter the data for the current session
        session_data = combined_chan_triggers[combined_chan_triggers['task_session'] == session]
    
        # Get the first and last trigger times in the session
        first_trig = session_data['time'].min()
        last_trig = session_data['time'].max()
    
        # Count the number of triggers in the session
        num_triggers = session_data.shape[0]  

        # Identify the file in which the session occurs
        data_file = session_data['file'].iloc[0]
    
        # Calculate the session duration
        session_duration = last_trig - first_trig
    
        # Append a dictionary with the session summary
        task_session_summary.append({
            'task_session': session,
            'data_file': data_file,
            'first_trig': first_trig,
            'last_trig': last_trig,
            'num_triggers': num_triggers,
            'session_duration': session_duration
        })

    # Convert the list of dictionaries into a DataFrame
    task_session_summary_df = pd.DataFrame(task_session_summary)

    # Return the combined triggers and task session summary
    return combined_chan_triggers, task_session_summary_df


def extract_adalearn_triggers_biopac(biopac_data, trig_channel, plot=False):
    """ 
    Function to extract the adalearn outcome presentation triggers inserted into the biopac file. Extracts and creates a dataframe to store the trigger information (i.e. 0 or 1) as well as the respective timestamp. Extracts the triggers from the combined channel in the biopac file (i.e. the channel that contains triggers for both yellow and blue). Also generates a dataframe with summary statistics for each identified task session (i.e. number of triggers, duration of task session, time at first stimulation, etc)

    Parameters:
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
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
    trig_thres = trig_data_df['triggers'].max() * 0.8

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
    Function to reduce the size of the data to only that within a task session. Creates a dictionary of stored dataframes housing the EMG data for each task session. Also creates a relative timestamp for each task session. Can plot rectified EMG against time for each task session for visualization.

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
            (emg_data_df['time'] <= end_time)].copy()  
        
        # Calculate relative time starting from the beginning of each task session
        task_data['rel_time'] = np.arange(0, len(task_data)) * 0.0005
        
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
            fig.add_trace(go.Scatter(x=df['rel_time'], y=df['rectified_emg'], mode='lines', name=f'Task Session {key}'),
                          row=i, col=1)

            # Update x-axis and y-axis titles for each subplot
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
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


def identify_vns_from_emg(task_session_dict, vns_task_session_keys, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 2, threshold_increase = 1.2):
    """ 
    Function to identify the start and end of VNS activation from emg recording. EMG signal is noisey and attempts to use conventional methods to extract the enevelope of the signal were unfruitful (e.g. low-pass filtering, RMS, hilbert transformation; low-pass filtering destroys the signal). Opted for a very simple approach which identifies peaks within the signal to create a new "signal" by averaging around peaks. Identification of the start of VNS activation define by arbitary increase in the signal (here, 20% increase from the average of a number of points preceding a given peak). Function plots the data for visual inspection and returns the identified relative start time of VNS activation. 

    Parameters:
    - task_session_dict: dict, contains EMG dataframe (value) for each task session (key).
    - vns_task_session_keys: list, contains numbers of task sessions that VNS activation was present in (pre-defined).
    - sampling_rate: int, sampling rate of the data acquisition (2000 Hz)
    - peak_window_size: int, defines the size (in seconds) of the window used to find peaks in the signal (default set to 0.5 s). 
    - average_window_size: int, defines the size (in seconds) of half the window used to smooth the data by creating an average around identified peaks (default set to 2 s).
    - threshold_increase: int, defines the (relative) magnitude of increase used to identify the start of VNS activation beyond preceding average (default set to +20%).

    Returns:
    - vns_start_times: pd.DataFrame, contains variables identifying task session, the relative index of the identified start and the timestamp of that identified start point  
    """
    # Setting the figure subplot parameters
    fig = make_subplots(rows=len(vns_task_session_keys), cols=1, subplot_titles=[f"Task session: {key}" for key in vns_task_session_keys], shared_xaxes=True, vertical_spacing=0.01)

    row_heights = [8000, 300, 500]
    
    # Creating an empty list to store the identified VNS start times
    vns_activation = []

    # For each key identifying a stored dataframe (task_sess_df), creates a new dataframe (data) housing the rectified_emg and rel_time. 
    for idx, key in enumerate(vns_task_session_keys):
        data = task_session_dict[key]
        rectified_emg = data['rectified_emg']
        rel_time = data['rel_time']

        task_sess_df = pd.DataFrame({'rel_time': rel_time, 'rectified_emg': rectified_emg})

        # Identifying peaks in rectified emg signal
        # Calculate the number of samples in the window used to find peaks using the window size and the sampling rate 
        window_samples = int(peak_window_size * sampling_rate)

        # Create empty lists to store the peak amplitudes and corresponding timestamps
        new_peaks = []
        new_times = []

        # For loop that iterates across the length of the rectified emg signal in chunks based on the size of the window sample
        for start in range(0, len(task_sess_df['rectified_emg']) - window_samples, window_samples):
            end = start + window_samples

            # Extract segments of the rectified emg signal and corresponding time values for the current chunk specified by start and end.
            segment = task_sess_df['rectified_emg'].iloc[start:end]
            segment_time = task_sess_df['rel_time'].iloc[start:end]

            # Checks if there are missing data before using the find_peaks function that identifies peaks in the segment array.
            if not np.isnan(segment).all():
                peaks, _ = find_peaks(segment)

                # If peaks identified (i.e. len(peaks) > 0), the greatest peak within the segment is identified. Global_peak_idx calculates the global index of this peak in the original rectified emg signal.
                if len(peaks) > 0:
                    max_peak_idx = np.argmax(segment.iloc[peaks])
                    global_peak_idx = start + peaks[max_peak_idx]

                    # The global index is used to append the amplitude of the signal as well as the corresponding time point. 
                    new_peaks.append(task_sess_df['rectified_emg'].iloc[global_peak_idx])
                    new_times.append(task_sess_df['rel_time'].iloc[global_peak_idx])

        # Converting lists to np.array
        new_peaks = np.array(new_peaks)
        new_times = np.array(new_times)

        # Creating dataframe to store the peaks and times
        identified_peaks = pd.DataFrame({'peak_amplitude': new_peaks, 'peak_time': new_times})


        # Smoothing data through averaging around identified peaks
        # Creating empty list to store the averaged peak amplitudes and corresponding timestamp
        average_window_samples = int(average_window_size * sampling_rate)
        smoothed_peaks = []
        smoothed_times = []

        # Creating a variable called half window variable which defines the size of the window used to average the data across each side of the identified peak 
        half_window = average_window_size / 1

        # For loop that iterates over the identifed peaks to calculate the mean amplitude and corresponding time. Appends the data to created lists.
        for i in range(len(identified_peaks['peak_time'])):
            window_start = identified_peaks['peak_time'].iloc[i] - half_window
            window_end = identified_peaks['peak_time'].iloc[i] + half_window

            mask = (identified_peaks['peak_time'] >= window_start) & (identified_peaks['peak_time'] <= window_end)
            if np.any(mask):
                smoothed_value = np.nanmean(identified_peaks['peak_amplitude'][mask])
                smoothed_peaks.append(smoothed_value)
                smoothed_times.append(identified_peaks['peak_time'].iloc[i])

        # Converts averaged data from lists to np.arrays
        smoothed_peaks = np.array(smoothed_peaks)
        smoothed_times = np.array(smoothed_times)

        # Genrates dataframe from np.arrays of averaged data and timestamps
        smoothed_averaged_peaks = pd.DataFrame({'peaks': smoothed_peaks, 'times': smoothed_times})


        # Identifying VNS activation based on threshold criterion
        # Initialize a placeholder variable for the index of the first point meeting the criterion (currently ste to none)
        first_increase_index = None

        # Starting from the 6th element and moving across the length of the averaged peaks, calculate the average amplitude of peaks across the previous 10 data points/elements
        for i in range(5, len(smoothed_averaged_peaks['peaks'])):
            previous_avg = np.mean(smoothed_averaged_peaks['peaks'][i-10:i])

            # Argument that is used to identify VNS start criterion (i.e. increase of 20% above preceding mean). Identifies whether current peak is 1.2 times greater than previous average. If it is, first_increase_index becomes the index of that peak and the for loop breaks.  
            if smoothed_averaged_peaks['peaks'][i] > threshold_increase * previous_avg:
                first_increase_index = i
                break

        # If an index is identified, the value stored is the corresponding timestamp (in addition to the task session key and index)
        if first_increase_index is not None:
            first_increase_value = smoothed_averaged_peaks['times'][first_increase_index]
            vns_activation.append((key, first_increase_index, first_increase_value))
        else:
            vns_activation.append((key, None, None))

        # Add traces to the subplot. Uses created dataframes for ease. 
        fig.add_trace(go.Scatter(x=task_sess_df['rel_time'], y=task_sess_df['rectified_emg'], mode='lines', name='Original Signal', line=dict(color='rgba(0, 0, 255, 0.2)')), row=idx+1, col=1)
        fig.add_trace(go.Scatter(x=identified_peaks['peak_time'], y=identified_peaks['peak_amplitude'], mode='markers', name='Identified Peaks', marker=dict(color='red', size=8)), row=idx+1, col=1)
        fig.add_trace(go.Scatter(x=smoothed_averaged_peaks['times'], y=smoothed_averaged_peaks['peaks'], mode='lines', name='Smoothed Peaks', line=dict(color='green')), row=idx+1, col=1)

        if first_increase_index is not None:
            fig.add_trace(go.Scatter(x=[first_increase_value], y=[smoothed_averaged_peaks['peaks'][first_increase_index]], mode='markers', name='VNS activation', marker=dict(color='black', size=10)), row=idx+1, col=1)

    fig.update_layout(height=sum(row_heights) + len(vns_task_session_keys) * 50, width=1200, title_text="VNS activation identification from rectified EMG signal", showlegend=False)

    fig.show()

    # The lists storing the timestamp of the first indication of conditions meeting crterion in the smoothed data converted to dataframe which is returned by the function.
    vns_start_times = pd.DataFrame({
    'task_session': [t[0] for t in vns_activation],
    'rel_idx': [t[1] for t in vns_activation],
    'rel_time': [t[2] for t in vns_activation]
    })

    return vns_start_times


def process_ecg_data(biopac_data, ecg_channels, task_session_summary, plot=False):
    """ 
    Function to extract and plot recorded ECG data. From raw signal, r-peaks are identified and consecutive r-r intervals and instantaneous heart rate is recorded. The data is reduced to that recorded over each task session, based on identified triggers. There is the option to plot the r-r interval across time for each task session for visualisation. 
    
    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - ecg_channels: string, the ECG channel name containing the raw ECG data. 
    - task_session_summary: pd.DataFrame, contains broad summary statistics of the characteristics of task sessions identified through triggers.
    - plot: boolean, if True, creates subplots for each task session of consecutive r-r interval across time using Plotly. 

    Returns:
    - task_session_dict: dict, contains r-peak dataframe (value) for each task session (key).     
    """

    # Objects to store dataframes for each raw ECG data file and identified r-peaks
    all_ecg_data = [] 
    all_r_peaks = []
    data_file_counter = 1 

    # For loop to extract ECG and time data from loaded data files
    for raw_biopac in biopac_data:
        ecgdata = raw_biopac.copy().pick(picks=ecg_channels)
        
        # Filter the ECG data and extract times
        cardiac_cycles, times = ecgdata.get_data(return_times=True)
        cardiac_cycles = np.squeeze(cardiac_cycles) # reduces data to a one dimensional array 
        
        # Creating dataframe for storage of ECG data from each data file
        ecg_data = pd.DataFrame({
            'cardiac_cycles': cardiac_cycles, 
            'time': times,
            'data_file': data_file_counter  # Add the data_file variable
        })

        # Applying filter to ECG data (using MNE's filter function, if needed)
        ecgdata_filtered = ecgdata.filter(l_freq=0.5, h_freq=None)
        ecg_data['cardiac_cycles_filtered'] = np.squeeze(ecgdata_filtered.get_data())

        # Identifying R-peaks
        rpeak_threshold = 0.001
        peaks, _ = find_peaks(ecg_data['cardiac_cycles_filtered'], height=rpeak_threshold)

        # Apply refractory period
        refractory_period = 0.2  # in seconds
        sampling_rate = 2000  # Hz
        refractory_samples = int(refractory_period * sampling_rate)

        # Remove peaks that are too close to each other
        filtered_peaks = [peaks[0]]  # Add the first peak
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] > refractory_samples:
                filtered_peaks.append(peak)

        # Convert these indices to time using the sampling rate
        rpeak_times = np.array(filtered_peaks) / sampling_rate

        # Creating dataframe to house r-peaks and convert r-r intervals into instantaneous heart rate
        heart_rate_data = pd.DataFrame({'rpeak_time': rpeak_times})
        heart_rate_data['r_r_interval'] = heart_rate_data['rpeak_time'].diff()
        heart_rate_data['heart_rate'] = [60 / rr if rr != 0 else np.nan for rr in heart_rate_data['r_r_interval']]
        heart_rate_data['data_file'] = data_file_counter   

        # Append the raw ECG dataframe to the list (one for each datafile)
        all_ecg_data.append(ecg_data)
        
        # Append the identified r-peaks dataframe to the list 
        all_r_peaks.append(heart_rate_data)

        # Increment the data_file counter for the next file
        data_file_counter += 1
    
    # Concatenating all raw ECG dataframes into a single dataframe
    ecg_data_df = pd.concat(all_ecg_data, ignore_index=True)

    # Concatenating all R-peak dataframes into a single dataframe
    rpeaks_df = pd.concat(all_r_peaks, ignore_index=True)

    # Reduce r-peak data down to time range delimited by task sessions
    task_session_dict = {}

    for index, row in task_session_summary.iterrows():
        data_file = row['data_file']
        start_time = row['first_trig']
        end_time = row['last_trig'] 
        
        # Filter the ECG dataframe to get the relevant rows
        task_data = rpeaks_df[
            (rpeaks_df['data_file'] == data_file) &
            (rpeaks_df['rpeak_time'] >= start_time) &
            (rpeaks_df['rpeak_time'] <= end_time)].copy()  
        
        # Calculate relative time starting from the beginning of each task session
        task_data['rel_time'] = task_data['rpeak_time'] - start_time
        
        # Store the filtered dataframe in the dictionary with the index as the key
        task_session_dict[index] = task_data

    # Plotting the ECG data stored within the dictionary if plot=True
    if plot:
        # Calculate the total number of subplots
        num_subplots = len(task_session_dict)

        # Dynamically adjust vertical spacing
        vertical_spacing = min(0.05, 1 / (num_subplots + 1))

        # Create a figure with subplots
        fig = make_subplots(
            rows=num_subplots, cols=1, 
            subplot_titles=[f"Task Session {key}" for key in task_session_dict.keys()],
            shared_xaxes=True, vertical_spacing=vertical_spacing,
            specs=[[{"secondary_y": True}] for _ in range(num_subplots)]
        )

        # Iterate over each key (task session) and its corresponding dataframe
        for i, (key, df) in enumerate(task_session_dict.items(), start=1):
            # Add traces for r-r interval and heart rate
            fig.add_trace(go.Scatter(x=df['rel_time'], y=df['r_r_interval'], mode='lines', name='R-R Interval'), row=i, col=1)
            fig.add_trace(go.Scatter(x=df['rel_time'], y=df['heart_rate'], mode='lines', name='Heart Rate', yaxis='y2'), row=i, col=1)

            # Update x-axis and y-axis titles for each subplot
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="R-R Interval (s)", row=i, col=1)
            fig.update_yaxes(title_text="Heart Rate (bpm)", row=i, col=1, secondary_y=True)

        # Update layout
        fig.update_layout(
            title="R-R Interval and Instantaneous Heart Rate During Each Task Session",
            height=1400 * num_subplots,
            showlegend = False  
        )

        # Show the plot
        fig.show()

    return task_session_dict