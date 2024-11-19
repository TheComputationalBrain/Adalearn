# Module imports
import pandas as pd
import numpy as np
import mne
import os
from itertools import combinations
import scipy as sc
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'


def load_biopac_data(participant_str, condition_str, run_ranges):
    """
    Function to load biopac .edf files. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. 
    
    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - run_ranges: list of dicts, each dict contains 'run_start' and 'run_end' denoting the range of task sessions in each file

    Returns:
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - stim_chan: string, the stimulation (emg) channel name
    - trig_chan: string, the adalearn outcome trigger channel names
    """
    # Set the working directory of the files
    base_dir = f'/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition_str}/physio'
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
        filename = f'sub-{participant_str}_ses-{condition_str}_task-AdalearnVNS_run-{run_start}to{run_end}_physio.edf'
        
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
    print(f"Selected stim channel: {stim_chan}")

    # Convert ECG channel set to a list and take the first element
    ecg_chan = list(ecg_chan_set)[0] if ecg_chan_set else None
    print(f"Selected ECG channel: {ecg_chan}")

    # Convert trigger set to a list and take the third element (need to make sure that combined channel is always the third element)
    trig_set_to_list = list(trig_chan_set)
    trig_set_to_list = sorted(trig_set_to_list)
    print(trig_set_to_list)
    trig_chan = trig_set_to_list[2]
    print(f"Selected trig channel: {trig_chan}")

    return biopac_data, stim_chan, ecg_chan, trig_chan


def process_emg_data(biopac_data, stim_channels, emg_by_file_plot = False):
    """ 
    Function to process the loaded emg data. It creates and returns a dataframe with the emg data and (relative) timestamp (from beginning of file recording). EMG signal is rectified and enveloped. Plots of the emg data are generated from each data file to enable quality check of data.

    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - stim_channels: string, the stim channel name (used to extract the EMG data).
    - emg_by_file_plot: boolean, if True, plots the emg signal for each file as subplots (default  =  False;  as uses lots of memory.)

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

        fig.update_layout(height=300 * len(all_emg_data), title_text='EMG Data Across Files',
                      xaxis_title='Time (s)', yaxis_title='EMG')

        fig.show()

    return emg_data_df



def biopac_trigger_extraction(biopac_data, run_ranges, plot_triggers_by_channel = True, plot_trigs_by_task_session = True):
    """
    Function to assess and extract the trigger data from the bipoac file. First it checks whether one "stimulation" channel contains all of the expected triggers. If it does not then it will look to integrate triggersfound across two channels. The value associated with that channel (i.e. 0 or 1; yellow or blue) is also generated based on the channel the trigger came from, before using the triggers to characterise the task sessions within each file. 
    
    Parameters:
    - bipoac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - run_ranges: list of dicts, each dictionary contains 'run_start' and 'run_end' denoting the range of task sessions in each file
    - plot_triggers_by_channel: boolean, option to plot the identified triggers in each trigger channel contained within the biopac file.
    - plot_trigs_by_task_session: boolean, option to plot triggers separated across task sessions (as defined by time differences in the identified triggers)

    Returns:
    - file_trig_df: pd.DataFrame, stores the identified triggers extracted across all trigger channels for each data file. 
    - trig_chan_df: pd.DataFrame, stores the identified triggers extracted from the selected channel
    - task_session_summary_df: pd.DataFrame, contains summary statistics of each calculated task session (time of first trigger, time of last trigger, duration of task session, number of triggers within each session). 
    - missing_trigger_df: pd.DataFrame, contains information on identified missing triggers (task session, trigger count, missing index [within a task session]). 
    """
    # Store the raw data and identified triggers within the processed trigger channels
    raw_trig_chan_data_by_file = []
    trig_by_file = []
    identified_chan_trigs = []

    for data_file_count, (raw_biopac, run_range) in enumerate(zip(biopac_data, run_ranges), start=1):
        # Identify the names of the trigger channels
        trig_channels_in_file = [ch for ch in raw_biopac.ch_names if ch.startswith('Digital')]
        print(f"Processing trigger channels in file {data_file_count}...")
        print(f"Trigger channels found in file {data_file_count}: {trig_channels_in_file}")

        # Process and store trigger data (signal amplitude) and associated times for each channel in the datafile. 
        trigger_channel_data = []
        for channel in trig_channels_in_file:
            trigdata = raw_biopac.copy().pick(picks=channel)
            trig_extracted, times = trigdata.get_data(return_times=True)
            trig_extracted = np.squeeze(trig_extracted)

            trig_df = pd.DataFrame({
                'triggers': trig_extracted, 
                'time': times,
                'channel': channel, 
                'file': data_file_count,
                'outcome': None
            })

            trigger_channel_data.append(trig_df)

        # Concatenate the data in a trigger channel in the current file
        raw_trig_chan_df = pd.concat(trigger_channel_data, ignore_index=True)
        raw_trig_chan_data_by_file.append(raw_trig_chan_df)

        # Reduce data by identifying triggers(based on signal exceeding a threshold based on amplitude) and the respective timestamps in each channel per file
        trig_thres = raw_trig_chan_df['triggers'].max() * 0.5
        identified_triggers = []
        trigger_count_per_channel = {}

        for channel in raw_trig_chan_df['channel'].unique():
            channel_data = raw_trig_chan_df[raw_trig_chan_df['channel'] == channel]
            initial_triggers = channel_data[channel_data['triggers'] >= trig_thres]

            if initial_triggers.empty:
                trigger_count_per_channel[channel] = 0
                continue

            valid_triggers = [initial_triggers.iloc[0]]
            for idx in range(1, len(initial_triggers)):
                time_diff = initial_triggers.iloc[idx]['time'] - valid_triggers[-1]['time']
                if time_diff >= 0.8: # This time difference can be changed if necessary. Difference between triggers should be ~1.5s, but may be lower for start/end triggers.
                    valid_triggers.append(initial_triggers.iloc[idx])

            identified_triggers.extend(valid_triggers)
            trigger_count_per_channel[channel] = len(valid_triggers)
            
            if len(valid_triggers) > 1:
                print(f"The number of triggers identified in channel {channel} is: {len(valid_triggers)}")
            else:
                print(f"No valid triggers found in channel {channel}")

        id_trig_df = pd.DataFrame(identified_triggers)

        # Defining the outcome associated with each channel. 
        id_trig_df.loc[id_trig_df['channel'] == 'Digital (STP Inp-0', 'outcome'] = 1
        id_trig_df.loc[id_trig_df['channel'] == 'Digital (STP Inp-1', 'outcome'] = 0

        if 'Digital (STP Inp-2' in id_trig_df['channel'].unique():
            channel_3_data = id_trig_df[id_trig_df['channel'] == 'Digital (STP Inp-2']
            for idx, row in channel_3_data.iterrows():
                if row['time'] in id_trig_df[(id_trig_df['channel'] == 'Digital (STP Inp-0')]['time'].values:
                    id_trig_df.loc[idx, 'outcome'] = 1
                elif row['time'] in id_trig_df[(id_trig_df['channel'] == 'Digital (STP Inp-1')]['time'].values:
                    id_trig_df.loc[idx, 'outcome'] = 0

        trig_by_file.append(id_trig_df)

        # Option to plot the triggers identified across all trigger channels
        if plot_triggers_by_channel:
            num_channels = len(id_trig_df['channel'].unique())
            fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True, 
                                subplot_titles=[f"Channel {i+1}" for i in range(num_channels)])
            
            for i, channel in enumerate(id_trig_df['channel'].unique()):
                channel_data = id_trig_df[id_trig_df['channel'] == channel]
                
                color_map = {1: 'blue', 0: 'yellow'}
                marker_colors = [color_map[val] if val in color_map else 'green' for val in channel_data['outcome']]

                fig.add_trace(go.Scatter(
                    x=channel_data['time'], y=channel_data['triggers'], mode='markers',
                    marker=dict(color=marker_colors),
                    name=f'Triggers in {channel}'),
                    row=i+1, col=1
                )

            fig.update_layout(
                height=300*num_channels, width=1000,
                title_text=f"Triggers within stimulation channels for file {data_file_count}"
            )

            fig.show()

        # Checking if one channel matches the expected number of triggers given the number of task sessions within the file
        expected_trigs = 77 * (int(run_range['run_end']) - int(run_range['run_start']) + 1)
        print(f"The number of triggers expected within file {data_file_count} is: {expected_trigs}")

        # Check for a single channel or paired channels with correct trigger count
        single_channel_match_found = False
        paired_channel_match = False
        count_diff_dict = {channel: abs(count - expected_trigs) for channel, count in trigger_count_per_channel.items()}
        channel_pairs_dict = {
            pair: abs(trigger_count_per_channel[pair[0]] + trigger_count_per_channel[pair[1]] - expected_trigs)
            for pair in combinations(trigger_count_per_channel.keys(), 2)
        }

        # Check single channels
        for channel, diff in count_diff_dict.items():
            if diff == 0:
                print(f"Correct number of triggers ({trigger_count_per_channel[channel]}) identified in {channel}. Appending trigger information...")
                identified_chan_trigs.append(id_trig_df[id_trig_df['channel'] == channel])
                single_channel_match_found = True
                break  # Exit loop once a match is found

        # Check pairs if no single match was found
        if not single_channel_match_found:
            for pair, diff in channel_pairs_dict.items():
                if diff == 0:
                    combined_count = trigger_count_per_channel[pair[0]] + trigger_count_per_channel[pair[1]]
                    print(f"Correct number of triggers ({combined_count}) found by combining {pair}. Appending trigger information...")
                    channel_1 = id_trig_df[id_trig_df['channel'] == pair[0]]
                    channel_2 = id_trig_df[id_trig_df['channel'] == pair[1]]
                    integrated_trigger_channel = pd.merge(channel_1, channel_2, on='time', how='outer')
                    integrated_trigger_channel['channel'] = integrated_trigger_channel['channel_x'].combine_first(integrated_trigger_channel['channel_y'])
                    integrated_trigger_channel['triggers'] = integrated_trigger_channel['triggers_x'].combine_first(integrated_trigger_channel['triggers_y'])
                    integrated_trigger_channel['file'] = integrated_trigger_channel['file_x'].combine_first(integrated_trigger_channel['file_y'])
                    integrated_trigger_channel['outcome'] = integrated_trigger_channel['outcome_x'].combine_first(integrated_trigger_channel['outcome_y'])
                    integrated_trigger_channel = integrated_trigger_channel.drop(columns=['channel_x', 'channel_y', 'triggers_x', 'triggers_y', 'file_x', 'file_y', 'outcome_x', 'outcome_y'])
                    integrated_trigger_channel = integrated_trigger_channel.drop_duplicates(subset='time', keep='first').reset_index(drop=True)
                    identified_chan_trigs.append(integrated_trigger_channel)
                    paired_channel_match = True
                    break  # Exit loop once a pair match is found

        # Handle closest match if no exact match was found
        if not single_channel_match_found and not paired_channel_match:
            closest_channel = min(count_diff_dict, key=lambda k: abs(count_diff_dict[k]))
            closest_pair = min(channel_pairs_dict, key=lambda k: abs(channel_pairs_dict[k]))
            if count_diff_dict[closest_channel] <= channel_pairs_dict[closest_pair]:
                print(f"Closest match found in single channel: {closest_channel} with {trigger_count_per_channel[closest_channel]} triggers.")
                identified_chan_trigs.append(id_trig_df[id_trig_df['channel'] == closest_channel])
            else:
                combined_count = trigger_count_per_channel[closest_pair[0]] + trigger_count_per_channel[closest_pair[1]]
                print(f"Closest match found in channel pair: {closest_pair} with {combined_count} triggers.")
                channel_1 = id_trig_df[id_trig_df['channel'] == closest_pair[0]]
                channel_2 = id_trig_df[id_trig_df['channel'] == closest_pair[1]]
                integrated_trigger_channel = pd.merge(channel_1, channel_2, on='time', how='outer')
                integrated_trigger_channel['channel'] = integrated_trigger_channel['channel_x'].combine_first(integrated_trigger_channel['channel_y'])
                integrated_trigger_channel['triggers'] = integrated_trigger_channel['triggers_x'].combine_first(integrated_trigger_channel['triggers_y'])
                integrated_trigger_channel['file'] = integrated_trigger_channel['file_x'].combine_first(integrated_trigger_channel['file_y'])
                integrated_trigger_channel['outcome'] = integrated_trigger_channel['outcome_x'].combine_first(integrated_trigger_channel['outcome_y'])
                integrated_trigger_channel = integrated_trigger_channel.drop(columns=['channel_x', 'channel_y', 'triggers_x', 'triggers_y', 'file_x', 'file_y', 'outcome_x', 'outcome_y'])
                integrated_trigger_channel = integrated_trigger_channel.drop_duplicates(subset='time', keep='first').reset_index(drop=True)
                identified_chan_trigs.append(integrated_trigger_channel)


    file_trig_df = pd.concat(trig_by_file, ignore_index=True)
    raw_trig_chan_df = pd.concat(raw_trig_chan_data_by_file, ignore_index=True)

    trig_chan_df = pd.concat(identified_chan_trigs, ignore_index=True)

    trig_chan_df.sort_values(by=['file', 'time'], inplace=True)
    trig_chan_df.reset_index(drop=True, inplace=True)

     # Create time_diff to act as criterion for separating task sessions. Again, difference greater of 1.5 s (here 3 s used) is the criterion as that is the time interval between stimuli. 
    trig_chan_df['time_diff'] = trig_chan_df['time'].diff().fillna(0)

    # Create task session variable based on criteria
    task_session = 0 
    task_sessions = []
    prev_data_file = None
    for index, row in trig_chan_df.iterrows():
        if row['time_diff'] > 20 or row['file'] != prev_data_file:
            task_session += 1
        task_sessions.append(task_session)
        prev_data_file = row['file']

    trig_chan_df['task_session'] = task_sessions
    trig_chan_df['task_session'] = trig_chan_df['task_session'].astype(int)

    # Storing missing trigger values present within task sessions
    # Group by task session
    trig_chan_df['time_diff'] = trig_chan_df.groupby('task_session')['time_diff'].transform(lambda x: x.mask(x.index == x.index[0], float('nan')))

    task_session_groups = trig_chan_df.groupby('task_session')

    # Initialize a list to store information about sessions with missing triggers
    missing_trigger_sessions = []

    # Iterate through each task session
    for session, group in task_session_groups:
        # Count the number of triggers in the session
        trigger_count = len(group)
        
        # Check for less than 77 triggers
        if trigger_count < 77:
            # Identify the indices where time_diff > 1.6 seconds
            missing_indices = group.index[group['time_diff'] > 1.6].tolist()
            
            if missing_indices:
                # Append the session number, trigger count, and indices of missing triggers
                missing_trigger_sessions.append({
                    'task_session': session,
                    'trigger_count': trigger_count,
                    'missing_indices': missing_indices
                })

    # Convert the results to a DataFrame
    missing_trigger_df = pd.DataFrame(missing_trigger_sessions)

    # Display the results
    if not missing_trigger_df.empty:
        print("Sessions with missing triggers:")
        print(missing_trigger_df)
    else:
        print("No sessions with missing triggers found.")

    # Insert NaN rows at missing indices 
    for _, row in missing_trigger_df.iterrows():
        task_session = row['task_session']
        for idx in row['missing_indices']:
            # Create a new row with NaN values except for task_session
            nan_row = pd.Series({col: np.nan for col in trig_chan_df.columns})
            nan_row['task_session'] = task_session
            
            # Insert the NaN row at the specific index
            trig_chan_df = pd.concat([trig_chan_df.iloc[:idx], pd.DataFrame([nan_row]), trig_chan_df.iloc[idx:]]).reset_index(drop=True)

    # Create a list to store task session information
    task_session_summary = []

    # Loop through each unique task session
    for session in trig_chan_df['task_session'].unique():
        # Filter the data for the current session
        session_data = trig_chan_df[trig_chan_df['task_session'] == session]
    
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

    if plot_trigs_by_task_session:
        task_sessions = len(trig_chan_df['task_session'].unique())
        fig = make_subplots(rows=task_sessions, cols=1, shared_xaxes=False, 
                                subplot_titles=[f"Task session {i+1}" for i in range(task_sessions)])
            
        for i, session in enumerate(trig_chan_df['task_session'].unique()):
            session_data = trig_chan_df[trig_chan_df['task_session'] == session]
            
            color_map = {1: 'blue', 0: 'yellow'}
            marker_colors = [color_map[val] if val in color_map else 'green' for val in session_data['outcome']]

            fig.add_trace(go.Scatter(
                x=session_data['time'], y=session_data['triggers'], mode='markers',
                marker=dict(color=marker_colors),
                name=f'Triggers in task session {session}'),
                row=i+1, col=1
            )

        fig.update_layout(
            height=300*task_sessions, width=1000,
            title_text=f"Triggers within identified trigger channel for task sessions",
            showlegend=False
        )

        fig.show()

    # Convert the list of dictionaries into a DataFrame
    task_session_summary_df = pd.DataFrame(task_session_summary)

    return file_trig_df, trig_chan_df, task_session_summary_df, missing_trigger_df


def extract_task_session_emg_data(condition_str, emg_data_df, task_session_summary,  plot_task_session_emg=False):
    """ 

    Function to reduce the size of the data to only that within a task session. Creates a dictionary of stored dataframes housing the EMG data for each task session. Also creates a relative timestamp for each task session. Can plot rectified EMG against time for each task session for visualization.

    Parameters:
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - emg_data_df: pd.DataFrame, contains EMG data, associated (relative) timestamp and data file number.  
    - task_session_summary: pd.DataFrame, contains broad summary statistics of the characteristics of task sessions identified through triggers.
    - 
    - plot_task_session_emg: boolean, if True, creates subplots for each task session of rectified EMG signal (default = False)

    Returns:
    - emg_task_session_dict: dict, contains EMG dataframe (value) for each task session (key).
    """
    # Initialize an empty dictionary to store the dataframes
    emg_task_session_dict = {}

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
        emg_task_session_dict[int(row['task_session'])] = task_data

    # Plotting the EMG data stored within the dictionary if plot=True
    if plot_task_session_emg:
        # Calculate the total number of subplots
        num_subplots = len(emg_task_session_dict)

        # Create a figure with subplots and specify row heights
        fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=[f"Task Session {key}" for key in emg_task_session_dict.keys()],
                            row_heights=[0.6] * num_subplots)

        # Iterate over each key (task session) and its corresponding dataframe
        for i, (key, df) in enumerate(emg_task_session_dict.items(), start=1):
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

    print(f"In condition {condition_str} the number of task sessions is: {len(emg_task_session_dict)}") 
    for key, df in emg_task_session_dict.items(): print(f"Task Session {key} shape: {df.shape}")


    return emg_task_session_dict


def process_ecg_data(participant_str, condition_str, biopac_data, ecg_channels, task_session_summary, ecg_trace = 'cardiac_cycles',apply_filter=True,  plot_raw_ecg_file=False, plot_rpeaks_session = True, plot_r_r_interval_session = True, plot_instant_hr_session = True):
    """ 
    Function to extract and plot recorded ECG data. From raw signal, r-peaks are identified and consecutive r-r intervals and instantaneous heart rate is recorded. The data is reduced to that recorded over each task session, based on identified triggers. There is the option to plot the r-r interval across time for each task session for visualisation. 
    
    Parameters:
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - ecg_channels: string, the ECG channel name containing the raw ECG data. 
    - plot_raw_ecg_file: boolean, if true, creates subplots for each task session of the raw ECG signal across time using Plotly.
    - plot_rpeaks_session: boolean, if true, creates subplots for each task session of with identified r-peaks superimposed on the raw ECG signal across time using Plotly. 
    - plot_r_r_interval_session: boolean, if true, creates subplots for each task session of consecutive r-r interval across time using Plotly. 
    - plot_raw_instant_hr_file: boolean, if true, creates subplots for each task session of consecutive instant HR across time using Plotly.  

    Returns:
    - ecg_data_df: DataFrame, contains concatenated ECG data across files.
    """
    
    all_ecg_data = [] 
    data_file_counter = 1 

    for raw_biopac in biopac_data:
        ecgdata = raw_biopac.copy().pick(picks=ecg_channels)
        
        # Extract ECG data and times
        cardiac_cycles, times = ecgdata.get_data(return_times=True)
        cardiac_cycles = np.squeeze(cardiac_cycles) # reduce data to 1D array 
        
        # Creating DataFrame for storage of ECG data from each file
        ecg_data = pd.DataFrame({
            'cardiac_cycles': cardiac_cycles, 
            'time': times,
            'data_file': data_file_counter  # Add the data_file variable
        })

        # Apply filter if specified
        if apply_filter:
            ecgdata_filtered = ecg_trace_filter(ecgdata, low_filter_cutoff = 40, high_filter_cutoff = None, sampling_rate = 2000, verbose = False)
            
            ecg_data['cardiac_cycles_filtered'] = np.squeeze(ecgdata_filtered)

        all_ecg_data.append(ecg_data)
        
        data_file_counter += 1

    ecg_data_df = pd.concat(all_ecg_data, ignore_index=True)

    # Option to plot raw ecg from each data file
    if plot_raw_ecg_file:
        fig = make_subplots(rows=len(all_ecg_data), cols=1, shared_xaxes=True, 
                            subplot_titles=[f'File {i+1}' for i in range(len(all_ecg_data))])

        for idx, ecg_data in enumerate(all_ecg_data):
            y_data = ecg_data['cardiac_cycles_filtered'] if 'cardiac_cycles_filtered' in ecg_data.columns else ecg_data['cardiac_cycles']
            fig.add_trace(
                go.Scatter(x=ecg_data['time'], y=y_data, mode='lines', name=f'File {idx + 1}'),
                row=idx + 1, col=1
            )

        fig.update_layout(height=300 * len(all_ecg_data), title_text='ECG Data Across Files',
                          xaxis_title='Time (s)', yaxis_title='ECG')

        fig.show()

    # Reducing ecg down to individual task sessions
    hr_task_session_dict = {}

    for index, row in task_session_summary.iterrows():
        data_file = row['data_file']
        start_time = row['first_trig']
        end_time = row['last_trig'] 
        
        # Filter the ECG DataFrame to get the relevant rows
        task_data = ecg_data_df[
            (ecg_data_df['data_file'] == data_file) &
            (ecg_data_df['time'] >= start_time) &
            (ecg_data_df['time'] <= end_time)
        ].copy()  
        
        # Calculate relative time starting from the beginning of each task session
        task_data['rel_time'] = task_data['time'] - start_time
        
        # Store the filtered DataFrame in the dictionary with the index as the key
        hr_task_session_dict[index +1] = task_data
    
    print("Initalising r-peak detection...")

    rpeaks_session_dict = rpeak_detection(participant_str=participant_str, condition_str=condition_str, hr_dict=hr_task_session_dict, ecg_trace=ecg_trace, height = None, threshold = None, distance = (20/60), prominence = 0.0005, sampling_rate = 2000, plot_rpeaks_session=plot_rpeaks_session)

    print("Calculating r-r intervals...")

    r_r_int_session_dict = r_r_interval_calculation(participant_str=participant_str, condition_str=condition_str, rpeaks_session_dict=rpeaks_session_dict, plot_r_r_interval_session=plot_r_r_interval_session, plot_instant_hr_session = plot_instant_hr_session)

    return ecg_data_df, hr_task_session_dict, r_r_int_session_dict



def ecg_trace_filter(ecgdata, low_filter_cutoff = 40, high_filter_cutoff = None, sampling_rate = 2000, verbose = False):
    """
    Function to apply bandpass filters to raw ECG trace using MNE filter function. Takes a copy of the raw ECG and returns an numpy array of the filtered signal. 

    Parameters:
    - ecgdata: mne.io.Raw, raw ECG data stored under the chosen ECG channel.
    - low_filter_cutoff: integer, low-pass filter cutoff frequency in Hz. This sets the upper bound of the frequencies that will be retained in the filtered signal
    - high_filter_cutoff: integer, high-pass filter cutoff frequency in Hz. This sets the lower bound of the frequencies that will be retained in the filtered signal.
    - sampling_rate: integer, sampling rate of the data. 
    - verbose:  If true, the function will print additional information about the filtering process (default = false).

    Returns:
    - ecgdata_filtered: NumPy array, the filtered ECG signal

    """
    ecgdata_filtered = mne.filter.filter_data(
        data=ecgdata.get_data(), 
        sfreq=sampling_rate, 
        l_freq=high_filter_cutoff, 
        h_freq=low_filter_cutoff, 
        verbose=verbose
            )
    
    return ecgdata_filtered



def rpeak_detection(participant_str, condition_str, hr_dict, ecg_trace = 'cardiac_cycles', height = None, threshold = None, distance = (20/60), prominence = 0.0005, sampling_rate = 2000, plot_rpeaks_session = True):
    """
    Function to detect r-peak of the QRS segment of an ECG trace. Uses SciPy's 'find_peaks' function, which uses parameters such as height, threshold, distance, prominence, and width to fine-tune peak detection. Returns a dictionary with the indices of identified r-peaks (the 'properties' object, is a dictionary containing properties of the returned peaks which were calculated as intermediate results during evaluation of the specified conditions).

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - hr_dict: dictionary, contains keys referencing task session number. Each key contains a DataFrame which contains the raw ECG trace for that session, data point time stamps, data file number and possibly a filtered version of the raw ECG trace.
    - ecg_trace: string, column identifier for the ECG trace to be used for r-peak detection.
    - 'find_peaks' parameters:
        - height: number or ndarray or sequence (optional), specifies the required height of peaks.
        - threshold: number or ndarray or sequence (optional), required threshold of peaks, the vertical distance to its neighboring samples.
        - distance: number (optional), required minimal horizontal distance (>= 1) in samples between neighboring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
        - prominence: number or ndarray or sequence (optional), required prominence of peaks.
        - width: number or ndarray or sequence (optional), specifies the required width of peaks in samples.
    - sampling_rate: integer: sampling rate of the data in Hz.
    - plot_rpeaks_session: boolean, if true, option to plot the raw ECG trace for each task session with detected r-peaks overlaid.

    Returns:
    - rpeaks_session_dict: dictionary, contains the indices of detected r-peaks in the ECG trace (values) for each task session (keys)
    """
    rpeaks_session_dict = {}

    # Identifying r-peaks in each key held in the ECG trace containing dictionary
    for session_key, session_data in hr_dict.items():

        ecg_signal = session_data[ecg_trace]  # Column containing ECG trace
        fs = sampling_rate  # Sampling frequency of data

        # Baseline correcting ECG trace by mean response
        ecg_signal = ecg_signal - np.mean(ecg_signal)

        # Find peaks in the ECG signal (r-peaks) using scipy's find_peaks function.
        rpeak_indices, properties = find_peaks(
            ecg_signal,
            height=height,
            threshold=threshold,
            distance=fs * distance,
            prominence=prominence,
            width=None)

        if len(rpeak_indices) == 0:
            print(f"No r-peaks detected for session {session_key}", flush=True)
            rpeaks_session_dict[session_key] = pd.DataFrame(columns=['rpeaks_indices', 'rpeak_rel_time'])
            continue

        rpeaks_time = session_data['time'].iloc[rpeak_indices].reset_index(drop=True)
        
        rpeak_rel_time = session_data['rel_time'].iloc[rpeak_indices].reset_index(drop=True)

        rpeaks_session_dict[session_key] = pd.DataFrame({
            'rpeaks_indices': rpeak_indices,
            'rpeaks_time': rpeaks_time,
            'rpeak_rel_time': rpeak_rel_time
        })

        # Collect the time of the detected R-peaks
        valid_rpeaks = session_data.iloc[rpeak_indices]

        if len(valid_rpeaks) > 1:
            print(f"The number of r-peaks identified in session {session_key} is: {len(valid_rpeaks)}")
        else:
            print(f"No valid r-peaks found in channel {session_key}")

        # Option to plot the raw ECG signal and detected r-peaks
    if plot_rpeaks_session:
        num_subplots = len(hr_dict)
        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Task Session {session_key}" for session_key in hr_dict.keys()]
        )

        # Loop through each key in ecg containing dictionary and then add the ECG signal and detected r-peaks to each subplot
        for i, (session_key, session_data) in enumerate(hr_dict.items(), start=1):
            ecg_signal = session_data[ecg_trace]
            rpeaks_data = rpeaks_session_dict[session_key]  # Extract stored R-peaks data
            
            # Add ECG signal trace
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=ecg_signal, mode='lines', name='ECG Signal'),
                row=i, col=1
            )
            
            # Add r-peak markers (use the actual ECG signal values corresponding to the r-peak indices)
            rpeak_values = ecg_signal.iloc[rpeaks_data['rpeaks_indices']].values
            fig.add_trace(
                go.Scatter(x=rpeaks_data['rpeak_rel_time'], 
                        y=rpeak_values, 
                        mode='markers', 
                        marker=dict(color='red', size=10), 
                        name='Detected R-peaks'),
                row=i, col=1
            )

            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="ECG Amplitude", row=i, col=1)

        fig.update_layout(
            title=f"ECG trace with detected r-peaks across each task session for participant {participant_str} in condition {condition_str}",
            height=600 * num_subplots,  
            showlegend=False  
        )

        fig.show()

    return rpeaks_session_dict



def r_r_interval_calculation(participant_str, condition_str, rpeaks_session_dict, plot_r_r_interval_session=True, plot_instant_hr_session = True):
    """"
    Function to calculate R-R intervals from R-peaks extracted from the QRS segment of an ECG trace. Takes identified R-peaks indices and converts the difference between peaks to a time, used to calculate the duration between peaks (R-R interval). Instantaneous HR is calculated as 60 divided by the interval.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - rpeaks_dict: dictionary, contains keys referencing task session number. Each key contains a dataframe containing the indices of identified R-peaks within the ECG signal and the relative time of each r-peak. 
    - hr_dict: dictionary, contains keys referencing task session number. Each key contains a DataFrame which contains the raw ECG trace for that session, data point time stamps, data file number and a possibly a filtered version of the raw ECG trace.  
    - hr_dict: dictionary, contains the full ECG time series data for each task session.
    - sampling_rate: integer, sampling rate of the data in Hz.
    - plot_r_r_interval_session: boolean, if true, option to plot the calculated R-R interval and instantaneous HR for each task session.

    Returns:
    - r_r_int_session_dict: dictionary, contains a DataFrame with the indices of detected R-peaks in the ECG trace, R-peak times, R-R intervals, and instantaneous HR for each task session.   
    """
    # Loop through the r-peaks dictionary
    for session, df in rpeaks_session_dict.items():
        # Calculate the R-R interval: time difference between consecutive R-peaks
        r_r_intervals = df['rpeaks_time'].diff().iloc[1:].reset_index(drop=True)  
        
        # Calculate Instantaneous HR (bpm)
        hr = 60 / r_r_intervals  
        
        # Add R-R interval and HR to the DataFrame
        df['r_r_intervals'] = pd.Series([None] + r_r_intervals.tolist())  
        df['instantaneous_hr'] = pd.Series([None] + hr.tolist()).fillna(0).round().astype(int)
        df['instantaneous_hr'] = df['instantaneous_hr'].replace(0, pd.NA)
        
        # Update the dictionary with the modified dataframe
        rpeaks_session_dict[session] = df

    # Option to plot the r-r interval by task session 
    if plot_r_r_interval_session:
        num_sessions = len(rpeaks_session_dict)
        fig = make_subplots(
            rows=num_sessions, cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Task Session {key}" for key in rpeaks_session_dict.keys()]
        )

        # Loop through each task session
        for i, (session_key, session_data) in enumerate(rpeaks_session_dict.items(), start=1):

            # Retrieve R-peaks times and R-R intervals for the session
            rpeak_times = session_data['rpeaks_time']
            r_r_intervals = session_data['r_r_intervals'][1:]  # Exclude first value (NaN)

            # Plot R-R intervals
            fig.add_trace(
                go.Scatter(
                    x=rpeak_times[1:],  # Exclude first time as it doesn't have an R-R interval
                    y=r_r_intervals,
                    mode='lines+markers',
                    name=f"R-R Interval (s) - {session_key}",
                    line=dict(color='blue')
                ),
                row=i, col=1,
            )

            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="R-R Intervals (s)", row=i, col=1)

        # Layout adjustments
        fig.update_layout(
            title_text=f"R-R Intervals Across Each Task Session for Participant {participant_str} in Condition {condition_str}",
            height=600 * num_sessions,
            showlegend=False,
        )

        # Show the plot
        fig.show()
    
    # Option to plot the instantaneous hr by task session 
    if plot_instant_hr_session:
        num_sessions = len(rpeaks_session_dict)
        fig = make_subplots(
            rows=num_sessions, cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Task Session {key}" for key in rpeaks_session_dict.keys()]
        )

        # Loop through each task session
        for i, (session_key, session_data) in enumerate(rpeaks_session_dict.items(), start=1):

            # Retrieve R-peaks times and R-R intervals for the session
            rpeak_times = session_data['rpeaks_time']
            r_r_intervals = session_data['instantaneous_hr'][1:]  # Exclude first value (NaN)

            # Plot R-R intervals
            fig.add_trace(
                go.Scatter(
                    x=rpeak_times[1:],  # Exclude first time as it doesn't have an R-R interval
                    y=r_r_intervals,
                    mode='lines+markers',
                    name=f"Instantaneous HR (bpm) - {session_key}",
                    line=dict(color='blue')
                ),
                row=i, col=1,
            )

            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="Instantaneous HR (bpm)", row=i, col=1)

        # Layout adjustments
        fig.update_layout(
            title_text=f"Instantaneous HR Across Each Task Session for Participant {participant_str} in Condition {condition_str}",
            height=600 * num_sessions,
            showlegend=False,
        )

        # Show the plot
        fig.show()

    return rpeaks_session_dict




def r_r_interval_by_change_point(participant_str, condition_str, rpeaks_session_dict, change_point_indices, pre=2, post=10, time_bin_size=0.5, plot_r_r_intervals_by_cp=False):
    """
    Computes R-R interval dynamics around change points using time bins.
    """

    r_r_intervals_by_cp = []

    # Iterate over each row in change_point_indices
    for idx, row in change_point_indices.iterrows():
        task_session = row['task_session']
        rel_time_cp = row['rel_time']  

        # Check if task session exists in dictionary
        if task_session not in rpeaks_session_dict:
            print(f"Task session {task_session} not found in rpeaks dictionary. Skipping.")
            continue

        session_data = rpeaks_session_dict[task_session]

        # Check if session data is not empty
        if session_data.empty:
            print(f"Session data for task session {task_session} is empty. Skipping.")
            continue

        # Filter the session data for r-r intervals within the specified time window
        filtered_data = session_data[
            (session_data['rpeak_rel_time'] >= rel_time_cp - pre) &
            (session_data['rpeak_rel_time'] <= rel_time_cp + post)
        ].copy()

        # Check if filtered_data is not empty
        if filtered_data.empty:
            print(f"No data points found in the specified time window for task session {task_session}. Skipping.")
            continue

        # Adjust the time to be relative to the change point
        filtered_data['rel_time_cp'] = filtered_data['rpeak_rel_time'] - rel_time_cp

        # Add metadata columns
        filtered_data['task_session'] = row['task_session']
        filtered_data['outcome_num'] = row.get('outcome_num', np.nan) 
        filtered_data['change_point_count'] = row.get('change_point_count', np.nan)

        # Append to the list
        r_r_intervals_by_cp.append(filtered_data)

    # Concatenate the list into a DataFrame
    if r_r_intervals_by_cp:
        r_r_interval_by_cp_df = pd.concat(r_r_intervals_by_cp, ignore_index=True)
    else:
        print("No R-R interval data around change points found.")
        return pd.DataFrame(), pd.DataFrame()

    # Define time bins
    time_bins = np.arange(-pre, post + time_bin_size, time_bin_size)

    # Create a time_bin column
    r_r_interval_by_cp_df['time_bin'] = pd.cut(r_r_interval_by_cp_df['rel_time_cp'], bins=time_bins, labels=time_bins[:-1])

    # Aggregate data by time bins
    binned_r_r_data = r_r_interval_by_cp_df.groupby(['time_bin']).agg(
        mean_r_r_interval=('r_r_intervals', 'mean'),
        std_r_r_interval=('r_r_intervals', 'std'),
        count=('r_r_intervals', 'size')
    ).reset_index()

    binned_r_r_data['sem_r_r_interval'] = binned_r_r_data['std_r_r_interval'] / np.sqrt(binned_r_r_data['count'])
    binned_r_r_data['time_bin'] = binned_r_r_data['time_bin'].astype(float)

    # Optional Plot
    if plot_r_r_intervals_by_cp:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=binned_r_r_data['time_bin'],
            y=binned_r_r_data['mean_r_r_interval'],
            mode='lines',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=binned_r_r_data['time_bin'],
            y=binned_r_r_data['mean_r_r_interval'] + binned_r_r_data['sem_r_r_interval'],
            mode='lines',
            line=dict(color='lightblue'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=binned_r_r_data['time_bin'],
            y=binned_r_r_data['mean_r_r_interval'] - binned_r_r_data['sem_r_r_interval'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='lightblue'),
            showlegend=False
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=f'R-R interval dynamics around change point for participant{participant_str} in condition {condition_str}',
            xaxis_title='Time (s)',
            yaxis_title='R-R Interval (ms)'
        )
        fig.show()

    return r_r_interval_by_cp_df, binned_r_r_data



def identify_vns_from_emg(emg_task_session_dict, vns_task_session_keys, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 2, threshold_increase = 1.2):
    """ 
    Function to identify the start and end of VNS activation from emg recording. EMG signal is noisey and attempts to use conventional methods to extract the enevelope of the signal were unfruitful (e.g. low-pass filtering, RMS, hilbert transformation; low-pass filtering destroys the signal). Opted for a very simple approach which identifies peaks within the signal to create a new "signal" by averaging around peaks. Identification of the start of VNS activation define by arbitary increase in the signal (here, 20% increase from the average of a number of points preceding a given peak). Function plots the data for visual inspection and returns the identified relative start time of VNS activation. 

    Parameters:
    - emg_task_session_dict: dict, contains EMG dataframe (value) for each task session (key).
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
        data = emg_task_session_dict[key]
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
