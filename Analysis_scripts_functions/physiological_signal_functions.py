# Module imports
import pandas as pd
import numpy as np
import mne
import neurokit2 as nk
import os
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
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


def load_biopac_data(directory_path, participant_str, condition_str, run_ranges):
    """
    Function to load biopac .edf files. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. 
    
    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - run_ranges: list of dicts, each dict contains 'run_start' and 'run_end' denoting the range of task sessions in each file

    Returns:
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - stim_chan: string, the stimulation (emg) channel name
    - trig_chan: string, the adalearn outcome trigger channel names
    - sampling_rate: float, the sampling rate of the data in files
    """
    # Create a list to store biopac data files
    biopac_data = []

    # Create a set to store unique stim channel names
    stim_chan_set = set()

    # Create a set to store unique ECG channel names
    ecg_chan_set = set()

    # Create a set to store unique trig channel names
    trig_chan_set = set()

    # Create list to store sampling rates
    sampling_rates = []

    # Iterate over each run range
    for run_range in run_ranges:
        run_start = run_range['run_start']
        run_end = run_range['run_end']
        filename = f'sub-{participant_str}_ses-{condition_str}_task-AdalearnVNS_run-{run_start}to{run_end}_physio.edf'
        filepath = f"{directory_path}/ses-{condition_str}/physio/{filename}"
        
        if os.path.exists(filepath):
            print(f"Loading file: {filename}")
            raw_biopac = mne.io.read_raw_edf(filepath, preload=True)
            fs = raw_biopac.info['sfreq']
            sampling_rates.append(fs)

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

    # Checking that the sampling rates are the same across files
    if len(set(sampling_rates)) == 1:
        sampling_rate = int(sampling_rates[0])  
    else:
        sampling_rate = None
        raise ValueError("Sampling frequencies are not consistent across files.")

    return biopac_data, stim_chan, ecg_chan, trig_chan, sampling_rate


def process_emg_data(participant_str, condition_str, biopac_data, stim_channels, emg_by_file_plot = False):
    """ 
    Function to process the loaded emg data. It creates and returns a dataframe with the emg data and (relative) timestamp (from beginning of file recording). EMG signal is rectified and enveloped. Plots of the emg data are generated from each data file to enable quality check of data.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
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
        
        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_emg-by-file.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    return emg_data_df



def biopac_trigger_extraction(participant_str, condition_str, biopac_data, run_ranges, plot_triggers_by_channel = True, plot_trigs_by_task_session = True):
    """
    Function to assess and extract the trigger data from the bipoac file. First it checks whether one "stimulation" channel contains all of the expected triggers. If it does not then it will look to integrate triggersfound across two channels. The value associated with that channel (i.e. 0 or 1; yellow or blue) is also generated based on the channel the trigger came from, before using the triggers to characterise the task sessions within each file. 
    
    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
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
        if len(trig_channels_in_file) == 3:
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
        
        elif len(trig_channels_in_file) == 4:
            id_trig_df.loc[id_trig_df['channel'] == 'Digital (STP Inp-1', 'outcome'] = 1
            id_trig_df.loc[id_trig_df['channel'] == 'Digital (STP Inp-2', 'outcome'] = 0

            if 'Digital (STP Inp-3' in id_trig_df['channel'].unique():
                channel_4_data = id_trig_df[id_trig_df['channel'] == 'Digital (STP Inp-3']
                for idx, row in channel_4_data.iterrows():
                    if row['time'] in id_trig_df[(id_trig_df['channel'] == 'Digital (STP Inp-1')]['time'].values:
                        id_trig_df.loc[idx, 'outcome'] = 1
                    elif row['time'] in id_trig_df[(id_trig_df['channel'] == 'Digital (STP Inp-2')]['time'].values:
                        id_trig_df.loc[idx, 'outcome'] = 0

            trig_by_file.append(id_trig_df)
        
        else:
            print("ERROR: Could not determine outcomes from physiological data file")
            
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

            # Save the figure to the desktop
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            file_name = f"sub-{participant_str}_ses-{condition_str}_biopac-triggers-by-file.png"
            fig.write_image(os.path.join(desktop_path, file_name))

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

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_biopac-triggers-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    # Convert the list of dictionaries into a DataFrame
    task_session_summary_df = pd.DataFrame(task_session_summary)

    return file_trig_df, trig_chan_df, task_session_summary_df, missing_trigger_df


def extract_task_session_emg_data(participant_str, condition_str, emg_data_df, task_session_summary, emg_signal = 'rectified_emg', plot_task_session_emg=False):
    """ 

    Function to reduce the size of the data to only that within a task session. Creates a dictionary of stored dataframes housing the EMG data for each task session. Also creates a relative timestamp for each task session. Can plot various EMG signals against time for each task session for visualization.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - emg_data_df: pd.DataFrame, contains EMG data, associated (relative) timestamp and data file number.  
    - task_session_summary: pd.DataFrame, contains broad summary statistics of the characteristics of task sessions identified through triggers.
    - emg_signal: string, defines which signal to plot. Can be the raw signal ('emg'), the rectified signal ('rectified_emg') or filtered signal ('lowpass_filtered')
    - plot_task_session_emg: boolean, if true, creates subplots for each task session of specified EMG signal (default = False)

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
            fig.add_trace(go.Scatter(x=df['rel_time'], y=df[emg_signal], mode='lines', name=f'Task Session {key}'),
                          row=i, col=1)

            # Update x-axis and y-axis titles for each subplot
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="EMG", row=i, col=1)

            # Hide the legend for each subplot
            fig.update_traces(showlegend=False)

        # Update layout
        fig.update_layout(
            title=f"EMG vs time across each task session for participant {participant_str} in condition {condition_str}",
            height=4000 
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_emg-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    print(f"In condition {condition_str} the number of task sessions is: {len(emg_task_session_dict)}") 
    for key, df in emg_task_session_dict.items(): print(f"Task Session {key} shape: {df.shape}")


    return emg_task_session_dict



def extract_ecg_signals(participant_str, condition_str, biopac_data, ecg_channels, plot_raw_ecg_file=False):
    """
    Function to extract ECG signal from raw biopac datafiles. Takes each file extracts the signal and returns a concatenated data frame across all data files

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - ecg_channels: string, the ECG channel name containing the raw ECG data. 
    - plot_raw_ecg_file: boolean, if true, creates subplots for each task session of the raw ECG signal across time using Plotly.

    Returns:
    - ecg_data_df: DataFrame, contains concatenated dataframe containing ECG data across files.
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
            'raw_ecg': cardiac_cycles, 
            'time': times,
            'data_file': data_file_counter 
        })

        all_ecg_data.append(ecg_data)
        
        data_file_counter += 1

    ecg_data_df = pd.concat(all_ecg_data, ignore_index=True)

    # Option to plot raw ecg from each data file
    if plot_raw_ecg_file:
        fig = make_subplots(rows=len(all_ecg_data), cols=1, shared_xaxes=True, 
                            subplot_titles=[f'File {i+1}' for i in range(len(all_ecg_data))])

        for idx, ecg_data in enumerate(all_ecg_data):
            y_data = ecg_data['raw_ecg']
            fig.add_trace(
                go.Scatter(x=ecg_data['time'], y=y_data, mode='lines', name=f'File {idx + 1}'),
                row=idx + 1, col=1
            )

        fig.update_layout(height=300 * len(all_ecg_data), title_text=f'ECG signals extracted across data files for participant {participant_str} in condition {condition_str}',
                          xaxis_title='Time (s)', yaxis_title='ECG')
        
        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_ecg-by-data-files.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    return ecg_data_df



def ecg_signal_filter(session_data, sampling_rate = 2000, low_cutoff = 40, high_cutoff = None):
    """
    Function to apply high bandpass filter to raw ECG trace using neurokits signal_filter function (https://neuropsychology.github.io/NeuroKit/functions/signal.html#neurokit2.signal_filter). Takes a copy of the raw ECG and returns an numpy array of the filtered signal. 

    Parameters:
    - session_data: pd.Dataframe, contains the raw ECG trace for that session, data point time stamps, data file number and possibly a filtered version of the raw ECG trace.
    - low_cutoff: float, low-pass filter cutoff frequency in Hz. This sets the upper bound of the frequencies that will be retained in the filtered signal.
    - high_cutoff: float, high-pass filter cutoff frequency in Hz. This sets the lower bound of the frequencies that will be retained in the filtered signal.
    - sampling_rate: float, sampling rate of the data. 

    Returns:
    - session_data: pd.DataFrame, updated dataframe now containing a column with the filtered ECG signal.
    """

    ecg_signal = np.array(session_data['raw_ecg'])

    filtered_ecg_signal = nk.signal_filter(signal=ecg_signal, sampling_rate=sampling_rate, lowcut=low_cutoff, highcut=high_cutoff)

    session_data['filtered_ecg'] = pd.Series(filtered_ecg_signal, index=session_data.index)
    
    return session_data



def rpeak_evaluation(participant_str, condition_str, ecg_session_dict, ecg_trace='raw_ecg', sampling_rate=2000, correct_artifacts=True, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False):
    """
    Function to detect r-peaks of the QRS segment of an ECG trace.;Uses neurokit2 'ecg_peaks' function. Returns a dictionary with the indices of identified r-peaks and their respective time intervals.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - ecg_session_dict: dictionary, contains keys referencing task session number. Each key contains a DataFrame which contains the raw ECG trace for that session, data point time stamps, and data file number.
    - ecg_trace: string, column identifier for the ECG trace to be used for r-peak detection.
    - sampling_rate: float, sampling rate of the data in Hz.
    - correct_artifacts: boolean, option to correct artifacts in r-peak detection.

    Returns:
    - rpeaks_session_dict: dictionary, contains the indices of detected r-peaks in the ECG trace (values) for each task session (keys).
    """
    rpeaks_session_dict = {}
    
    # Identifying r-peaks in each key held in the ECG trace-containing dictionary
    for session_key, session_data in ecg_session_dict.items():
        ecg_signal = session_data[ecg_trace]

        # Baseline correcting and cleaning (de-noising) signal using neurokit2 ecg_clean function:
        if ecg_trace == "raw_ecg":
            ecg_signal = ecg_signal - np.mean(ecg_signal)
            clean_ecg_signal = pd.Series(
                nk.ecg_clean(ecg_signal=ecg_signal, sampling_rate=sampling_rate, method="neurokit")
            )
            session_data['clean_ecg_signal'] = clean_ecg_signal.values

            # Find peaks in the ECG signal (r-peaks) using neurokit2 r_peaks function:
            rpeaks_indices = nk.ecg_peaks(clean_ecg_signal, sampling_rate=sampling_rate, correct_artifacts=correct_artifacts)[1]
        else:
            # Find peaks in the ECG signal (r-peaks) using neurokit2 r_peaks function:
            rpeaks_indices = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, correct_artifacts=correct_artifacts)[1]

        # Extracting the R-peaks indices and the associated timestamp
        rpeaks = pd.Series(rpeaks_indices['ECG_R_Peaks'])
        rpeaks_time = session_data['time'].iloc[rpeaks].reset_index(drop=True)
        rpeak_rel_time = session_data['rel_time'].iloc[rpeaks].reset_index(drop=True)
        print(f"In task session {session_key}: {len(rpeaks)} R-peaks identifed")

        # Extract R-R intervals and handle erroneous peaks if artifacts are corrected
        if correct_artifacts:
            r_r_intervals = rpeaks_indices['ECG_fixpeaks_rr']
            erroneous_peaks = {
                'ectopic_beats': rpeaks_indices['ECG_fixpeaks_ectopic'], 
                'missed_beats': rpeaks_indices['ECG_fixpeaks_missed'], 
                'extra_beats': rpeaks_indices['ECG_fixpeaks_extra'],
                'long_short_beats': rpeaks_indices['ECG_fixpeaks_longshort']
            }
            total_erroneous_peaks = sum(len(erroneous_peaks[key]) for key in erroneous_peaks.keys())

            print(f"Number of erroneous peaks/artifacts identified and corrected: {total_erroneous_peaks}")
            print(f"Ectopic beats: {len(erroneous_peaks['ectopic_beats'])}")
            print(f"Missed beats: {len(erroneous_peaks['missed_beats'])}")
            print(f"Extra beats: {len(erroneous_peaks['extra_beats'])}")
            print(f"Long/short beats: {len(erroneous_peaks['long_short_beats'])}")
        else:
            r_r_intervals = rpeaks.diff() / sampling_rate

        # Calculate instantaneous HR
        hr = nk.signal_rate(rpeaks, sampling_rate=sampling_rate)

        # Store rpeaks in a dictionary
        rpeaks_session_dict[session_key] = pd.DataFrame({
            'rpeaks_indices': rpeaks,
            'r_r_intervals': r_r_intervals,
            'instant_hr': hr,
            'rpeaks_time': rpeaks_time,
            'rpeak_rel_time': rpeak_rel_time,
            
        })

        # Store the uncorrected peaks to check correction process
        if correct_artifacts:
            uncorrected_peaks = pd.Series(rpeaks_indices['ECG_R_Peaks_Uncorrected'])

            rpeaks_session_dict[session_key]['uncorrected_rpeaks'] = uncorrected_peaks

    # Option to plot the R-peaks (corrected and uncorrected) by task session 
    if plot_rpeaks_session:
        num_sessions = len(ecg_session_dict)
        fig = make_subplots(
            rows=num_sessions,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Task Session {session_key}" for session_key in ecg_session_dict.keys()]
        )

        # Loop through each key in ecg containing dictionary and then add the ECG signal and detected r-peaks to each subplot
        for i, (session_key, session_data) in enumerate(ecg_session_dict.items(), start=1):
            ecg_signal = session_data[ecg_trace]
            time = session_data['rel_time']
            rpeaks_data = rpeaks_session_dict[session_key]  
            
            # Add ECG signal trace
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=ecg_signal, mode='lines', line=dict(color='rgba(224,176,255,0.7)'), name='ECG Signal'),
                row=i, col=1
            )
            
            # Add r-peak markers (use the actual ECG signal values corresponding to the r-peak indices)
            rpeak_values = ecg_signal.iloc[rpeaks_data['rpeaks_indices']].values
            rpeak_times = time.iloc[rpeaks_data['rpeaks_indices']].values
            fig.add_trace(
                go.Scatter(x=rpeak_times, 
                        y=rpeak_values, 
                        mode='markers', 
                        marker=dict(color='blue', size=8), 
                        name='Detected R-peaks - Corrected'),
                row=i, col=1
            )

            if correct_artifacts:
                uncor_rpeak_values = ecg_signal.iloc[rpeaks_data['uncorrected_rpeaks']].values
                uncor_rpeak_times = time.iloc[rpeaks_data['uncorrected_rpeaks']].values
                fig.add_trace(
                    go.Scatter(x=uncor_rpeak_times, 
                            y=uncor_rpeak_values, 
                            mode='markers', 
                            marker=dict(color='red', size=8), 
                            name='Detected R-peaks - Uncorrected'),
                    row=i, col=1
                )

            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="R-R Intervals (s)", row=i, col=1)

        # Layout adjustments
        fig.update_layout(
            title_text=f"Identified R-peaks across each task session for participant {participant_str} in condition {condition_str}",
            height=600 * num_sessions,
            showlegend=False,
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_r-peaks-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        # Show the plot
        fig.show()
    
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
            rpeak_times = session_data['rpeak_rel_time']
            r_r_intervals = session_data['r_r_intervals']

            # Plot R-R intervals
            fig.add_trace(
                go.Scatter(
                    x=rpeak_times, 
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
            title_text=f"R-R intervals across each task session for participant {participant_str} in condition {condition_str}",
            height=600 * num_sessions,
            showlegend=False,
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_r-r-intervals-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

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

            # Retrieve R-peaks times and instantaneous HR for the session
            rpeak_times = session_data['rpeak_rel_time']
            instant_hr = session_data['instant_hr'] 

            # Plot R-R intervals
            fig.add_trace(
                go.Scatter(
                    x=rpeak_times[1:],  # Exclude first time as it doesn't have an R-R interval
                    y=instant_hr,
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
            title_text=f"Instantaneous HR across each task session for participant {participant_str} in condition {condition_str}",
            height=600 * num_sessions,
            showlegend=False,
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_instant-hr-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        # Show the plot
        fig.show()

    return rpeaks_session_dict


def process_ecg_data(participant_str, condition_str, biopac_data, ecg_channels, task_session_summary, sampling_rate = 2000, low_cutoff = 40, high_cutoff = None, ecg_trace='raw_ecg', rpeak_artifact_correction = True, plot_raw_ecg_file=False, plot_ecg_filter_process = False, plot_rpeaks_session = True, plot_r_r_interval_session = True, plot_instant_hr_session = True):
    """ 
    Wrapper function used to extract and plot recorded ECG data. From raw signal, r-peaks are identified and consecutive r-r intervals and instantaneous heart rate is recorded. The data is reduced to that recorded over each task session, based on identified triggers. There is the option to plot the the identified r-peaks, the calculated r-r intervals and instantaneous HR across each task session for visualisation. 
    
    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - biopac_data: list, contains the dictionaries storing the data extracted from the biopac files for each run of task sessions. 
    - ecg_channels: string, the ECG channel name containing the raw ECG data. 
    - task_session_summary: pd.DataFrame, contains summary statistics of each calculated task session (time of first trigger, time of last trigger, duration of task session, number of triggers within each session). 
    - sampling_rate: float, sampling rate of the data in Hz (default = 2000).
    - low_cutoff: float, low-pass filter cutoff frequency in Hz. This sets the upper bound of the frequencies that will be retained in the filtered signal (default = 40 Hz).
    - high_cutoff: float, high-pass filter cutoff frequency in Hz. This sets the lower bound of the frequencies that will be retained in the filtered signal (default = None).
    - ecg_trace: string, column name containing the ECG signal to be processed. Can be 'raw_ecg' or 'filtered_ecg' (default = 'raw_ecg')
    - rpeak_artifact_correction: boolean, if true, will apply artifact correction to r-peak detection using neurokits2 'ecg_peaks' function (https://neuropsychology.github.io/NeuroKit/functions/ecg.html#neurokit2.ecg.ecg_peaks).
    - plot_raw_ecg_file: boolean, if true, creates subplots for each task session of the raw ECG signal across time using Plotly.
    - plot_ecg_filter_process: boolean, if true, plots the raw ECG signal overlaid with the filtered ECG trace for visualisation of the improvement gained for R-peak detection using Plotly (default = false)
    - plot_rpeaks_session: boolean, if true, creates subplots for each task session of with identified r-peaks superimposed on the raw ECG signal across time using Plotly. 
    - plot_r_r_interval_session: boolean, if true, creates subplots for each task session of consecutive r-r interval across time using Plotly. 
    - plot_raw_instant_hr_file: boolean, if true, creates subplots for each task session of consecutive instant HR across time using Plotly.  

    Returns:
    - ecg_data_df: DataFrame, contains concatenated ECG data across files.
    - ecg_session_dict: dict, contains the raw and filtered ECG traces reduced down to the duration of each task session.  
    - rpeaks_session_dict: dict, contains the indices of detected r-peaks in the ECG trace (values) for each task session (keys).
    """
    # Extract ecg signal from data files
    ecg_data_df = extract_ecg_signals(participant_str=participant_str, condition_str=condition_str, biopac_data=biopac_data, ecg_channels=ecg_channels, plot_raw_ecg_file=plot_raw_ecg_file)

    # Reducing ECG signal down to that within individual task sessions
    ecg_session_dict = {}

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
        ecg_session_dict[index +1] = task_data

    # Filter ECG data to improve peak detection 
    for session_key, session_data in ecg_session_dict.items():
        
        session_data = ecg_signal_filter(session_data=session_data,sampling_rate=sampling_rate, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    if plot_ecg_filter_process:
        num_subplots = len(ecg_session_dict)
        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Task Session {session_key}" for session_key in ecg_session_dict.keys()]
        )

        # Loop through each key in ecg containing dictionary and then add the raw and filtered ECG signal
        for i, (session_key, session_data) in enumerate(ecg_session_dict.items(), start=1):
            ecg_signal = session_data['raw_ecg']
            filtered_ecg_signal = session_data['filtered_ecg'] 
            
            # Add raw ECG signal trace
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=ecg_signal, mode='lines', line=dict(color='rgba(0, 0, 255, 0.3)'), name='ECG Signal - Raw'),
                row=i, col=1
            )
            
            # Add cleaned ECG signal
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=filtered_ecg_signal, mode='lines', line=dict(color='rgba(255, 0, 0, 0.4)'), name='ECG Signal - Filtered'),
                row=i, col=1
            )
            
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="ECG Amplitude", row=i, col=1)

        fig.update_layout(
            title=f"Raw and cleaned ECG traces across each task session for participant {participant_str} in condition {condition_str}",
            height=600 * num_subplots,  
            showlegend=False  
        )

        fig.show()
    
    # Extract R-peaks and R-R intervals
    print("Initalising R-peak detection and calculation of R-R intervals...")
    rpeaks_session_dict = rpeak_evaluation(participant_str=participant_str, condition_str=condition_str, ecg_session_dict=ecg_session_dict, ecg_trace=ecg_trace, sampling_rate = sampling_rate, correct_artifacts=rpeak_artifact_correction, plot_rpeaks_session=plot_rpeaks_session, plot_r_r_interval_session=plot_r_r_interval_session, plot_instant_hr_session=plot_instant_hr_session)

    return ecg_data_df, ecg_session_dict, rpeaks_session_dict



def ecg_markers_by_change_point(participant_str, condition_str, rpeaks_session_dict, change_point_indices, ecg_parameter = 'r_r_intervals', pre=2, post=10, time_bin_size=0.5, plot_ecg_intervals_by_cp=False):
    """
    Function to computes R-R interval or instantaneous HR dynamics around change points using time bins. Returns a dataframe with the raw R-R intervals within the designated time window, then the averaged response across bins. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - rpeaks_session_dict: dictionary, contains the indices of detected r-peaks in the ECG trace (values) for each task session (keys).
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
    - ecg_parameter: string, column heading of ECG parameter that is to be evaluated around change points.  
    - pre: float, time (in seconds) before the change point to start the window. 
    - post: float, time (in seconds) after the change point to end the window. 
    - time_bin_size: float, defines the size of the time bins within the window (default = 0.5 s)
    - plot_ecg_intervals_by_cp: boolean, if true, will plot the mean a(plus variance) of the ECG parameter around change points. 

    Returns:
    - ecg_interval_by_cp_df: pd.DataFrame, contains the the extracted data within the window around change points. 
    - binned_ecg_data: pd.DataFrame, contains the mean and variance measures of the data binned within the window around change points.
    """
    ecg_intervals_by_cp = []

    # Iterate over each row in change_point_indices
    for idx, row in change_point_indices.iterrows():
        task_session = row['task_session']
        rel_time_cp = row['rel_time']  

        session_data = rpeaks_session_dict[task_session]

        # Filter the session data for r-r intervals within the specified time window
        filtered_data = session_data[
            (session_data['rpeak_rel_time'] >= rel_time_cp - pre) &
            (session_data['rpeak_rel_time'] <= rel_time_cp + post)
        ].copy()

        # Adjust the time to be relative to the change point
        filtered_data['rel_time_cp'] = filtered_data['rpeak_rel_time'] - rel_time_cp

        # Add metadata columns
        filtered_data['task_session'] = row['task_session']
        filtered_data['outcome_num'] = row.get('outcome_num', np.nan) 
        filtered_data['change_point_count'] = row.get('change_point_count', np.nan)

        # Append to the list
        ecg_intervals_by_cp.append(filtered_data)

    # Concatenate the list into a DataFrame
    ecg_interval_by_cp_df = pd.concat(ecg_intervals_by_cp, ignore_index=True)

    # Define time bins
    time_bins = np.arange(-pre, post + time_bin_size, time_bin_size)

    # Create a time_bin column
    ecg_interval_by_cp_df['time_bin'] = pd.cut(ecg_interval_by_cp_df['rel_time_cp'], bins=time_bins, labels=time_bins[:-1])

    # Aggregate data by time bins
    binned_ecg_data = ecg_interval_by_cp_df.groupby(['time_bin']).agg(
        mean_r_r_interval=(ecg_parameter, 'mean'),
        std_r_r_interval=(ecg_parameter, 'std'),
        count=(ecg_parameter, 'size')
    ).reset_index()

    binned_ecg_data['sem_r_r_interval'] = binned_ecg_data['std_r_r_interval'] / np.sqrt(binned_ecg_data['count'])
    binned_ecg_data['time_bin'] = binned_ecg_data['time_bin'].astype(float)

    # Optional Plot
    if plot_ecg_intervals_by_cp:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=binned_ecg_data['time_bin'],
            y=binned_ecg_data['mean_r_r_interval'],
            mode='lines',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=binned_ecg_data['time_bin'],
            y=binned_ecg_data['mean_r_r_interval'] + binned_ecg_data['sem_r_r_interval'],
            mode='lines',
            line=dict(color='lightblue'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=binned_ecg_data['time_bin'],
            y=binned_ecg_data['mean_r_r_interval'] - binned_ecg_data['sem_r_r_interval'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='lightblue'),
            showlegend=False
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=f'{ecg_parameter} dynamics around change points for participant {participant_str} in condition {condition_str}',
            xaxis_title='Time (s)',
            yaxis_title=f'{ecg_parameter}',
            showlegend=False
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_ecg-by-change-point.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    return ecg_interval_by_cp_df, binned_ecg_data



def vns_activation_id(participant_str, condition_str, emg_task_sess_dict, sampling_rate = 2000, window_size_adjustment = 1, vns_task_sessions = None, threshold = 0.20, frequency = '950.00-1000.00Hz', plot_start_end_vns_power = False, plot_start_end_vns_emg = False):
    """
    Function to identify the start and end of the VNS activation. Decomposses EMG signal into the power across defined frequency bands (using neurokit2 signal processing function, signal_power: https://neuropsychology.github.io/NeuroKit/functions/signal.html#neurokit2.signal_power) to better identify the period of activation within the EMG signal. After extracting the power within a select frequency band, the power signal is smoothed using a rolling average, before the first derivative is calculated. Derivatives are used to create blocks based on the derivative signal exceeding a set threshold. If there are not enough data points in a given window, points are not deemed to form a group around a burst of stimulation and therefore excluded. After this exclusion phase, the first index of the first block and the last index of the last block are taken as the start and end of the stimulation. Function returns a data frame containing the start, end and duration information of the identified activation period. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - emg_task_session_dict: dict, contains EMG dataframe (value) for each task session (key).
    - sampling_rate: float, specifies the recording frequency of the data
    - window_size_adjustment: float, modifier which the sampling rate is divided by to create the window size for the rolling smoothing kernel
    - vns_task_sessions: list, contains the number of the task sessions during which VNS was active
    - threshold: float, bound defining the level at which derivative data points are recorded in order to define the start and end of the VNS activation
    - frequency: string, column name of the frequency band to perfom the VNS activation identification on.
    - plot_start_end_vns_power: boolean, if true, will plot the power signal of the selected frequency band across time for each task session in which VNS was active, with the start and end of the activation marked   
    - plot_start_end_vns_emg: boolean, if true, will plot the rectified EMG signal across time for each task session in which VNS was active, with the start and end of the activation marked

    Returns:
    - vns_start_end_markers: dict, contains a dataframe (value) in which the start (index and relative time), end (index and relative time) and duration (absolute time) of the VNS activation is identified and stored for a given task session (keys)
    """
    # Creates empty dictionary to store dataframes containing power decomposition for each task session 
    power_dict = {}

    for session_key, session_df in emg_task_sess_dict.items():
        if session_key not in vns_task_sessions:
            continue

        emg_signal = session_df['emg'].reset_index(drop=True)
        time = session_df['rel_time'].reset_index(drop=True)


        # Extracting power within defined band ranges
        print("Computing power in designated frequency bands.....")
        power_by_band = nk.signal_power(emg_signal, sampling_rate=sampling_rate, frequency_band=[(100.0, 200.00), (200.0, 300.0), (300.0, 400.0), (400.0, 500.0), (500.0, 600.0), (600.0, 700.0), (700.0, 800.0), (800.0, 900.0), (900.0, 1000.0)], continuous=True)

        power_to_average = np.squeeze(power_by_band[frequency])

        window_size = int(sampling_rate/window_size_adjustment)

        smoothed_power_signal = power_to_average.rolling(window=window_size, center=True).mean()

        derivative = np.gradient(smoothed_power_signal, time)

        power_dict[session_key] = pd.DataFrame({
                'power': power_to_average,
                'smoothed_power': smoothed_power_signal, 
                'derivative': derivative,
                'time': time
            })

        # Identifying start and end times of the VNS based on first derivative exceeding a threshold
        print("Identifying markers...")       
        vns_start_end_markers = {}

        def group_indices(indices, diff_threshold, min_length):
            groups = []
            current_group = [indices[0]]
            for i, diff in enumerate(np.diff(indices)):
                if diff <= diff_threshold:
                    current_group.append(indices[i + 1])
                else:
                    if len(current_group) >= min_length:
                        groups.append(current_group)
                    current_group = [indices[i + 1]]
            if len(current_group) >= min_length:
                groups.append(current_group)
            return groups

        print("Calculating first derivatives....")
        for session_key, df in power_dict.items():
            derivative = df['derivative'].dropna()
            time = df['time']

            threshold_max = derivative.max() * threshold
            threshold_min = derivative.min() * threshold

            above_markers = np.where(derivative > threshold_max)[0]
            below_markers = np.where(derivative < threshold_min)[0]

            if len(above_markers) == 0 or len(below_markers) == 0:
                print(f"No activation markers for session {session_key}.")
                continue

            start_groups = group_indices(above_markers, diff_threshold=1000, min_length=20)
            end_groups = group_indices(below_markers, diff_threshold=1000, min_length=20)

            start_idx = start_groups[0][0]
            end_idx = end_groups[-1][-1]

            start_time = time.iloc[start_idx]
            end_time = time.iloc[end_idx]
            duration = end_time - start_time

            vns_start_end_markers[session_key] = pd.DataFrame({
                'start_index': [start_idx],
                'start_time': [start_time],
                'end_index': [end_idx],
                'end_time': [end_time],
                'duration': [duration]
            })

        print(f"The identified start and end indices for the VNS activation for task session {session_key} are: start - {start_idx}; end - {end_idx}. The start time was {start_time} and the end time was {end_time} resulting in a VNS duration of {duration} seconds")

    # Option to plot the identified start and end times of VNS activation for each task session, overlaid on top of power signal within the selected frequency band.
    if plot_start_end_vns_power:
        num_sessions = len(power_dict)
        fig = make_subplots(
            rows=num_sessions,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.02,
            subplot_titles=[f"Session {key}" for key in power_dict.keys()]
        )

        for i, (session_key, df) in enumerate(power_dict.items(), start=1):
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['power'],
                    mode='lines',
                    name=f"Extracted power in frequency band {frequency} (Session {session_key})",
                    line=dict(color='rgba(0, 0, 255, 0.4)')
                ),
                row=i, col=1
            )

            y_max = df['power'].max()
            
            if session_key in vns_start_end_markers:
                markers_df = vns_start_end_markers[session_key]
                for _, marker_row in markers_df.iterrows():

                    fig.add_shape(
                        type="line",
                        x0=marker_row['start_time'],
                        x1=marker_row['start_time'],
                        y0=0,
                        y1=y_max,
                        xref=f'x{i}',
                        yref=f'y{i}',
                        line=dict(color="green", width=5, dash="dot")
                    )

                    fig.add_shape(
                        type="line",
                        x0=marker_row['end_time'],
                        x1=marker_row['end_time'],
                        y0=0,
                        y1=y_max,
                        xref=f'x{i}',
                        yref=f'y{i}',
                        line=dict(color="red", width=5, dash="dot")
                    )

        fig.update_layout(
            height=900 * num_sessions,
            width=1200,
            title_text=f'Identified start and end times of VNS activation using power in frequency band ({frequency}) for participant {participant_str} in condition {condition_str}',
            xaxis_title='Time (s)',
            yaxis_title='Power',
            showlegend=False
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_vns-activation-on-power-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()
    
    # Option to plot the identified start and end times of VNS activation for each task session, overlaid on top of rectified EMG signal.
    if plot_start_end_vns_emg:
        filtered_sessions = {key: df for key, df in emg_task_sess_dict.items() if key in vns_task_sessions}
        num_sessions = len(filtered_sessions)
        fig = make_subplots(
            rows=num_sessions,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.02,
            subplot_titles=[f"Session {key}" for key in filtered_sessions.keys()]
        )

        for i, (session_key, df) in enumerate(filtered_sessions.items(), start=1):
            fig.add_trace(
                go.Scatter(
                    x=df['rel_time'],
                    y=df['rectified_emg'],
                    mode='lines',
                    name=f"Rectified EMG (Session {session_key})",
                    line=dict(color='rgba(0, 0, 255, 0.4)')
                ),
                row=i, col=1
            )

            y_max = df['rectified_emg'].max()
            

            if session_key in vns_start_end_markers:
                markers_df = vns_start_end_markers[session_key]
                for _, marker_row in markers_df.iterrows():

                    fig.add_shape(
                        type="line",
                        x0=marker_row['start_time'],
                        x1=marker_row['start_time'],
                        y0=0,
                        y1=y_max,
                        xref=f'x{i}',
                        yref=f'y{i}',
                        line=dict(color="green", width=5, dash="dot")
                    )

                    fig.add_shape(
                        type="line",
                        x0=marker_row['end_time'],
                        x1=marker_row['end_time'],
                        y0=0,
                        y1=y_max,
                        xref=f'x{i}',
                        yref=f'y{i}',
                        line=dict(color="red", width=5, dash="dot")
                    )

        fig.update_layout(
            height=900 * num_sessions,
            width=1200,
            title_text=f'Identified start and end times of VNS activation using power in frequency band ({frequency}) for participant {participant_str} in condition {condition_str}',
            xaxis_title='Time (s)',
            yaxis_title='EMG', 
            showlegend=False
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_vns-activation-on-emg-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    return vns_start_end_markers



def ecg_response_to_vns(participant_str, condition_str, vns_start_end_markers, rpeak_session_dict, plot_hr_with_vns=False):
    """
    Function to calculate the heart rate response to VNS activation periods. Takes the identified VNS markers and the ECG data for each task session and calculates the mean and standard deviation of ECG parameters for the periods inside and outside the VNS activation per task session. Returns the raw ECG-related data for inside and outside VNS periods, and a summary DataFrame with the calculated statistics.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers).
    - condition_str: string, identifier of condition order (set to 2 numbers).
    - vns_start_end_markers: dict, contains a dataframe (value) in which the start (index and relative time), end (index and relative time) and duration (absolute time) of the VNS activation is identified and stored for a given task session (keys).
    - rpeak_session_dict: dict, contains the indices of detected r-peaks in the ECG trace (values) for each task session (keys).
    - plot_hr_with_vns: boolean, if true, will plot instantaneous HR across each task session containing VNS, with the start and the end of VNS activation marked.  
    """
    inside_vns_dict = {}
    outside_vns_dict = {}

    # Check if the HR data exists for the task session
    for task_session, events_df in vns_start_end_markers.items():
        if task_session in rpeak_session_dict:  
            hr_df = rpeak_session_dict[task_session].copy()  

            hr_df['inside_event'] = False
            for _, row in events_df.iterrows():
                start, end = row['start_time'], row['end_time']

                hr_df.loc[(hr_df['rpeak_rel_time'] >= start) & (hr_df['rpeak_rel_time'] <= end), 'inside_event'] = True
            

            inside_df = hr_df[hr_df['inside_event']].drop(columns=['inside_event'])
            outside_df = hr_df[~hr_df['inside_event']].drop(columns=['inside_event'])
            
            # Store these DataFrames in respective dictionaries
            inside_vns_dict[task_session] = inside_df
            outside_vns_dict[task_session] = outside_df


    hr_summary_stats = []

    # Summary stats for inside VNS periods
    for task_session, inside_df in inside_vns_dict.items():
        
        hr_summary_stats.append({
            'task_session': task_session,
            'type': 'inside',
            'r_r_interval_mean': inside_df['r_r_intervals'].mean(),
            'r_r_interval_std': inside_df['r_r_intervals'].std(),
            'instant_hr_mean': inside_df['instant_hr'].mean(),
            'instant_hr_std': inside_df['instant_hr'].std(),
        })
        
        # Summary stats for outside VNS periods
        if task_session in outside_vns_dict:
            outside_df = outside_vns_dict[task_session]
        
            hr_summary_stats.append({
                'task_session': task_session,
                'type': 'outside',
                'r_r_interval_mean': outside_df['r_r_intervals'].mean(),
                'r_r_interval_std': outside_df['r_r_intervals'].std(),
                'instant_hr_mean': outside_df['instant_hr'].mean(),
                'instant_hr_std': outside_df['instant_hr'].std(),
            })


    vns_hr_summary_df = pd.DataFrame(hr_summary_stats)

    avg_hr = (
        vns_hr_summary_df
        .groupby('type')['instant_hr_mean']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    # Create a subplot figure for each task session in VNS list
    if plot_hr_with_vns:
        task_sessions = vns_start_end_markers.keys()

        fig = make_subplots(rows=len(task_sessions), cols=1, 
                            subplot_titles=[f"Task Session {ts}" for ts in task_sessions])


        for i, task_session in enumerate(task_sessions, start=1):
            if task_session in rpeak_session_dict: 
                hr_df = rpeak_session_dict[task_session]
                events_df = vns_start_end_markers[task_session]
                
                fig.add_trace(
                    go.Scatter(x=hr_df['rpeak_rel_time'], y=hr_df['instant_hr'], mode='lines', line=dict(color='blue'),
                            name=f"HR - Task Session {task_session}"),
                    row=i, col=1
                )
                
                # Add vertical lines for VNS start and end times
                for _, row in events_df.iterrows():
                    start, end = row['start_time'], row['end_time']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[start, start], y=[min(hr_df['instant_hr']), max(hr_df['instant_hr'])],
                            mode='lines', line=dict(dash='dot', color='green'), 
                            name=f"Start (Session {task_session})"
                        ),
                        row=i, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[end, end], y=[min(hr_df['instant_hr']), max(hr_df['instant_hr'])],
                            mode='lines', line=dict(dash='dot', color='red'), 
                            name=f"End (Session {task_session})"
                        ),
                        row=i, col=1
                    )

        fig.update_layout(
            title=f"Instantaneous heart rate across task sessions with VNS activation markers for participant {participant_str} in condition {condition_str}",
            xaxis_title="Time (s)",
            yaxis_title="Heart Rate (bpm)",
            height=500 * len(task_sessions), 
            showlegend=False
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_ecg-by vns-activation-by-task-session.png"
        fig.write_image(os.path.join(desktop_path, file_name))

        fig.show()

    return vns_hr_summary_df, inside_vns_dict, outside_vns_dict, avg_hr