import pandas as pd
import numpy as np
import os
import sys
from itertools import zip_longest
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

sys.path.append('/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil



def identifying_best_eye (data_eyelink, file_name, file_dir, session):
    """
    Function to identify the best eye to use in pupil analysis. Based on Florent's pick_best_eye function, but less generalisable. Loads the data using pupil class methods, then compares the amount of missing data (nan values) between eyes to determine the best eye to use. Returns a string variable describing the best eye. 

    Parameters:
    - data_eyelink: data_eyelink: class 'pupil' object, instantiation of the 'pupil' class. Directories within objject can call specific methods and objects, such as data storage objects (e.g. data_eyelink.data call dataframe containing pupil data, etc).
    - file_name: string, name of file to load pupil data from. 
    - file_dir: string, higher level directories housing file.
    - session: string, session number (zero-padded to two digits) 

    Returns:
    - best_eye: string, eye that contains the least missing data, 'left' or 'right' 
    """
    print(f"Identifying best eye for Session {session}. Processing...")

    data_eyelink.import_data(file_name, events_warner='MSG', directory = file_dir, eye='right', col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='eyelink')

    # Returns the number of NaN values in the file for the left and right eye diameter
    nan_left = data_eyelink.nb_diam_nan_left
    nan_right = data_eyelink.nb_diam_nan_right

    if nan_left > nan_right:
        best_eye = 'right'
        print(f"Session {session}: The best eye is: right")
    else:
        best_eye = 'left'
        print(f"Session {session}: The best eye is: left")
    
    return best_eye




def load_and_process_pupil_data(directory_path, participant_str, list_subject_selected, condition_task_sessions, from_session, to_session, dict_onsets, dict_num, session_list = None, outlier_limit_std = 10, outlier_correct_average_amount = 15,  window_start_before = 0.25, window_end_after = 2, baseline_cor = [-0.25, 0], lpf = [5], blink_ratio_threshold = 0.20, select_eye = True, remove_epochs = True, z_score = True, plot_interpolation_process = False):
    """
    Function to load and preprocess pupil data. Preprocessing involves defining blinks, outliers and the interpolation of pupil data. Preprocessing taken from preprocess_eyedata script, created by Florent, Maëva, Aziza and Yannaël. Option to plot the raw and interpolated data. If dictionaries passed defining events of interest, will compute epochs around defined outcomes of interest (e.g. change points, high and low surprise, etc). If epochs computed, will calculate the mean, std, sem across each identified epoch for all task sessions.

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - dict_onsets: dict, contains timestamps of outcomes of interest (e.g. change points, high and low surprise, etc). Keys are task session and values are time stamps within that task session. (Default = None)
    - dict_nums: dict, contains a cumulative count of the number of outcomes of interest identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session. (Default = None)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session. (Default = None)
    - outlier_limit_std: int, defines the threshold of the standard deviation of the derivative when preprocessing the pupil data. (Default = 10)
    - outlier_correct_average_amount: int, the number of data points to the left and at the right of the artifact which will replace the value of the artifact by the value of the mean of the data around it. (Default = 15).
    - window_start_before: int, value defining the start of the epoch window in time (seconds). (Default = 0.3)
    - window_end_after: int, value defining the end of the epoch window in time (seconds). (Default = 1.2)
    - baseline_cor: list, contains the start and end time of the baseline correction window. If left empty (e.g. []), no baseline correction will be applied. (Default = [-0.2, 0])
    - lpf: list, cutoff (Hz) to low pass filter the data with. If [], do nothing. (Default = 5 Hz)
    - blink_ratio_threshold:, float, value determining the threshold criteria for the removal of an epoch due to excessive blink ratio. (Default = 0.2)
    - select_eye: boolean, if true, will use a predfined eye (Default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (Default = True)
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - z_score: boolean, option to z_score data before computing epochs. Default = True.
    - plo_interpolation_process: boolean, option to plot the mean and standard deviation across epochs using Plotly. (Default = True) 

    Returns:
    - preprocessed_pupil_df: pd.DataFrame, contains the pupil data, preprocessed to interpolate across identified blinks, recorded from each task session
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session
    - pupil_ses_df: pd.DataFrame, contains each task session start time, end time and duration.
    - blink_df: pd.DataFrame, contains the start and end times of each identified blink within a task session. Also contains the duration of each blink, plus the relative and absolute number of blinks, within and across task sessions.   

    If dict_onsets present:
    - epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics and timestamp across epochs.
    - blink_ratio_epoch_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch.

    If remove_epochs = True:
    - epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within. 
    """
    # Running data through methods within class pupil taken from preprocess_eyedata. 
    # Initialise class
    data_eyelink = pupil()

    # Create lists to store data for each task session
    pupil_data_ses_list, pupil_trig_ses_list, pupil_ses_list, blink_ses_list, used_eye_list = [], [], [], [], []
    pre_process_data_list, epoched_data_list, epoched_data_time_list, blink_ratio_epoch_list = [], [], [], []

    sessions_to_process = [str(session) for session in range(from_session, to_session + 1)]

    # Loop over sessions to extract and process pupil data
    for session in sessions_to_process:
        file_dir = f"{directory_path}"
        file_found = False  # Flag to track if a valid file is found

        file_name = f"Prob{participant_str}_{session}.edf"
        file_path = f"{file_dir}/{file_name}"

        if os.path.exists(file_path):
            print(f"File found for session {session}: {file_path}")
            file_found = True  # Mark the file as found
            # Defining the eye to be used in the pupil analysis - if selec_eye right is selected as default
            if select_eye:
                print(f"Eye pre-selected for Session {session}. The chosen eye is: right")
                used_eye_list.append(select_eye)

                data_eyelink.import_data(
                    file_name,
                    events_warner='MSG',
                    directory=file_dir,
                    eye='right',
                    col=[0, 1, 2],
                    col_name=['x_pos', 'y_pos', 'diameter'],
                    eyetracker='eyelink' # Updating eyetracker
                )
            else: # If identifying the best eye, files will be run to find the eye with the least missing data
                best_eye = identifying_best_eye(
                    data_eyelink=data_eyelink,
                    file_name=file_name,
                    file_dir=file_dir,
                    session=session
                )

                used_eye_list.append(best_eye)

                data_eyelink.import_data(
                    file_name,
                    events_warner='MSG',
                    directory=file_dir,
                    eye=best_eye,
                    col=[0, 1, 2],
                    col_name=['x_pos', 'y_pos', 'diameter'],
                    eyetracker='tobii'
                )

        if not file_found:
            print(f"No valid file found for session {session}")

        # Extract relevant data from class
        trigger_data = data_eyelink.events
        blink_start = data_eyelink.blink_start
        blink_end = data_eyelink.blink_end

        # Create lists to extract data from class objects such that the data can be stored in a dataframe. For example, trigger_data contains a list of dictionaries, which contain keys: 'time' and 'msg', within which values and lists of strings containing the information within that line of the .txt file. 
        outcome_data, session_start, session_end = [], [], []

        # Extracts data based on contents of the strings ('msg_str') within the value list of strings.
        for item in trigger_data:
            msg_str = item['msg']  # Already a string

            if 'Stim A' in msg_str:
                outcome_data.append({'time_stamp': item['time'], 'outcome': 'A'})
            elif 'Stim G' in msg_str:
                outcome_data.append({'time_stamp': item['time'], 'outcome': 'G'})
            elif 'T0received' in msg_str:
                session_start.append(item['time'])
            elif 'REC_STOP' in msg_str:
                session_end.append(item['time'])


        # Creates a dataframe to store the extracted trigger information for each task session 
        pupil_trig_ses = pd.DataFrame(outcome_data)
        pupil_trig_ses['time_stamp'] = pd.to_numeric(pupil_trig_ses['time_stamp'], errors='coerce').astype('Int64')
        pupil_trig_ses['task_session'] = session

        # Deal with or raise error due to missing triggers
        if not outcome_data:
            print(f"ERROR: Outcome triggers missing. May need to remove this task session.")
        if not session_start:
            print(f"ERROR: Missing trigger and list is empty: session_start = {session_start}")
        if not session_end:
            print(f"ERROR: Missing trigger and list is empty: session_end = {session_end}")
            session_end.append(data_eyelink.events[-1]['time'])  # Use the last event time as session end
            print(f"Added the last time stamp within extracted events as the session end time: {session_end}")

        # Creates a dataframe to store the extracted session characteristics information for each task session 
        pupil_ses = pd.DataFrame({
            'task_session': session, 
            'start': session_start,
            'end': session_end
        })

        # Check if blink_end has exactly one extra item
        if len(blink_end) == len(blink_start) + 1:
            blink_end = blink_end[:-1] 
            data_eyelink.blink_end = blink_end # Remove the last item from blink_end

        elif len(blink_start) != len(blink_end):
            raise ValueError("blink_start and blink_end lengths differ by more than 1. Cannot resolve automatically.")

        # Creates a dataframe to store the extracted blink information for each task session 
        blink_ses = pd.DataFrame({
            'task_session': session,
            'blink_start': blink_start,
            'blink_end': blink_end
        })
        blink_ses['blink_dur'] = (blink_ses['blink_end'] - blink_ses['blink_start']) / 1000

        # Amended from original function
        pupil_data = data_eyelink.data.copy()

        # Append the data to the empty list to store the data for that task session
        pupil_data_ses_list.append(pupil_data)
        pupil_trig_ses_list.append(pupil_trig_ses)
        pupil_ses_list.append(pupil_ses)
        blink_ses_list.append(blink_ses)

        # Preprocess the data. If needed, can plot the preprocessed pupil time series. To see the blink ratio use data_eyelink.epoch_blink_ratio. For more complete summary can insert data_eyelink.epochs_info (provides a data frame with the epoch number, the blink ratio, NaN ratio per epoch and conditions [rejected etc])
        data_eyelink.preprocess(eyetracker='eyelink', lpf=lpf, plot=False, remove_outlier=True, outlier_limit_std=outlier_limit_std, outlier_correct_average_amount=outlier_correct_average_amount)   

        processed_data = data_eyelink.data.copy()
        processed_data['task_session'] = session  
        pre_process_data_list.append(processed_data)

        if dict_onsets is not None:
            data_eyelink.epoch_eyelink(dict_onsets[session], variable='diameter_int', decimate=1, reject=False, blink_ratio_thd=blink_ratio_threshold, before=window_start_before, after=window_end_after, baseline=baseline_cor, conditions={'event_num': dict_num[session]}, z_score_data=z_score)
 
            # Check to ensure the correct number of epochs
            print(f"For participant {participant_str} in session {session} there are: {len(data_eyelink.epochs)} epochs")
            print(data_eyelink.epochs_info)

            # Appending dataframes containing epoched data to list for each task session
            epoched_data = pd.DataFrame([epoch for epoch in data_eyelink.epochs])
            epoched_data_list.append(epoched_data)

            epoched_time_data = pd.DataFrame([epoch for epoch in data_eyelink.epochs_times])
            epoched_data_time_list.append(epoched_time_data)

            # Append list to blink_ratio list 
            blink_ratio_epoch_list.append(data_eyelink.epochs_info['blink_ratio'].values)

    # Concatenate pre processed pupil dataframes (trigger information, blink information, task session timings, epoched data, etc) across task sessions
    preprocessed_pupil_df = pd.concat(pre_process_data_list, ignore_index=True)

    pupil_trig_df = pd.concat(pupil_trig_ses_list, ignore_index=True)
    pupil_trig_df['outcome_num'] = pupil_trig_df.groupby('task_session').cumcount() 
    pupil_trig_df['cum_outcome_num'] = range(len(pupil_trig_df))
    pupil_trig_df['time_stamp'] = pupil_trig_df['time_stamp'].astype('int64')
 
    pupil_ses_df = pd.concat(pupil_ses_list, ignore_index=True)
    pupil_ses_df['duration'] = (pupil_ses_df['end'].astype(int) - pupil_ses_df['start'].astype(int)) / 1000
    if select_eye:
        pupil_ses_df['used_eye'] = 'right'
    else:
        pupil_ses_df['used_eye'] = used_eye_list

    blink_df = pd.concat(blink_ses_list, ignore_index=True)
    blink_df['blink_count'] = blink_df.groupby('task_session').cumcount() 
    blink_df['cum_blink_count'] = range(len(blink_df))

    if hasattr(data_eyelink, 'epochs_info'):
        # Generate dataframe with blink_ratio for each epoch vertically stacked
        flattened_array = np.concatenate(blink_ratio_epoch_list)

        blink_ratio_epoch_df = pd.DataFrame(flattened_array, columns=['blink_ratio'])
        blink_ratio_epoch_df['above_threshold'] = blink_ratio_epoch_df['blink_ratio'] >= blink_ratio_threshold

        # Collating the timelocked epoched pupil data to calculate descriptive statistics for plotting
        epoched_pupil_df = pd.concat(epoched_data_list, ignore_index=True).T

        # Removing epochs in which interpolation exceeds defined threshold (and in which the shape of the epoch is off due to sampling issues)
        if remove_epochs:
            # Get indices of epochs (columns) to remove due to excessive blinking
            epochs_to_remove = list(blink_ratio_epoch_df.index[blink_ratio_epoch_df['above_threshold']])

            num_excess_blink_columns = len(epochs_to_remove)

            # Removing columns that have null values due to sampling issues
            nan_columns = epoched_pupil_df.columns[epoched_pupil_df.isna().any()].tolist()
            print(f'Number of columns containing NaN values: {len(nan_columns)}')

            # For each column with NaN values, check if the NaN is in the last row. If it is then pad the null value with the data in the previous row for that column 
            # For each column with NaN values, check if NaNs are in the last 3 rows max
            for col in nan_columns:
                series = epoched_pupil_df[col]
                # Reverse the series and count how many NaNs are at the end
                reversed_series = series[::-1]
                trailing_nans = reversed_series.isna().cumprod().sum()

                if trailing_nans > 0:
                    if trailing_nans <= 3:
                        # Get the last non-NaN value to use for padding
                        last_valid_value = series.iloc[:-trailing_nans].dropna().iloc[-1]
                        # Fill all trailing NaNs with the last valid value
                        epoched_pupil_df.loc[epoched_pupil_df.index[-trailing_nans:], col] = last_valid_value
                    else:
                        raise ValueError(f"Column '{col}' has {trailing_nans} trailing NaNs — exceeds limit of 3.")

            # Final check for any remaining NaNs
            remaining_nans = epoched_pupil_df.columns[epoched_pupil_df.isna().any()].tolist()
            print(f'Number of columns containing NaN values after rectifying sampling issue: {len(remaining_nans)}')

            # Remove only columns that had excessive blinking
            epochs_to_remove = sorted(set(epochs_to_remove)) 
            epoched_pupil_df.drop(columns=epochs_to_remove, inplace=True)

            expected_epochs = 420*len(sessions_to_process)

            print(f"Removed {len(epochs_to_remove)} epochs due to excessive blinking ({num_excess_blink_columns}).")
            print(f"The number of epochs removed represents {len(epochs_to_remove)/expected_epochs*100:.2f}% of the total epochs.")
            print(f"Columns removed: {epochs_to_remove}")
        else:
            epochs_to_remove = None

        numeric_data = epoched_pupil_df.iloc[:, :]

        epoched_pupil_df['mean'] = numeric_data.mean(axis=1)
        epoched_pupil_df['std'] = numeric_data.std(axis=1)
        epoched_pupil_df['sem'] = epoched_pupil_df['std'] / np.sqrt(numeric_data.shape[1])

        # Trim the DataFrame to match the length
        target_length = len(data_eyelink.epochs_times)
        epoched_pupil_df = epoched_pupil_df.iloc[:target_length].reset_index(drop=True)
        epoched_pupil_df['time'] = data_eyelink.epochs_times

    else:
        epoched_pupil_df = None
        blink_ratio_epoch_df = None
        epochs_to_remove = None 

    # Bespoke option to visulaize interpolation process (pupil method 'preprocess' has an option, but I didn't like the visualisation)
    if plot_interpolation_process:
        unique_sessions = preprocessed_pupil_df['task_session'].unique()
        num_sessions = len(unique_sessions)

        # Create subplots, one for each task session
        fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=False,
                            subplot_titles=[f'Preprocessing of pupil diameter in task session {session}' for session in unique_sessions])

        # Add pupil traces to each subplot
        for subplot_idx, session in enumerate(unique_sessions, start=1):
            session_data = preprocessed_pupil_df[preprocessed_pupil_df['task_session'] == session]

            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['diameter'], mode='lines', name='Original Pupil Diameter', showlegend=False,
                        line=dict(color='rgba(0, 0, 255, 0.5)')), row=subplot_idx, col=1)
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['diameter_int'], mode='lines', name='Interpolated Pupil Diameter', showlegend= False,
                        line=dict(color='rgba(255, 0, 0, 1)', dash='dot')), row=subplot_idx, col=1) 
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['x_pos_int'], mode='lines', name='Interpolated x position', showlegend=False,
                        line=dict(color='rgba(0, 255, 0, 0.75)', dash='dash')), row=subplot_idx, col=1) 
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['y_pos_int'], mode='lines', name='Interpolated y position', showlegend=False,
                        line=dict(color='rgba(255, 240, 0, 0.75)', dash='dash')), row=subplot_idx, col=1) 

            # Highlight interpolation periods
            interpolating = False
            start_idx = None

            for i in range(len(session_data)):
                if session_data['track_int'].iloc[i] == 1 and not interpolating:
                    # Start of interpolation
                    start_idx = session_data['time_stamp_rel_s'].iloc[i]
                    interpolating = True
                elif session_data['track_int'].iloc[i] == 0 and interpolating:
                    # End of interpolation
                    end_idx = session_data['time_stamp_rel_s'].iloc[i]
                    fig.add_shape(type="rect",
                                x0=start_idx, x1=end_idx, y0=0, 
                                y1=max(session_data['diameter_int']) + 0.5,
                                fillcolor="LightPink", opacity=0.3, line_width=0,
                                xref=f'x{subplot_idx}',  
                                yref=f'y{subplot_idx}')  
                    interpolating = False

        fig.update_layout(height=600*num_sessions, 
                        width=1000,
                        title_text=f'Preprocessed pupil diameter and x/y position across task sessions for participant {participant_str}',
                        xaxis_title='Time (s)',
                        yaxis_title='Pupil Diameter (a.u)')
        

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_pupil-interpolation-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()

    return preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove



def create_pupil_epoch_dicts(pupil_event_aligned_df):
    """
    Function to generate dictionaries of the pupil data 'time_stamps' (and nums) based on some event of interest. Takes the dataframe with pupil and event indices aligned and then generates a pair of dictionaries, with task sessions (z-filled) as keys and timestamps/cumulative count of events as values. 

    Parameters:
    - pupil_event_aligned_df: pd.DataFrame, contains the dataframe in which pupil data and event of interest have been aligned. Requirement is the 'time_stamp' from the pupil trigger dataframe. 

    Returns:
    - onsets: dict, contains timestamps of all outcomes taken from the pupil data, across all task sessions. Keys are task session and values are time stamps within that task session. 
    - nums: dict, contains the cumulative count of events taken from the pupil data, across all task sessions. Keys are task session and values are time stamps within that task session. 
    """
    onsets = {}
    nums = {}

    for session, group in pupil_event_aligned_df.groupby('task_session'):
        # Extract the timeflow values for the current group (session)
        onset_values = group['time_stamp'].tolist()

        # Create a cumulative count list for the current group (session)
        num_values = list(range(1, len(group) + 1))

        # Define dictionaries used to define epochs
        onsets[session] = onset_values
        nums[session] = num_values

    return onsets, nums


def all_outcome_pupil_epochs(directory_path, participant_str, list_subject_selected, condition_task_sessions, from_session, to_session, pupil_trig_df, baseline_cor = [-0.25, 0], variable_df=None, variable_column=None, select_eye = True, remove_epochs = True, plot_averaged_pupil_response = False, plot_raster_pupil_response = False):
    """
    Function to preprocess and analysis pupil data for each outcome across all task sessions. Uses load_and_process_pupil_data method to load and run the preporocessing analysis. There is an option to plot the mean response from all outcomes across all task sessions, as well as a raster plot from each outcome. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data.
    - variable_df: pd.DataFrame, optional variable of interest associated with outcomes for sorting the heatmap (default = None).
    - variable_column: string, defines the column containing the variable of interest used for sorting heatmap.
    - select_eye: boolean, if true, will use a predfined eye (default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye.
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - plot_average_pupil_response: boolean, option to plot the mean and sem across each output using Plotly. (Default = False).
    - plot_raster_pupil_response: boolean, option to plot raster plot for each output using Plotly. (Default = False).

    Returns:
    - outcome_onsets: dict, contains timestamps of all outcomes from all task sessions. Keys are task session and values are time stamps within that task session. 
    - outcome_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - outcome_epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean, standard deviation and sem across all outcome defined epochs. Includes the timestamp.
    - outcome_blink_ratio_cp_df: pd.DataFrame, contains the blink ratio information for each generated epoch around outcome observations
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within. 

    """
    # Creating dictionaries for storage of each outcome time and number 
    outcome_onsets, outcome_nums = create_pupil_epoch_dicts(pupil_trig_df)

    # Checking the number of outcomes
    for session, events in outcome_nums.items():
      print(f"In task session {session} there are {len(events)} outcome observations") 

    # Analysing pupil data based on identified outcome indices
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_onsets, dict_num=outcome_nums, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=baseline_cor, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

    outcome_epochs_to_remove_df = pd.DataFrame({
        'epochs': outcome_epochs_to_remove,
        'participant': participant_str
    })

    # Match outcomes to the input variable of interest
    if variable_df is not None and variable_column is not None:
        print(f"Order of heatmap defined by {variable_column}")
        variable_of_interest = variable_df[variable_column]
        variable_of_interest_cleaned = variable_of_interest.drop(outcome_epochs_to_remove)
        variable_of_interest_cleaned = variable_of_interest_cleaned.sort_values().dropna()

        heatmap_df = outcome_epoched_pupil_df.iloc[:, :-4].copy()
        heatmap_df = heatmap_df[variable_of_interest_cleaned.index]

        pupil_epoch_matrix = [heatmap_df.iloc[:, i].values for i in range(heatmap_df.shape[1])]
        print(f"Number of epochs added to heatmap: {len(pupil_epoch_matrix)}")

        title = f'Epoched pupil diameter heatmap across all outcome observations for participant {participant_str} (ordered by {variable_column})'

    else:
        heatmap_df = outcome_epoched_pupil_df.copy()

        pupil_epoch_matrix = [heatmap_df.iloc[:, i].values for i in range(heatmap_df.shape[1] - 4)]
        print(f"Number of epochs added to heatmap: {len(pupil_epoch_matrix)}")

        title = f'Epoched pupil diameter heatmap across all outcome observations for participant {participant_str}'


    # Option to plot the average pupil response across all outcomes
    if plot_averaged_pupil_response:
        if len(baseline_cor) == 0:
            title = f"Baseline uncorrected pupil response to each observations for participant {participant_str} (Mean +/- SEM)"
        else:
            title = f"Baseline corrected pupil response to each observations for participant {participant_str} (Mean +/- SEM)"
              
        fig_pupil_avg = go.Figure()

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_epoched_pupil_df['time'],
            y=outcome_epoched_pupil_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='Mean',
            showlegend=False
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_epoched_pupil_df['time'],
            y=outcome_epoched_pupil_df['mean'] + outcome_epoched_pupil_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_epoched_pupil_df['time'],
            y=outcome_epoched_pupil_df['mean'] - outcome_epoched_pupil_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)


        fig_pupil_avg.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)',
            height=950,
            width=1800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_pupil-all-outcomes.png"

        file_path = os.path.join(plot_dir, file_name)

        fig_pupil_avg.write_image(file_path)


        fig_pupil_avg.show()

    # Option to plot rasterplot from each pupil epoch
    if plot_raster_pupil_response:
       
        fig = px.imshow(pupil_epoch_matrix, 
                        labels=dict(x="Time (s)", y="Outcome number", color="Pupil Diameter (z-score)"),
                        x=outcome_epoched_pupil_df['time'].values,  
                        aspect="auto",  
                        origin="lower"  
        )

        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(title=title,
                          height=950,
                          width=1800,
                          showlegend=True,
                          margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_pupil-all-outcomes-heatmap.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    return outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df
