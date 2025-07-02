import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

sys.path.append('/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Analysis/external_reference_functions')
from preprocess_eyedata import pupil



def check_pupil_file_triggers(file_path, threshold=2):
    """
    Function to check whether a pupil file has an obvious issue. This function reads in the .txt file and checks whether the number of triggers in the file that are consecutive with each other does not exceed a set threshold. If yes, may be provide grounds to remove this file as something has obviously gone wrong with the data capture.

    Parameters:
    - file_path: string or Path, reprsents the directory of the file.
    - threshold: int, number of consecutive triggers deemed to signal an issue with the file. 

    Returns:
    - consecutive_count > thrshold: boolean, if true, the number of consecutive triggers in the file exceeds the predefined threshold
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    consecutive_count = 0
    for i in range(1, len(lines)):
        if 'MSG' in lines[i-1] and 'MSG' in lines[i]:
            consecutive_count += 1

    return consecutive_count > threshold



def pupil_data_file_check(data_tob, pupil_trig_ses, blink_ses, pupil_ses, pupil_end, file_path, session, file_size_threshold = 2, plot_original_data = True, plot_reduced_data = True):
    """
    Function to ensure that the pupil data loaded from files does not contain erroneous data. Also checks the number of triggers within the file.  Currently function is based on the large files that were found after the first couple of participants, but should be adapted to rectify any other issues that crop up. 

    Parameters:
    - data_tob: class 'pupil' object, instantiation of the 'pupil' class. Directories within objject can call specific methods and objects, such as data storage objects (e.g. data_tob.data call dataframe containing pupil data, etc).
    - pupil_trig_ses: pd.DataFrame, contains the 
    - blink_ses: pd.DataFrame, contains the start and end times of each blink, for each task session, identified in the import_data method of the pupil class. Also contains the duration of each blink.
    - pupil_ses: pd.DataFrame, contains the start and end times of task session, based on the 'start' and 'end' triggers inserted within each file. The duration of each session is calculated.
    - pupil_end: list, contains the information of the last trigger inserted into the file. 
    - file_path: string, combined from the name of the directory and the file. 
    - session: string, session identifier (two numbers, e.g. '01') 
    - file_size_threshold: float, defines the size of file above which would be cause for concern, which should be raised and investigation into the file contents is needed (default = 2 mb).
    - plot_original_data: boolean, if true will plot the raw pupil diameter data taken from the file. 
    - plot_reduced_data: boolean, if true will plot the pupil data across the reduced time interval now defined within that file.

    Returns:
    -  data_tob: class 'pupil' object, instantiation of the 'pupil' class. Directories within objject can call specific methods and objects, such as data storage objects (e.g. data_tob.data call dataframe containing pupil data, etc). Data_tob.data, data_tob.blink_start and data_tob.blink_end should now be inline with the new start and end points of the session. 
    - pupil_data: pd.Dataframe, contains pupil data updated based on the outcome of the check. If no check required, passes on a copy of data_tob.data.   
    - blink_ses: pd.Dataframe, contains blink data updated based on the outcome of the check. 
    """
    # Initial trigger check. Checks for instances of all known triggers inserted into the file
    if len(pupil_trig_ses['outcome']) == 75:
        print(f"Correct number of outcome triggers within file {session}: {len(pupil_trig_ses['outcome'])}")
        outcome_1 = pupil_trig_ses[pupil_trig_ses['outcome'] == 1]
        print(f"Number of outcomes = 1: {len(outcome_1['outcome'])}")
        outcome_0 = pupil_trig_ses[pupil_trig_ses['outcome'] == 0]
        print(f"Number of outcomes = 0: {len(outcome_0['outcome'])}")
    else:
        print(f"ERROR: Number of outcome triggers in file {session} do not meet the expected total. Please check file {session}.")

    if len(pupil_ses['start']) == 1 & len(pupil_ses['end'] == 1):
        print(f"Start and end triggers found in file {session}")
    else:
        print(f"ERROR: Start and/or end triggers are not correctly inserted in file {session}")

    if len(pupil_end) == 1:
        print(f"End recording trigger found in file {session}")
    else:
        print(f"ERROR: No end recording trigger found in file {session}")  

    total_triggers = len(pupil_trig_ses['outcome']) + len(pupil_ses['start']) + len(pupil_ses['end']) + len(pupil_end)
    print(f"The total number of triggers in file {session} is: {total_triggers}")  

    # Check of file data contents is then performed. Initiatally based on erroneously large file sizes 
    def get_file_size(file_path = file_path):
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        return file_size_mb

    file_size_check = get_file_size(file_path=file_path)

    # After finding the size of the file, checks it against a set threshold. Threshold can be adjusted accordingly
    if file_size_check >= file_size_threshold:
        print(f'File {session} identified as excessively large. Processing...')
        # Plots the content of the large file that is saved in the 'pupil data' object 
        if plot_original_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_tob.data['time_stamp_rel_s'], y=data_tob.data['diameter'], mode='lines', name=f'pupil diameter'))

            fig.update_layout(
                    xaxis_title='Time (s)',
                    yaxis_title='Pupil Diameter (mm)')

            fig.show()

        # After identifying the large file, checks if any of the data in the file (the time stamp) is duplicated. 
        if data_tob.data['time_stamp'].duplicated().any():
            print(f"There are duplicated data/time stamps in file {session}")

            # The reduction of the large file is primarily based on triggers. 'pupil_ses' is a data frame containing the start and end times of each file based on the 'start' and 'end' triggers inserted into the file
            start_value = int(pupil_ses['start'].iloc[0])
            end_value = int(pupil_ses['end'].iloc[0])

            # Uses the start and end triggers to define the range across pupil data will be reduced down to
            start_indices = data_tob.data.loc[np.isclose(data_tob.data['time_stamp'], start_value)].index
            end_indices = data_tob.data.loc[np.isclose(data_tob.data['time_stamp'], end_value)].index

            # Given the duplication of data, if there are more than two values defining the location of where the 'start' and 'end' triggers should be location, takes the first instance. 
            if len(start_indices) > 1 and len(end_indices) > 1:
                start_idx, end_idx = start_indices[0], end_indices[0]

                # Overwrites the pupil data object based on defined range. Given the need to epoch around data points, the range based on the 'start' and 'end' trigger is extended a little bit in both directions. Also makes this fle relatively similar in size to other files. Adjustment values are pretty arbitary, may need to be adjusted in future. 
                data_tob.data = data_tob.data.loc[start_idx-70:end_idx+60]

                data_tob.data = data_tob.data.reset_index(drop=True)
            
                # Get the last timestamp in pupil data
                last_pupil_time = data_tob.data['time_stamp'].iloc[-1]
            
                # Adjust blink_ses to ensure no blink start exceeds last_pupil_time
                blink_ses = blink_ses[blink_ses['blink_start'] <= last_pupil_time]

                # Align blink_ses with pupil data by time range
                blink_ses = blink_ses[(blink_ses['blink_start'] >= data_tob.data['time_stamp'].iloc[0]) & 
                                    (blink_ses['blink_start'] <= data_tob.data['time_stamp'].iloc[-1])]
            
                blink_ses = blink_ses.drop_duplicates()
                
                # Overwrites blink start and blink end object in pupil class 
                data_tob.blink_start = np.array(blink_ses['blink_start'])
                data_tob.blink_end = np.array(blink_ses['blink_end'])

                # Plots the reduced data for visualisation
                if plot_reduced_data:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data_tob.data['time_stamp_rel_s'], y=data_tob.data['diameter'], mode='lines', name=f'pupil diameter'))

                    fig.update_layout(
                        xaxis_title='Time (s)',
                        yaxis_title='Pupil Diameter (mm)')

                    fig.show()

                print(f'Multiple indices for the start and end triggers identified. File {session} has been reduced to data between first instances of start and end triggers')
            else:
                print(f'File {session} excessively large, but multiple indices of start and end triggers not found ')
    
        else:
            print("There is no duplicated data found in file {session}. But file {session} excessively large")
    else:
        print(f"File {session} did not exceed the file size threshold.")
    
    pupil_data = data_tob.data.copy()
    pupil_data['task_session'] = session
    
    # Returns both the self object of the pupil class and a copy of the data 
    return data_tob, pupil_data, blink_ses



def identifying_best_eye (data_tob, file_name, file_dir, session):
    """
    Function to identify the best eye to use in pupil analysis. Based on Florent's pick_best_eye function, but less generalisable. Loads the data using pupil class methods, then compares the amount of missing data (nan values) between eyes to determine the best eye to use. Returns a string variable describing the best eye. 

    Parameters:
    - data_tob: data_tob: class 'pupil' object, instantiation of the 'pupil' class. Directories within objject can call specific methods and objects, such as data storage objects (e.g. data_tob.data call dataframe containing pupil data, etc).
    - file_name: string, name of file to load pupil data from. 
    - file_dir: string, higher level directories housing file.
    - session: string, session number (zero-padded to two digits) 

    Returns:
    - best_eye: string, eye that contains the least missing data, 'left' or 'right' 
    """
    print(f"Identifying best eye for Session {session}. Processing...")

    data_tob.import_data(file_name, events_warner='MSG', directory = file_dir, eye='right', col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='tobii')

    # Returns the number of NaN values in the file for the left and right eye diameter
    nan_left = data_tob.nb_diam_nan_left
    nan_right = data_tob.nb_diam_nan_right

    if nan_left > nan_right:
        best_eye = 'right'
        print(f"Session {session}: The best eye is: right")
    else:
        best_eye = 'left'
        print(f"Session {session}: The best eye is: left")
    
    return best_eye



def load_and_process_pupil_data(directory_path, participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, dict_onsets, dict_num, session_list = None, outlier_limit_std = 10, outlier_correct_average_amount = 15,  window_start_before = 0.25, window_end_after = 1.4, baseline_cor = [-0.25, 0], lpf = [5], blink_ratio_threshold = 0.20, select_eye = True, remove_epochs = True, z_score = True, plot_interpolation_process = False):
    """
    Function to load and preprocess pupil data. Preprocessing involves defining blinks, outliers and the interpolation of pupil data. Preprocessing taken from preprocess_eyedata script, created by Florent, Maëva, Aziza and Yannaël. Option to plot the raw and interpolated data. If dictionaries passed defining events of interest, will compute epochs around defined outcomes of interest (e.g. change points, high and low surprise, etc). If epochs computed, will calculate the mean, std, sem across each identified epoch for all task sessions.

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - dict_onsets: dict, contains timestamps of outcomes of interest (e.g. change points, high and low surprise, etc). Keys are task session and values are time stamps within that task session. (Default = None)
    - dict_nums: dict, contains a cumulative count of the number of outcomes of interest identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session. (Default = None)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session. (Default = None)
    - outlier_limit_std: int, defines the threshold of the standard deviation of the derivative when preprocessing the pupil data. (Default = 10)
    - outlier_correct_average_amount: int, the number of data points to the left and at the right of the artifact which will replace the value of the artifact by the value of the mean of the data around it. (Default = 15).
    - window_start_before: int, value defining the start of the epoch window in time (seconds). (Default = 0.25)
    - window_end_after: int, value defining the end of the epoch window in time (seconds). (Default = 1.4)
    - baseline_cor: list, contains the start and end time of the baseline correction window. If left empty (e.g. []), no baseline correction will be applied. (Default = [-0.25, 0])
    - lpf: list, cutoff (Hz) to low pass filter the data with. If [], do nothing. (Default = 5 Hz)
    - blink_ratio_threshold:, float, value determining the threshold criteria for the removal of an epoch due to excessive blink ratio. (Default = 0.2)
    - select_eye: boolean, if true, will use a predfined eye (Default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (Default = True)
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - z_score: boolean, option to z_score data before computing epochs. Default = True.
    - plo_interpolation_process: boolean, option to plot the mean and standard deviation across epochs using Plotly. (Default = True) 

    Returns:
    - preprocessed_pupil_df: pd.DataFrame, contains the pupil data, preprocessed to interpolate across identified blinks, recorded from each task session
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session
    - pupil_ses_df: pd.DataFrame, contains each task session start time, end time nan_ratio and duration.
    - blink_df: pd.DataFrame, contains the start and end times of each identified blink within a task session. Also contains the duration of each blink, plus the relative and absolute number of blinks, within and across task sessions.   

    If dict_onsets present:
    - epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics and timestamp across epochs.
    - blink_ratio_epoch_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch.

    If remove_epochs = True:
    - epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within. 
    """
    # Running data through methods within class pupil taken from preprocess_eyedata. 
    # Initialise class
    data_tob = pupil()

    # Create lists to store data for each task session
    pupil_data_ses_list, pupil_trig_ses_list, pupil_ses_list, blink_ses_list, used_eye_list = [], [], [], [], []
    pre_process_data_list, epoched_data_list, blink_ratio_epoch_list = [], [], []

    if session_list is not None:
        sessions_to_process = [str(session).zfill(2) for session in session_list]
    else:
        sessions_to_process = [str(session).zfill(2) for session in range(from_session, to_session + 1)]

    # Loop over sessions to extract and process pupil data
    task_names = ["AdalearnVNS", "AdalearnPupil"]

    # Loop over sessions to extract and process pupil data
    for session in sessions_to_process:
        file_dir = f"{directory_path}/ses-{condition_str}/eye"
        file_found = False  # Flag to track if a valid file is found

        for task in task_names:
            file_name = f"sub-{participant_str}_ses-{condition_str}_task-{task}_run-{session}_eyetracker.txt"
            file_path = f"{file_dir}/{file_name}"

            if os.path.exists(file_path):
                file_found = True  # Mark the file as found
                # Defining the eye to be used in the pupil analysis - if selec_eye right is selected as default
                if select_eye:
                    print(f"Eye pre-selected for Session {session}. The chosen eye is: right")
                    used_eye_list.append(select_eye)

                    data_tob.import_data(
                        file_name,
                        events_warner='MSG',
                        directory=file_dir,
                        eye='right',
                        col=[0, 1, 2],
                        col_name=['x_pos', 'y_pos', 'diameter'],
                        eyetracker='tobii'
                    )
                else: # If identifying the best eye, files will be run to find the eye with the least missing data
                    best_eye = identifying_best_eye(
                        data_tob=data_tob,
                        file_name=file_name,
                        file_dir=file_dir,
                        session=session
                    )

                    used_eye_list.append(best_eye)

                    data_tob.import_data(
                        file_name,
                        events_warner='MSG',
                        directory=file_dir,
                        eye=best_eye,
                        col=[0, 1, 2],
                        col_name=['x_pos', 'y_pos', 'diameter'],
                        eyetracker='tobii'
                    )
                break  # Exit the task loop once a valid file is processed

        if not file_found:
            print(f"No valid file found for session {session}")

        # Extract relevant data from class
        trigger_data = data_tob.events
        nan_ratio = data_tob.nan_ratio
        blink_start = data_tob.blink_start
        blink_end = data_tob.blink_end

        # Create lists to extract data from class objects such that the data can be stored in a dataframe. For example, trigger_data contains a list of dictionaries, which contain keys: 'time' and 'msg', within which values and lists of strings containing the information within that line of the .txt file. 
        outcome_data, session_start, session_end, pupil_end = [], [], [], []

        # Extracts data based on contents of the strings ('msg_str') within the value list of strings.
        for item in trigger_data:
            # Join the list of strings in 'msg' into a single string
            msg_str = ' '.join(item['msg'])
    
            if 'stim_1' in msg_str:
                outcome_data.append({'time_stamp': item['time'], 'outcome': 1})
            elif 'stim_0' in msg_str:
                outcome_data.append({'time_stamp':item['time'], 'outcome': 0})
            elif 'starts' in msg_str:
                session_start.append(item['time'])
            elif 'ends' in msg_str:
                session_end.append(item['time'])
            elif 'End' in msg_str:
                pupil_end.append(item['time'])

        # Creates a dataframe to store the extracted trigger information for each task session 
        pupil_trig_ses = pd.DataFrame(outcome_data)
        pupil_trig_ses['time_stamp'] = pd.to_numeric(pupil_trig_ses['time_stamp'], errors='coerce').astype('Int64')
        pupil_trig_ses['task_session'] = session

        # Deal with or raise error due to missing triggers
        if not outcome_data:
            print(f"ERROR: Outcome triggers missing. May need to remove this task session.")
        if not session_start:
            print(f"ERROR: Missing trigger and list is empty: session_start = {session_start}")
            if len(outcome_data) == 75:
                print(f"Adding timestamp of first outcome trigger in this task session")
                session_start.append(outcome_data[0]['time_stamp'])
            else:
                print("Missing outcome triggers also found. Consider removing this task session")
        if not session_end:
            print(f"ERROR: Missing trigger and list is empty: session_end = {session_end}")
            if len(outcome_data) == 75:
                print(f"Adding timestamp of last outcome trigger in this task session")
                session_end.append(outcome_data[-1]['time_stamp'])
            else:
                print("Missing outcome triggers also found. Consider removing this task session")
        if not pupil_end:
            print(f"ERROR: Missing trigger and list is empty: pupil_end = {pupil_end}")
            if len(outcome_data) == 75:
                print(f"Adding timestamp of last outcome trigger in this task session")
                pupil_end.append(outcome_data[-1]['time_stamp'])
            else:
                print("Missing outcome triggers also found. Consider removing this task session")


        # Creates a dataframe to store the extracted session characteristics information for each task session 
        pupil_ses = pd.DataFrame({
            'task_session': session, 
            'start': session_start,
            'end': session_end, 
            'nan_ratio': nan_ratio
        })

        # Creates a dataframe to store the extracted blink information for each task session 
        blink_ses = pd.DataFrame({
            'task_session': session,
            'blink_start': blink_start,
            'blink_end': blink_end
        })
        blink_ses['blink_dur'] = (blink_ses['blink_end'] - blink_ses['blink_start']) / 1000000

        # Check that data files are not clearly erroneous 
        data_tob, pupil_data, blink_ses = pupil_data_file_check(data_tob, pupil_trig_ses=pupil_trig_ses, blink_ses=blink_ses, pupil_ses=pupil_ses, pupil_end=pupil_end, file_path=file_path, file_size_threshold=2, session=session, plot_original_data=False, plot_reduced_data=False)

        # Append the data to the empty list to store the data for that task session
        pupil_data_ses_list.append(pupil_data)
        pupil_trig_ses_list.append(pupil_trig_ses)
        pupil_ses_list.append(pupil_ses)
        blink_ses_list.append(blink_ses)

        # Preprocess the data. If needed, can plot the preprocessed pupil time series. To see the blink ratio use data_tob.epoch_blink_ratio. For more complete summary can insert data_tob.epochs_info (provides a data frame with the epoch number, the blink ratio, NaN ratio per epoch and conditions [rejected etc])
        time, diameter, diameter_int, x_pos, x_pos_int, y_pos, y_pos_int = data_tob.preprocess(lpf=lpf, plot=False, remove_outlier=True, outlier_limit_std=outlier_limit_std, outlier_correct_average_amount=outlier_correct_average_amount)   

        processed_data = data_tob.data.copy()
        processed_data['task_session'] = session  
        pre_process_data_list.append(processed_data)

        if dict_onsets is not None:
            data_tob.epoch_tobii(dict_onsets[session], variable='diameter_int', decimate=1, reject=False, blink_ratio_thd=blink_ratio_threshold, before=window_start_before, after=window_end_after, baseline=baseline_cor, conditions={'event_num': dict_num[session]}, z_score_data=z_score)

            # Check to ensure the correct number of epochs
            print("For participant ", participant_str, " in session", session, "there are:", len(data_tob.epochs), "epochs")
            print(data_tob.epochs_info)

            # Appending dataframes containing epoched data to list for each task session
            epoched_data = pd.DataFrame([epoch for epoch in data_tob.epochs])
            epoched_data_list.append(epoched_data)

            # Append list to blink_ratio list 
            blink_ratio_epoch_list.append(data_tob.epoch_blink_ratio)

    # Concatenate pre processed pupil dataframes (trigger information, blink information, task session timings, epoched data, etc) across task sessions
    preprocessed_pupil_df = pd.concat(pre_process_data_list, ignore_index=True)

    pupil_trig_df = pd.concat(pupil_trig_ses_list, ignore_index=True)
    pupil_trig_df['outcome_num'] = pupil_trig_df.groupby('task_session').cumcount() 
    pupil_trig_df['cum_outcome_num'] = range(len(pupil_trig_df))
    pupil_trig_df['time_stamp'] = pupil_trig_df['time_stamp'].astype('int64')
 
    pupil_ses_df = pd.concat(pupil_ses_list, ignore_index=True)
    pupil_ses_df['duration'] = (pupil_ses_df['end'].astype(int) - pupil_ses_df['start'].astype(int)) / 1000000
    if select_eye:
        pupil_ses_df['used_eye'] = 'right'
    else:
        pupil_ses_df['used_eye'] = used_eye_list

    blink_df = pd.concat(blink_ses_list, ignore_index=True)
    blink_df['blink_count'] = blink_df.groupby('task_session').cumcount() 
    blink_df['cum_blink_count'] = range(len(blink_df))

    if hasattr(data_tob, 'epochs_info'):
        # Generate dataframe with blink_ratio for each epoch vertically stacked
        flattened_array = np.concatenate(blink_ratio_epoch_list)

        blink_ratio_epoch_df = pd.DataFrame(flattened_array, columns=['blink_ratio'])
        blink_ratio_epoch_df['above_threshold'] = blink_ratio_epoch_df['blink_ratio'] >= blink_ratio_threshold

        # Checking that  all epoch df have the same shape - possibly not due to sampling issues related to trigger insertion.
        expected_num_columns = round(((window_start_before + window_end_after) * 60) + 1) # 60Hz is the sampling frequency of the tobii eyetracker. Add 1, because the epoching window includes the last time point exactly.

        mismatched_shape_info = [(i, df.shape[1]) for i, df in enumerate(epoched_data_list) if df.shape[1] != expected_num_columns]

        if mismatched_shape_info:
            print("The following task session(s) have a different number of columns than expected (expected: {}):".format(expected_num_columns))
            for idx, shape in mismatched_shape_info:
                print(f"  - Task session index {idx} has {shape} columns")
        else:
            print("All task sessions have the expected number of columns.") 

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

            # For each column with NaN values, check if the NaN is in the last row. If it is then pad the null value with the data in the previous rpow for that column 
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

            print(f"Removed {len(epochs_to_remove)} epochs due to excessive blinking ({num_excess_blink_columns}).")
            print(f"Columns removed: {epochs_to_remove}")
        else:
            epochs_to_remove = None

        numeric_data = epoched_pupil_df.iloc[:, :]

        epoched_pupil_df['mean'] = numeric_data.mean(axis=1)
        epoched_pupil_df['std'] = numeric_data.std(axis=1)
        epoched_pupil_df['sem'] = epoched_pupil_df['std'] / np.sqrt(numeric_data.shape[1])
        epoched_pupil_df['time'] = data_tob.epochs_times

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
                        title_text=f'Preprocessed pupil diameter and x/y position across task sessions for participant {participant_str} in condition {condition_str}',
                        xaxis_title='Time (s)',
                        yaxis_title='Pupil Diameter (a.u)')
        

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-interpolation-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()

    return preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove



def align_pupil_to_events(pupil_trig_df, event_df, to_session, print_event_df_shape=True):
    """
    Function to align pupil data with adalearn outcome presentations marking some event of interest. Aligns outcome triggers between different files based on the cumulative number (id) of that outcome, or based on a relative number within task sessions. Provides initial checks to ensure that the data is matched and determines which route to progress with during the alignment.

    Parameters:
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files. 
    - event_df: pd.DataFrame, contains the outcome trigger information concerning some parameter of interest. Must include variables: task_session (task session identifier), outcome_num (relative number of the outcome of interest within a particular task session: 0-74), cum_outcome_num (cumulative count of the outcome number across all task sessions). 
    - to_session: int, defining the number of the last task session within the condition.
    - print_event_df_shape: boolean, if true will print descriptive stats of the event_df so one can ensure that the function is working correctly 

    Returns:
    - pupil_by_event: pd.DataFrame, merged dataframe defined by the outcome identifiers within event dataframe. Importantly extracts the timestamp of that outcome from the pupil data which is used in the process of the pupil response.   
    """
    if print_event_df_shape:
        print(f'The dataframe has shape: {event_df.shape}')
    
    # Convert task session columns in both dataframes to zero-padded strings
    pupil_trig_df['task_session'] = pupil_trig_df['task_session'].astype(str).str.zfill(2)
    event_df['task_session'] = event_df['task_session'].astype(str).str.zfill(2)

    # Step 1: Ensure that the same number of unique task session values exist in both the pupil and event dataframes.
    pupil_sessions = set(pupil_trig_df['task_session'].unique())
    event_sessions = set(event_df['task_session'].unique())

    if pupil_sessions != event_sessions:
        print(f"ERROR MISMATCH: Pupil data has {len(pupil_sessions)} unique task sessions ({pupil_sessions}), while event data has {len(event_sessions)} unique task sessions ({event_sessions}).")
    else:
        print("MATCH: Task sessions match across both dataframes")

    # Step 2: Check that the length of the pupil trigger data equals 75 * the number of unique task sessions.
    expected_length = 75 * to_session

    if len(pupil_trig_df) != expected_length:
        print(f"ERROR MISMATCH: Pupil data has {len(pupil_trig_df)} rows, but there should be {expected_length} expected rows.")
    else:
        print("MATCH: Pupil data is as expected given the number of task sessions performed.")

    # Step 3: If any of the checks above are violated, proceed with finding the matching rows based on relative task session trigger number.
    if pupil_sessions != event_sessions or len(pupil_trig_df) != expected_length:
        pupil_grouped = pupil_trig_df.groupby('task_session').apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)
        behav_event_grouped = event_df.groupby('task_session').apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)

        pupil_by_event = pd.merge(
            pupil_grouped,
            behav_event_grouped,
            left_on=['task_session', 'outcome_num'],
            right_on=['task_session', 'outcome_num'],
            how='inner'
        )

        print(f"Merged dataframe based on relative triggers. The number of relative triggers found was: {pupil_by_event.shape[0]}.")

    # If checks were not violated, match pupil to event data based on cumulative outcome number across all task sessions.
    else:
        pupil_by_event = pupil_trig_df[pupil_trig_df['cum_outcome_num'].isin(event_df['cum_outcome_num'])]

        print(f"Merged dataframe based on cumulative triggers. The number of triggers found was: {pupil_by_event.shape[0]}.")

    # Ensure the 'outcome' column exists in the resulting DataFrame
    if 'outcome' not in pupil_by_event.columns:
        # Align 'outcome' based on task_session and outcome_num after the merge
        pupil_by_event = pd.merge(
            pupil_by_event,
            event_df[['task_session', 'outcome_num', 'outcome']],
            on=['task_session', 'outcome_num'],
            how='left',
            suffixes=('', '_event') 
        )
    
    # Fill NaN values in 'outcome' column if needed
    pupil_by_event['outcome'].fillna('No Outcome', inplace=True)

    # Remove duplicate columns that may arise from the merge operation
    duplicate_cols = [col for col in pupil_by_event.columns if col.endswith('_x') or col.endswith('_y')]
    if duplicate_cols:
        pupil_by_event = pupil_by_event.drop(columns=duplicate_cols)

    return pupil_by_event


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


def all_outcome_pupil_epochs(directory_path, participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, baseline_cor = [-0.25, 0], variable_df=None, variable_column=None, select_eye = True, remove_epochs = True, plot_averaged_pupil_response = False, plot_raster_pupil_response = False):
    """
    Function to preprocess and analysis pupil data for each outcome across all task sessions. Uses load_and_process_pupil_data method to load and run the preporocessing analysis. There is an option to plot the mean response from all outcomes across all task sessions, as well as a raster plot from each outcome. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
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
    print(f"Condition {condition_str}:")
    for session, events in outcome_nums.items():
      print(f"In task session {session} there are {len(events)} outcome observations") 

    # Analysing pupil data based on identified outcome indices
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_onsets, dict_num=outcome_nums, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.4, baseline_cor=baseline_cor, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

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

        title = f'Epoched pupil diameter heatmap across all outcome observations for participant {participant_str} in condition {condition_str} (ordered by {variable_column})'

    else:
        heatmap_df = outcome_epoched_pupil_df.copy()

        pupil_epoch_matrix = [heatmap_df.iloc[:, i].values for i in range(heatmap_df.shape[1] - 4)]
        print(f"Number of epochs added to heatmap: {len(pupil_epoch_matrix)}")

        title = f'Epoched pupil diameter heatmap across all outcome observations for participant {participant_str} in condition {condition_str}'


    # Option to plot the average pupil response across all outcomes
    if plot_averaged_pupil_response:
        if len(baseline_cor) == 0:
            title = f"Baseline uncorrected pupil response to each observations for participant {participant_str} in condition {condition_str} (Mean +/- SEM)"
        else:
            title = f"Baseline corrected pupil response to each observations for participant {participant_str} in condition {condition_str} (Mean +/- SEM)"
              
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
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-all-outcomes.png"

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
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-all-outcomes-heatmap.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    return outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df



def outcome_colour_pupil_epochs(directory_path, participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, baseline_cor = [-0.25, 0], select_eye = True, remove_epochs = True, plot_pupil_by_colour_response = False):
    """
    Function to preprocess and analysis pupil data based on the specific colour of the outcome. Uses load_and_process_pupil_data method to run the preporocessing analysis. Option to plot the mean and sem derived from all instances of 'blue' and 'yellow' outcomes across all trials. Parameters defining the preprocessing of the pupil data should be modified in the load_and_process_pupil_data function. 

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - baseline_cor: list, contains the start and end time for the baseline correction. If left empty (e.g. []), then no baseline correction will be applied (default = [-0.2, 0]).
    - select_eye: boolean, if true, will use a predfined eye (default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (default = True).
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - plot_pupil_by_colour: boolean, option to plot the mean and sem of the epoched pupil data around the different coloured outcomes using PLotly. (Default = False).

    Returns:
    - outcome_1_onsets: dict, contains timestamps for outcomes = 1. Keys are task session and values are time stamps within that task session. 
    - outcome_1_nums: dict, contains a cumulative count of the number of outcomes = 1 within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - outcome_1_epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean std, sem and time stamps for each epochs defined by an outcome = 1.
    - outcome_1_blink_ratio_cp_df: pd.DataFrame, contains the blink ratio information for each epoch around outcomes = 1.
    - outcome_0_onsets: dict, contains timestamps for outcomes = 0. Keys are task session and values are time stamps within that task session. 
    - outcome_0_nums: dict, contains a cumulative count of the number of outcomes = 0 within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - outcome_0_epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean std, sem and time stamps for each epochs defined by an outcome = 0.
    - outcome_0_blink_ratio_cp_df: pd.DataFrame, contains the blink ratio information for each epoch around outcomes = 0.
    """
    # Check if pupil_trig_df is in the correct format
    pupil_trig_df['task_session'] = pupil_trig_df['task_session'].apply(lambda x: str(x) if not isinstance(x, str) else x).str.zfill(2)

    # Splitting the pupil trig dataframe into two based on outcomes 
    pupil_trig_1_df = pupil_trig_df[pupil_trig_df['outcome'] == 1]
    pupil_trig_0_df = pupil_trig_df[pupil_trig_df['outcome'] == 0]

    # Creating dictionaries
    outcome_1_onsets, outcome_1_nums = create_pupil_epoch_dicts(pupil_trig_1_df)
    outcome_0_onsets, outcome_0_nums = create_pupil_epoch_dicts(pupil_trig_0_df)

    # Checking the number of 'blue' outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in outcome_1_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes = 1") 

    # Checking the number of 'yellow' outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in outcome_0_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes = 0") 

    # Analysing pupil data based on outcome values indices
    preprocessed_pupil_outcome_1_df, pupil_trig_outcome_1_df, pupil_ses_outcome_1_df, blink_outcome_1_df, outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df, outcome_1_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_1_onsets, dict_num=outcome_1_nums, session_list=None, baseline_cor=baseline_cor, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

    preprocessed_pupil_outcome_0_df, pupil_trig_outcome_0_df, pupil_ses_outcome_0_df, blink_outcome_0_df, outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df, outcome_0_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_0_onsets, dict_num=outcome_0_nums, session_list=None, baseline_cor=baseline_cor, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)


    print(f'Number of epochs added for outcome 1: {len(range(outcome_1_epoched_pupil_df.shape[1] - 4))}')
    print(f'Number of epochs added for outcome 0: {len(range(outcome_0_epoched_pupil_df.shape[1] - 4))}')

    outcome_pupil_dict = {
        'outcome_1_onsets': outcome_1_onsets,
        'outcome_1_nums': outcome_1_nums,
        'outcome_1_epoched_pupil_df': outcome_1_epoched_pupil_df,
        'outcome_1_blink_ratio_df': outcome_1_blink_ratio_df,
        'outcome_1_epochs_to_remove': outcome_1_epochs_to_remove,
        'outcome_0_onsets': outcome_0_onsets,
        'outcome_0_nums': outcome_0_nums,
        'outcome_0_epoched_pupil_df': outcome_0_epoched_pupil_df,
        'outcome_0_blink_ratio_df': outcome_0_blink_ratio_df,
        'outcome_0_epochs_to_remove': outcome_0_epochs_to_remove
    }

    # Option to plot the two average pupil responses to different outcome colours
    if plot_pupil_by_colour_response:
              
        fig_pupil_avg = go.Figure()

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_1_epoched_pupil_df['time'],
            y=outcome_1_epoched_pupil_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='Outcome = 1',
            showlegend=True
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_1_epoched_pupil_df['time'],
            y=outcome_1_epoched_pupil_df['mean'] + outcome_1_epoched_pupil_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_1_epoched_pupil_df['time'],
            y=outcome_1_epoched_pupil_df['mean'] - outcome_1_epoched_pupil_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_0_epoched_pupil_df['time'],
            y=outcome_0_epoched_pupil_df['mean'],
            mode='lines',
            line=dict(color='rgba(234, 239, 44, 1)'),
            name='Outcome = 0',
            showlegend=True
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_0_epoched_pupil_df['time'],
            y=outcome_0_epoched_pupil_df['mean'] + outcome_0_epoched_pupil_df['sem'],
            mode='lines',
            line=dict(color='rgba(234, 239, 44, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_trace(go.Scatter(
            x=outcome_0_epoched_pupil_df['time'],
            y=outcome_0_epoched_pupil_df['mean'] - outcome_0_epoched_pupil_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(234, 239, 44, 0.2)',
            line=dict(color='rgba(234, 239, 44, 0.1)'),
            showlegend=False
        ))

        fig_pupil_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)


        fig_pupil_avg.update_layout(
            title=f'Baseline corrected pupil response to each outcome colour presentation for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)',
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-by-colour.png"

        file_path = os.path.join(plot_dir, file_name)

        fig_pupil_avg.write_image(file_path)
        

        fig_pupil_avg.show()


    return outcome_pupil_dict



def change_point_pupil_epochs(directory_path, participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, change_point_indices, baseline_cor = [-0.25, 0], select_eye = True, remove_epochs = True,  plot_pupil_by_cp=False):
    """
    Function to preprocess and analysis pupil data around change points in the hidden task structure. Uses load_and_process_pupil_data method to run the preporocessing analysis. In addition to analysising the data around change points, there is an option to plot the mean and sem derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
    - baseline_cor: list, contains the start and end time for the baseline correction. If left empty (e.g. []), then no baseline correction will be applied (default = [-0.2, 0]).
    - select_eye: boolean, if true, will use a predfined eye (default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (default = True).
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - plot_pupil_by_cp: boolean, option to plot the mean and standard deviation of the epoched pupil data around the change points using PLotly. (Default = False).

    Returns:
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across change point defined epochs and the timestamp.
    - blink_ratio_cp_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around change points
    """
    # Ensure that change points and pupil triggers and aligned correctly
    pupil_by_change_points = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=change_point_indices, to_session=to_session, print_event_df_shape=True)

    # Getting timestamps of all non change point triggers to plot as a comparison
    non_change_point_triggers = pupil_trig_df.merge(change_point_indices[['outcome_num', 'task_session']], on=['outcome_num', 'task_session'],how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')

    # Creating dictionaries for storage of change point time and number 
    change_point_onsets, change_point_nums = create_pupil_epoch_dicts(pupil_by_change_points)
    non_cp_onsets, non_cp_nums = create_pupil_epoch_dicts(non_change_point_triggers)

    # Checking the number of change points identified
    print(f"Condition {condition_str}:")
    for session, events in change_point_nums.items():
      print(f"In task session {session} there are {len(events)} identified change points") 


    # Analysing pupil data based on identified change point indices
    preprocessed_pupil_cp_df, pupil_trig_cp_df, pupil_ses_cp_df, blink_cp_df, epoched_pupil_cp_df, blink_ratio_cp_df, cp_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=change_point_onsets, dict_num=change_point_nums, baseline_cor=baseline_cor, lpf=[5], select_eye=select_eye, remove_epochs=remove_epochs)

    preprocessed_pupil_non_cp_df, pupil_trig_non_cp_df, pupil_ses_non_cp_df, blink_non_cp_df, epoched_pupil_non_cp_df, blink_ratio_non_cp_df, non_cp_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=non_cp_onsets, dict_num=non_cp_nums,baseline_cor=baseline_cor, lpf=[5],select_eye=select_eye, remove_epochs=remove_epochs)


    change_point_pupil_dict = {
        'change_point_onsets': change_point_onsets,
        'change_point_nums': change_point_nums,
        'epoched_pupil_cp_df': epoched_pupil_cp_df,
        'blink_ratio_cp_df': blink_ratio_cp_df,
        'cp_epochs_to_remove': cp_epochs_to_remove
    }

    print(f'Number of epochs added for change points: {len(range(epoched_pupil_cp_df.shape[1] - 4))}')
    print(f'Number of epochs added for non change points: {len(range(epoched_pupil_non_cp_df.shape[1] - 4))}')

    # Option to plot the average pupil response timelocked to change points
    if plot_pupil_by_cp:
        
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='Change point',
            showlegend=True 
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'] + epoched_pupil_cp_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'] - epoched_pupil_cp_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_non_cp_df['time'],
            y=epoched_pupil_non_cp_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 255, 0, 1)'),
            name='Non change point',
            showlegend=True 
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_non_cp_df['time'],
            y=epoched_pupil_non_cp_df['mean'] + epoched_pupil_non_cp_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 255, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_non_cp_df['time'],
            y=epoched_pupil_non_cp_df['mean'] - epoched_pupil_non_cp_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(0, 255, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title=f'Baseline corrected right pupil response epoched around change points for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)',
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-by-change-point.png"

        file_path = os.path.join(plot_dir, file_name)

        fig_epoch_avg.write_image(file_path)
            

        fig_epoch_avg.show()

    return change_point_pupil_dict



def comp_var_pupil_quartile_epochs(directory_path, participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, computational_var_df, baseline_cor = [-0.25, 0], variable_cat = None, select_eye = True, remove_epochs = True, plot_pupil_by_quartile=True):
    """"
     Function to preprocess and then compute epoched pupil diameter response around separate quartiles of a computational variable of interest. Preprocessing taken from load_and_process_pupil_data method. Calculates the mean, std and sem  across each identified epoch for all task sessions. In addition to analysising the data around the computational variable, there is an option to plot the mean and sem derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function.

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - computational_var_df: pd.DataFrame, contains computational variable of interest (e.g. surprise, prior confidence, change point probability) and the categorised values of this variable. Used to define timestamps for the categorised values (which are contained in a dictionary, with keys being task sessions and values are time stamps within that task session) and a cumulative count of the number of surprise values categorised as either high or low (again, a dictionary with keys being task sessions and values being the cumulative count within that task session).
    - baseline_cor: list, contains the start and end time for the baseline correction. If left empty (e.g. []), then no baseline correction will be applied (default = [-0.2, 0]).
    - variable_cat: string, column header of the categorized computational variable of interest. 
    - select_eye: boolean, if true, will use a predfined eye (default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (default = True)
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - plot_pupil_by_quartile: boolean, option to plot the mean and standard deviation across epochs using Plotly. (Default = True). 

    Returns:
    - first_quartile_onsets: dict, contains the timetamps of outcomes within the first quartile for the computational variable of interest. Keys are task session and values are timestamps within that task session. 
    - first_quartile_nums: dict, contains a cumulative count of the number of observations within the first quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of observations within that task session.
    - epoched_pupil_first_quartile_df: pd.DataFrame, containing transposed epoched data for the first quartile (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - first_quartile_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the first quartile.
    - first_quartile_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_first_quartile_df that have been removed due to an excessive blink ratio.
    - second_quartile_onsets: dict, contains the timetamps of outcomes within the second quartile of the computational variable of interest. Keys are task session and values are timestamps within that task session. 
    - second_quart_nums: dict, contains a cumulative count of the number of observations within the second quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of observations within that task session.
    - epoched_pupil_second_quartile_df: pd.DataFrame, containing transposed epoched data for the second quartile (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - second_quartile_blink_ratio_df: pd.DataFrame, contains the blink ratio information for each generated epoch around the second quartile.
    - second_quartile_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_second_quartile_df that have been removed due to an excessive blink ratio.
    - third_quartile_onsets: dict, contains the timetamps of outcomes within the third quartile of the computational variable. Keys are task session and values are timestamps within that task session. 
    - third_quartile_nums: dict, contains a cumulative count of the number of observations within the third quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of observations within that task session.
    - epoched_pupil_third_quartile_df: pd.DataFrame, containing transposed epoched data for the third quartile (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - third_quartile_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the third quartile.
    - third_quartile_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_third_quartile_df that have been removed due to an excessive blink ratio.
    - fourth_quartile_onsets: dict, contains the timetamps of outcomes within the fourth quartile of the computational variable. Keys are task session and values are timestamps within that task session. 
    - surprise_fourth_quartile_nums: dict, contains a cumulative count of the number of observations within the fourth quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of surprising observations within that task session.
    - epoched_pupil_fourth_quartile_df: pd.DataFrame, containing transposed epoched data for the fourth quartile (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - fourth_quartile_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the fourth quartile.
    - fourth_quartile_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_fourth_quartile_df that have been removed due to an excessive blink ratio.
    """
    # Filter across quartiles within the computational variable of interest
    filtered_first_quartile = computational_var_df[computational_var_df[variable_cat] == 'first_quartile'].copy()
    filtered_second_quartile = computational_var_df[computational_var_df[variable_cat] == 'second_quartile'].copy()
    filtered_third_quartile = computational_var_df[computational_var_df[variable_cat] == 'third_quartile'].copy()
    filtered_fourth_quartile = computational_var_df[computational_var_df[variable_cat] == 'fourth_quartile'].copy()

    pupil_by_first_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_first_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_second_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_second_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_third_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_third_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_fourth_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_fourth_quartile, to_session=to_session, print_event_df_shape=True)
   
    # Creating dictionaries
    first_quartile_onsets, first_quartile_nums = create_pupil_epoch_dicts(pupil_by_first_quart)
    second_quartile_onsets, second_quartile_nums = create_pupil_epoch_dicts(pupil_by_second_quart)
    third_quartile_onsets, third_quartile_nums = create_pupil_epoch_dicts(pupil_by_third_quart)
    fourth_quartile_onsets, fourth_quartile_nums = create_pupil_epoch_dicts(pupil_by_fourth_quart)

    # Checking the number of outcomes in each quartile 
    print(f"Condition {condition_str}:")
    for session, events in first_quartile_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes in the first quartile") 

    print(f"Condition {condition_str}:")
    for session, events in second_quartile_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes in the second quartile") 

    print(f"Condition {condition_str}:")
    for session, events in third_quartile_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes in the third quartile") 

    print(f"Condition {condition_str}:")
    for session, events in fourth_quartile_nums.items():
      print(f"In task session {session} there are {len(events)} outcomes in the fourth quartile") 

    # Generating session lists for events within each quartile
    first_quart_ses_list = list(first_quartile_onsets.keys())
    second_quart_ses_list = list(second_quartile_onsets.keys())
    third_quart_ses_list = list(third_quartile_onsets.keys())
    fourth_quart_ses_list = list(fourth_quartile_onsets.keys())

    # Generating pupil response around observations for the variable of interest 
    print("Analysing first quartile. Processing...") 
    preprocessed_pupil_first_quart_df, pupil_trig_first_quart_df, pupil_ses_first_quart_df, blink_first_quart_df, epoched_pupil_first_quart_df, first_quart_blink_ratio_df, first_quart_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=first_quartile_onsets, dict_num=first_quartile_nums, baseline_cor=baseline_cor, lpf=[5],session_list=first_quart_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysing second quartile. Processing...") 
    preprocessed_pupil_second_quart_df, pupil_trig_second_quart_df, pupil_ses_second_quart_df, blink_second_quart_df, epoched_pupil_second_quart_df, second_quart_blink_ratio_df, second_quart_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=second_quartile_onsets, dict_num=second_quartile_nums, baseline_cor=baseline_cor, lpf=[5],session_list=second_quart_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysisng third quartile. Processing...") 
    preprocessed_pupil_third_quart_df, pupil_trig_third_quart_df, pupil_ses_third_quart_df, blink_third_quart_df, epoched_pupil_third_quart_df, third_quart_blink_ratio_df, third_quart_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=third_quartile_onsets, dict_num=third_quartile_nums, baseline_cor=baseline_cor, lpf=[5],session_list=third_quart_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysing fourth quartile. Processing...") 
    preprocessed_pupil_fourth_quart_df, pupil_trig_fourth_quart_df, pupil_ses_fourth_quart_df, blink_fourth_quart_df, epoched_pupil_fourth_quart_df, fourth_quart_blink_ratio_df, fourth_quart_epochs_to_remove = load_and_process_pupil_data(directory_path=directory_path, participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=fourth_quartile_onsets, dict_num=fourth_quartile_nums, baseline_cor=baseline_cor, lpf=[5],session_list=fourth_quart_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)


    print(f'Number of epochs added for the first quartile: {len(range(epoched_pupil_first_quart_df.shape[1] - 4))}')
    print(f'Number of epochs added for the second quartile: {len(range(epoched_pupil_second_quart_df.shape[1] - 4))}')
    print(f'Number of epochs added for the third quartile: {len(range(epoched_pupil_third_quart_df.shape[1] - 4))}')
    print(f'Number of epochs added for the fourth quartile: {len(range(epoched_pupil_fourth_quart_df.shape[1] - 4))}')


    com_var_pupil_quartile_dict = {
        'first_quartile_onsets': first_quartile_onsets,
        'first_quartile_nums': first_quartile_nums,
        'epoched_pupil_first_quart_df': epoched_pupil_first_quart_df,
        'first_quart_blink_ratio': first_quart_blink_ratio_df,
        'first_quart_epochs_to_remove': first_quart_epochs_to_remove,
        'second_quartile_onsets': second_quartile_onsets,
        'second_quartile_nums':  second_quartile_nums,
        'epoched_pupil_second_quart_df': epoched_pupil_second_quart_df,
        'second_quart_blink_ratio_df': second_quart_blink_ratio_df,
        'second_quart_epochs_to_remove': second_quart_epochs_to_remove,
        'third_quartile_onsets': third_quartile_onsets,
        'third_quartile_nums': third_quartile_nums,
        'epoched_pupil_third_quart_df': epoched_pupil_third_quart_df,
        'third_quart_blink_ratio': third_quart_blink_ratio_df, 
        'third_quart_epochs_to_remove':third_quart_epochs_to_remove,
        'fourth_quartile_onsets': fourth_quartile_onsets,
        'fourth_quartile_nums': fourth_quartile_nums,
        'epoched_pupil_fourth_quart_df': epoched_pupil_fourth_quart_df,
        'fourth_quart_blink_ratio': fourth_quart_blink_ratio_df,
        'fourth_quart_epochs_to_remove':fourth_quart_epochs_to_remove
        }
    
    # Additional title elements
    if "surprise_cat" in variable_cat:
        title_addon = "surprise"
    elif "prior_conf_cat" in variable_cat:
        title_addon = "prior-conf"
    elif "cpp_cat" in variable_cat:
        title_addon = "cpp"
    else:
        title_addon = ""

    # Option to plot pupil response of the pupil diameter timelocked to quartiles of computational variable of interest
    if plot_pupil_by_quartile:
              
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_quart_df['time'],
            y=epoched_pupil_first_quart_df['mean'],
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 1)'),
            name='First Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_quart_df['time'],
            y=epoched_pupil_first_quart_df['mean'] + epoched_pupil_first_quart_df['sem'],
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_quart_df['time'],
            y=epoched_pupil_first_quart_df['mean'] - epoched_pupil_first_quart_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 255, 0, 0.2)',
            line=dict(color='rgba(255, 255, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_quart_df['time'],
            y=epoched_pupil_second_quart_df['mean'],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 1)'),
            name='Second Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_quart_df['time'],
            y=epoched_pupil_second_quart_df['mean'] + epoched_pupil_second_quart_df['sem'],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_quart_df['time'],
            y=epoched_pupil_second_quart_df['mean'] - epoched_pupil_second_quart_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255, 165, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_quart_df['time'],
            y=epoched_pupil_third_quart_df['mean'],
            mode='lines',
            line=dict(color='rgba(255, 102, 102, 1)'),
            name='Third Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_quart_df['time'],
            y=epoched_pupil_third_quart_df['mean'] + epoched_pupil_third_quart_df['sem'],
            mode='lines',
            line=dict(color='rgba(255, 102, 102, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_quart_df['time'],
            y=epoched_pupil_third_quart_df['mean'] - epoched_pupil_third_quart_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 102, 102, 0.2)',
            line=dict(color='rgba(255, 102, 102, 0.1)'),
            showlegend=False
        ))


        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_quart_df['time'],
            y=epoched_pupil_fourth_quart_df['mean'],
            mode='lines',
            line=dict(color='rgba(139, 0, 0, 1)'),
            name='Fourth Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_quart_df['time'],
            y=epoched_pupil_fourth_quart_df['mean'] + epoched_pupil_fourth_quart_df['sem'],
            mode='lines',
            line=dict(color='rgba(139, 0, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_quart_df['time'],
            y=epoched_pupil_fourth_quart_df['mean'] - epoched_pupil_fourth_quart_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(139, 0, 0, 0.2)',
            line=dict(color='rgba(139, 0, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title=f'Baseline corrected right pupil response epoched around outcomes relating to {title_addon}, split by quartile, for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)',
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-by-{title_addon}-quartile.png"

        file_path = os.path.join(plot_dir, file_name)

        fig_epoch_avg.write_image(file_path)
       

        fig_epoch_avg.show()

    return com_var_pupil_quartile_dict
