import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from preprocess_eyedata import pupil



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
    - file_size_threshold: integer, defines the size of file above which would be cause for concern, which should be raised and investigation into the file contents is needed (default = 2 mb).
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



def load_and_process_pupil_data(participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, dict_onsets, dict_num, session_list = None, outlier_limit_std = 10, outlier_correct_average_amount = 15,  window_start_before = 0.2, window_end_after = 3, lpf = [5], blink_ratio_threshold = 0.20, select_eye = True, remove_epochs = True, z_score = True, plot_interpolation_process = False):
    """
    Function to load and preprocess pupil data. Preprocessing involves defining blinks, outliers and the interpolation of pupil data. Preprocessing taken from preprocess_eyedata script, created by Florent, Maëva, Aziza and Yannaël. Option to plot the raw and interpolated data. If dictionaries passed defining events of interest, will compute epochs around defined outcomes of interest (e.g. change points, high and low surprise, etc). If epochs computed, will calculate the mean, std, sem across each identified epoch for all task sessions.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - dict_onsets: dict, contains timestamps of outcomes of interest (e.g. change points, high and low surprise, etc). Keys are task session and values are time stamps within that task session. (Default = None)
    - dict_nums: dict, contains a cumulative count of the number of outcomes of interest identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session. (Default = None)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session. (Default = None)
    - outlier_limit_std: int, defines the threshold of the standard deviation of the derivative when preprocessing the pupil data. (Default = 10)
    - outlier_correct_average_amount: int, the number of data points to the left and at the right of the artifact which will replace the value of the artifact by the value of the mean of the data around it. (Default = 15).
    - window_start_before: int, value defining the start of the epoch window in time (seconds). (Default = 1)
    - window_end_after: int, value defining the end of the epoch window in time (seconds). (Default = 3)
    - lpf: list, cutoff (Hz) to low pass filter the data with. If [], do nothing. (Default = 5 Hz)
    - blink_ratio_threshold:, integer, value determining the threshold criteria for the removal of an epoch due to excessive blink ratio. (Default = 0.2)
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
    for session in sessions_to_process:
        file_dir = f"/home/aaron/Documents/Python/Ada_Learn/VNS_Ada_Learn/Data/sub-{participant_str}/ses-{condition_str}/eye"
        file_name = f"sub-{participant_str}_ses-{condition_str}_task-AdalearnVNS_run-{session}_eyetracker.txt"
        file_path = f"{file_dir}/{file_name}"

        # Defining the eye to be used in the pupil analysis
        if select_eye:
            print(f"Eye pre-selected for Session {session}. The chosen eye is: right")
            used_eye_list.append(select_eye)

            data_tob.import_data(file_name, events_warner='MSG', directory = file_dir, eye='right', col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='tobii')
     
        else:
            best_eye = identifying_best_eye(data_tob=data_tob, file_name=file_name, file_dir=file_dir, session=session)

            used_eye_list.append(best_eye)

            data_tob.import_data(file_name, events_warner='MSG', directory = file_dir, eye=best_eye, col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='tobii')

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

        # Preprocess the data. If need, can plot the preprocessed pupil time series. To see the blink ratio use data_tob.epoch_blink_ratio. For more complete summary can insert data_tob.epochs_info (provides a data frame with the epoch number, the blink ratio, NaN ratio per epoch and conditions [rejected etc])
        time, diameter, diameter_int, x_pos, x_pos_int, y_pos, y_pos_int = data_tob.preprocess(lpf=lpf, plot=False, remove_outlier=True, outlier_limit_std=outlier_limit_std, outlier_correct_average_amount=outlier_correct_average_amount)   

        processed_data = data_tob.data.copy()
        processed_data['task_session'] = session  
        pre_process_data_list.append(processed_data)

        if dict_onsets is not None:
            data_tob.epoch_tobii(dict_onsets[session], variable='diameter_int', decimate=1, reject=False, blink_ratio_thd=blink_ratio_threshold, before=window_start_before, after=window_end_after, baseline=[-0.2, 0], conditions={'event_num': dict_num[session]}, z_score_data=z_score)

            # Check to ensure the correct number of epochs
            print("For participant ", participant_str, " in session", session, "there are:", len(data_tob.epochs), "epochs")
            print(data_tob.epochs_info)

            # Appending dataframes containing epoched data to list for each task session
            epoched_data = pd.DataFrame([epoch for epoch in data_tob.epochs])
            epoched_data_list.append(epoched_data)

            # Append list to blink_ratio list 
            blink_ratio_epoch_list.append(data_tob.epoch_blink_ratio)

    # Concatenate pre_processed pupil dataframes across task sessions
    preprocessed_pupil_df = pd.concat(pre_process_data_list, ignore_index=True)

    pupil_trig_df = pd.concat(pupil_trig_ses_list, ignore_index=True)
    pupil_trig_df['outcome_num'] = pupil_trig_df.groupby('task_session').cumcount() 
    pupil_trig_df['cum_outcome_num'] = range(len(pupil_trig_df))
 
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

        # Collating the timelocked epoched pupil data to calculate descriptive statistics for plotting
        epoched_pupil_df = pd.concat(epoched_data_list, ignore_index=True).T

        # Removing epochs in which interpolation exceeds defined threshold
        if remove_epochs:
            # Get indices of epochs (columns) to remove
            epochs_to_remove = blink_ratio_epoch_df.index[blink_ratio_epoch_df['above_threshold']]
            epoched_pupil_df.drop(columns=epochs_to_remove, inplace=True)

            print(f"Removed {len(epochs_to_remove)} epochs due to excessive interpolation.")
            print(f"Columns removed: {list(epochs_to_remove)}")
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

        fig.update_layout(height=900*num_sessions, 
                        width=1200,
                        title_text=f'Preprocessed pupil diameter and x/y position across task sessions for participant {participant_str} in condition {condition_str}',
                        xaxis_title='Time (s)',
                        yaxis_title='Pupil Diameter (a.u)')

        fig.show()

    return preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove


def align_pupil_to_events(pupil_trig_df, event_df, to_session, print_event_df_shape=True):
    """
    Function to align pupil data with adalearn outcome presentations marking some event of interest. Aligns outcome triggers between different files based on the cumulative number (id) of that outcome, or based on a relative number within task sessions. Provides initial checks to ensure that the data is macthed and determine which route to progress with during the alignment.

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


def all_outcome_pupil_epochs(participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, variable_df=None, variable_column=None, select_eye = True, remove_epochs = True, plot_averaged_pupil_response = False, plot_raster_pupil_response = False):
    """
    Function to preprocess and analysis pupil data for each outcome across all task sessions. Uses load_and_process_pupil_data method to load and run the preporocessing analysis. There is an option to plot the mean response from all outcomes across all task sessions, as well as a raster plot from each outcome. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
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
    """
    # Creating dictionaries for storage of each outcome time and number 
    outcome_onsets, outcome_nums = create_pupil_epoch_dicts(pupil_trig_df)

    # Checking the number of outcomes
    print(f"Condition {condition_str}:")
    for session, events in outcome_nums.items():
      print(f"In task session {session} there are {len(events)} outcome observations") 

    # Analysing pupil data based on identified outcome indices
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_onsets, dict_num=outcome_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

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

        #for col in outcome_epoched_pupil_df.columns[:1499]:
        #    fig_pupil_avg.add_trace(go.Scatter(
        #    x=outcome_epoched_pupil_df['time'],
        #    y=outcome_epoched_pupil_df[col],  
        #    mode='lines',
        #    line=dict(color='rgba(0, 0, 255, 0.1)', dash='dot'),
        #    showlegend=False
        #))

        fig_pupil_avg.update_layout(
            title=f'Baseline corrected pupil response to each observations for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

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

        fig.update_layout(title=title)

        fig.show()

    return outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove


def outcome_colour_pupil_epochs(participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, select_eye = True, remove_epochs = True, plot_pupil_by_colour_response = False):
    """
    Function to preprocess and analysis pupil data based on the specific colour of the outcome. Uses load_and_process_pupil_data method to run the preporocessing analysis. Option to plot the mean and sem derived from all instances of 'blue' and 'yellow' outcomes across all trials. Parameters defining the preprocessing of the pupil data should be modified in the load_and_process_pupil_data function. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
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
    preprocessed_pupil_outcome_1_df, pupil_trig_outcome_1_df, pupil_ses_outcome_1_df, blink_outcome_1_df, outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df, outcome_1_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_1_onsets, dict_num=outcome_1_nums, session_list=None, window_start_before=0.2, window_end_after=1.2, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

    preprocessed_pupil_outcome_0_df, pupil_trig_outcome_0_df, pupil_ses_outcome_0_df, blink_outcome_0_df, outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df, outcome_0_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=outcome_0_onsets, dict_num=outcome_0_nums, session_list=None, window_start_before=0.2, window_end_after=1.2, lpf=[5], blink_ratio_threshold=0.2, select_eye=select_eye, remove_epochs=remove_epochs, z_score=True, plot_interpolation_process=False)

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
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_pupil_avg.show()


    return outcome_1_onsets, outcome_1_nums, outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df, outcome_1_epochs_to_remove, outcome_0_onsets, outcome_0_nums, outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df, outcome_0_epochs_to_remove


def change_point_pupil_epochs(participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, change_point_indices, select_eye = True, remove_epochs = True,  plot_pupil_by_cp=False):
    """
    Function to preprocess and analysis pupil data around change points in the hidden task structure. Uses load_and_process_pupil_data method to run the preporocessing analysis. In addition to analysising the data around change points, there is an option to plot the mean and sem derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
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
    preprocessed_pupil_cp_df, pupil_trig_cp_df, pupil_ses_cp_df, blink_cp_df, epoched_pupil_cp_df, blink_ratio_cp_df, cp_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=change_point_onsets, dict_num=change_point_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5], select_eye=select_eye, remove_epochs=remove_epochs)

    preprocessed_pupil_non_cp_df, pupil_trig_non_cp_df, pupil_ses_non_cp_df, blink_non_cp_df, epoched_pupil_non_cp_df, blink_ratio_non_cp_df, non_cp_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=non_cp_onsets, dict_num=non_cp_nums,window_start_before=0.2, window_end_after=1.2, lpf=[5],select_eye=select_eye, remove_epochs=remove_epochs)

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
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_epoch_avg.show()

    return change_point_onsets, change_point_nums, epoched_pupil_cp_df, blink_ratio_cp_df, cp_epochs_to_remove


def surprise_pupil_epochs(participant_str, list_subject_selected, condition_str, condition_task_sessions, from_session, to_session, pupil_trig_df, surprise_df, select_eye = True, remove_epochs = True, plot_pupil_by_surprise=True):
    """"
     Function to preprocess and then compute epoched pupil diameter response around separate quartiles of suprising observations. Preprocessing taken from load_and_process_pupil_data method. Calculates the mean, std and sem  across each identified epoch for all task session. In addition to analysising the data around surprise, there is an option to plot the mean and sem derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - surprise_df: pd.DataFrame, contains computed surprise values and the categorised values. Used to define timestamps for the categorised surprise values (which are contained in a dictionary, with keys being task sessions and values are time stamps within that task session) and a cumulative count of the number of surprise values categorised as either high or low (again, a dictionary with keys being task sessions and values being the cumulative count within that task session).
    - select_eye: boolean, if true, will use a predfined eye (default = right). If false will run a function to identify the best eye (one with the least nan values) and collect data using that eye (default = True)
    - remove_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed. (Default = True)
    - plot_pupil_by_surprise: boolean, option to plot the mean and standard deviation across epochs using Plotly. (Default = True). 

    Returns:
    - surprise_first_quart_onsets: dict, contains the timetamps of outcomes within the first quartile of surprise. Keys are task session and values are timestamps within that task session. 
    - surprise_first_quart_nums: dict, contains a cumulative count of the number of surprising observations within the first quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of surprising observations within that task session.
    - epoched_pupil_first_surprise_df: pd.DataFrame, containing transposed epoched data for the first quartile of surprise (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - first_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the first quartile of surprise.
    - first_surprise_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_first_surprise_df that have been removed due to an excessive blink ratio.
    - surprise_second_quart_onsets: dict, contains the timetamps of outcomes within the second quartile of surprise. Keys are task session and values are timestamps within that task session. 
    - surprise_second_quart_nums: dict, contains a cumulative count of the number of surprising observations within the second quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of surprising observations within that task session.
    - epoched_pupil_second_surprise_df: pd.DataFrame, containing transposed epoched data for the second quartile of surprise (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - second_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio information for each generated epoch around the second quartile of surprise.
    - second_surprise_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_second_surprise_df that have been removed due to an excessive blink ratio.
    - surprise_third_quart_onsets: dict, contains the timetamps of outcomes within the third quartile of surprise. Keys are task session and values are timestamps within that task session. 
    - surprise_third_quart_nums: dict, contains a cumulative count of the number of surprising observations within the third quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of surprising observations within that task session.
    - epoched_pupil_third_surprise_df: pd.DataFrame, containing transposed epoched data for the third quartile of surprise (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - third_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the third quartile of surprise.
    - third_surprise_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_third_surprise_df that have been removed due to an excessive blink ratio.
    - surprise_fourth_quart_onsets: dict, contains the timetamps of outcomes within the fourth quartile of surprise. Keys are task session and values are timestamps within that task session. 
    - surprise_fourth_quart_nums: dict, contains a cumulative count of the number of surprising observations within the fourth quartile identified within each task session. Keys are task sessions and values are the cumulative count of the number of surprising observations within that task session.
    - epoched_pupil_fourth_surprise_df: pd.DataFrame, containing transposed epoched data for the fourth quartile of surprise (each column represents one epoch). Computed variables include the calculated mean and measures of variance (std and sem). 
    - fourth_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around the fourth quartile of surprise.
    - fourth_surprise_epochs_to_remove: pd.index, indices of epochs within the epoched_pupil_fourth_surprise_df that have been removed due to an excessive blink ratio.
    """
    # Filter across quartiles within the surprise category
    filtered_first_quartile = surprise_df[surprise_df['surprise_cat'] == 'first_quartile'].copy()
    filtered_second_quartile = surprise_df[surprise_df['surprise_cat'] == 'second_quartile'].copy()
    filtered_third_quartile = surprise_df[surprise_df['surprise_cat'] == 'third_quartile'].copy()
    filtered_fourth_quartile = surprise_df[surprise_df['surprise_cat'] == 'fourth_quartile'].copy()

    pupil_by_surprise_first_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_first_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_surprise_second_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_second_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_surprise_third_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_third_quartile, to_session=to_session, print_event_df_shape=True)
    pupil_by_surprise_fourth_quart = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_fourth_quartile, to_session=to_session, print_event_df_shape=True)
   
    # Creating dictionaries
    surprise_first_quart_onsets, surprise_first_quart_nums = create_pupil_epoch_dicts(pupil_by_surprise_first_quart)
    surprise_second_quart_onsets, surprise_second_quart_nums = create_pupil_epoch_dicts(pupil_by_surprise_second_quart)
    surprise_third_quart_onsets, surprise_third_quart_nums = create_pupil_epoch_dicts(pupil_by_surprise_third_quart)
    surprise_fourth_quart_onsets, surprise_fourth_quart_nums = create_pupil_epoch_dicts(pupil_by_surprise_fourth_quart)

    # Checking the number of surprising outcomes in each quartile 
    print(f"Condition {condition_str}:")
    for session, events in surprise_first_quart_nums.items():
      print(f"In task session {session} there are {len(events)} surprising outcomes in the first quartile") 

    print(f"Condition {condition_str}:")
    for session, events in surprise_second_quart_nums.items():
      print(f"In task session {session} there are {len(events)} surprising outcomes in the second quartile") 

    print(f"Condition {condition_str}:")
    for session, events in surprise_third_quart_nums.items():
      print(f"In task session {session} there are {len(events)} surprising outcomes in the third quartile") 

    print(f"Condition {condition_str}:")
    for session, events in surprise_fourth_quart_nums.items():
      print(f"In task session {session} there are {len(events)} surprising outcomes in the fourth quartile") 

    # Generating session lists for surprise events within each quartile
    first_quart_surp_ses_list = list(surprise_first_quart_onsets.keys())
    second_quart_surp_ses_list = list(surprise_second_quart_onsets.keys())
    third_quart_surp_ses_list = list(surprise_third_quart_onsets.keys())
    fourth_quart_surp_ses_list = list(surprise_fourth_quart_onsets.keys())

    # Generating pupil response around surprising observations 
    print("Analysing first quartile of surprise. Processing...") 
    preprocessed_pupil_first_surp_df, pupil_trig_first_surp_df, pupil_ses_first_surp_df, blink_first_surp_df, epoched_pupil_first_surprise_df, first_surprise_blink_ratio_df, first_surprise_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=surprise_first_quart_onsets, dict_num=surprise_first_quart_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5],session_list=first_quart_surp_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysing second quartile of surprise. Processing...") 
    preprocessed_pupil_second_surp_df, pupil_trig_second_surp_df, pupil_ses_second_surp_df, blink_second_surp_df, epoched_pupil_second_surprise_df, second_surprise_blink_ratio_df, second_surprise_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=surprise_second_quart_onsets, dict_num=surprise_second_quart_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5],session_list=second_quart_surp_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysisng third quartile of surprise. Processing...") 
    preprocessed_pupil_third_surp_df, pupil_trig_third_surp_df, pupil_ses_third_surp_df, blink_third_surp_df, epoched_pupil_third_surprise_df, third_surprise_blink_ratio_df, third_surprise_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=surprise_third_quart_onsets, dict_num=surprise_third_quart_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5],session_list=third_quart_surp_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)
    print("Analysing fourth quartile of surprise. Processing...") 
    preprocessed_pupil_fourth_surp_df, pupil_trig_fourth_surp_df, pupil_ses_fourth_surp_df, blink_fourth_surp_df, epoched_pupil_fourth_surprise_df, fourth_surprise_blink_ratio_df, fourth_surprise_epochs_to_remove = load_and_process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, condition_task_sessions=condition_task_sessions, from_session=from_session, to_session=to_session, dict_onsets=surprise_fourth_quart_onsets, dict_num=surprise_fourth_quart_nums, window_start_before=0.2, window_end_after=1.2, lpf=[5],session_list=fourth_quart_surp_ses_list, select_eye=select_eye, remove_epochs=remove_epochs)

    # Option to plot pupil response of the pupil diameter timelocked to high and low surprising observations
    if plot_pupil_by_surprise:
              
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_surprise_df['time'],
            y=epoched_pupil_first_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(255, 211, 0, 1)'),
            name='First Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_surprise_df['time'],
            y=epoched_pupil_first_surprise_df['mean'] + epoched_pupil_first_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(255, 211, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_first_surprise_df['time'],
            y=epoched_pupil_first_surprise_df['mean'] - epoched_pupil_first_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 211, 0, 0.2)',
            line=dict(color='rgba(255, 211, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_surprise_df['time'],
            y=epoched_pupil_second_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 183, 235, 1)'),
            name='Second Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_surprise_df['time'],
            y=epoched_pupil_second_surprise_df['mean'] + epoched_pupil_second_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 183, 235, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_second_surprise_df['time'],
            y=epoched_pupil_second_surprise_df['mean'] - epoched_pupil_second_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 183, 235, 0.2)',
            line=dict(color='rgba(0, 183, 235, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_surprise_df['time'],
            y=epoched_pupil_third_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(213, 0, 249, 1)'),
            name='Third Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_surprise_df['time'],
            y=epoched_pupil_third_surprise_df['mean'] + epoched_pupil_third_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(213, 0, 249, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_third_surprise_df['time'],
            y=epoched_pupil_third_surprise_df['mean'] - epoched_pupil_third_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(213, 0, 249, 0.2)',
            line=dict(color='rgba(213, 0, 249, 0.1)'),
            showlegend=False
        ))


        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_surprise_df['time'],
            y=epoched_pupil_fourth_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 204, 102, 1)'),
            name='Fourth Quartile'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_surprise_df['time'],
            y=epoched_pupil_fourth_surprise_df['mean'] + epoched_pupil_fourth_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 204, 102, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_fourth_surprise_df['time'],
            y=epoched_pupil_fourth_surprise_df['mean'] - epoched_pupil_fourth_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 204, 102, 0.2)',
            line=dict(color='rgba(0, 204, 102, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title=f'Baseline corrected right pupil response epoched around surprising outcomes, split by quartile, for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_epoch_avg.show()

    return surprise_first_quart_onsets, surprise_first_quart_nums, epoched_pupil_first_surprise_df, first_surprise_blink_ratio_df, first_surprise_epochs_to_remove, surprise_second_quart_onsets, surprise_second_quart_nums, epoched_pupil_second_surprise_df, second_surprise_blink_ratio_df, second_surprise_epochs_to_remove, surprise_third_quart_onsets, surprise_third_quart_nums, epoched_pupil_third_surprise_df, third_surprise_blink_ratio_df, third_surprise_epochs_to_remove, surprise_fourth_quart_onsets, surprise_fourth_quart_nums, epoched_pupil_fourth_surprise_df, fourth_surprise_blink_ratio_df, fourth_surprise_epochs_to_remove