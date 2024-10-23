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
from pick_best_eye import pick_best_eye



def pupil_data_check(data_tob, blink_ses, pupil_ses, file_path, session, file_size_threshold = 2,  plot_original_data = True, plot_reduced_data = True):
    """
    Function to ensure that the pupil data loaded from files does not contain erroneous data. Currently function is based on the large files that were found after the first couple of fparticipants, but should be adapted to rectify any other issues that crop up. 

    Parameters:
    - data_tob: class 'pupil' object, instantiation of the 'pupil' class. Directories within objject can call specific methods and objects, such as data storage objects (e.g. data_tob.data call dataframe containing pupil data, etc).
    - blink_ses: pd.DataFrame, contains the start and end times of each blink, for each task session, identified in the import_data method of the pupil class. Also contains the duration of each blink.
    - pupil_ses: pd.DataFrame, contains the start and end times of task session, based on the 'start' and 'end' triggers inserted within each file. The duration of each session is calculated.
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
    # Check is initiatally based on erroneously large file sizes 
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

            # 'Diff' here could be used to identify - if the duplicated sequence is consecutive - the end of one sequence and the beginning of another. Not developed this yet.  
            data_tob.data['time_diff'] = data_tob.data['time_stamp_rel_s'].diff()

            # The reduction of the large file is primarily based on triggers. 'pupil_ses' is a data frame containing the start and end times of each file based on the 'start' and 'end' triggers inserted into the file
            start_value = int(pupil_ses['start'].iloc[0])
            end_value = int(pupil_ses['end'].iloc[0])

            # Uses the start and end triggers to define the range across pupil data will be reduced down to
            start_indices = data_tob.data.loc[np.isclose(data_tob.data['time_stamp'], start_value)].index
            end_indices = data_tob.data.loc[np.isclose(data_tob.data['time_stamp'], end_value)].index

            # Given the duplication of data, if there are more than two values defining the location of the 'start' and 'end' trigger location, takes the first instance. 
            if len(start_indices) > 1 and len(end_indices) > 1:
                start_idx, end_idx = start_indices[0], end_indices[0]
                # Overwrites the pupil data object based on defined range. Given the need to epoch around data points, the range based on the 'start' and 'end' trigger is extendeda little bit in both directions. Also makes this fle relatively similar in size to other files. Adjustment values are pretty arbitary, may need to be adjusted in future. 
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

                print(f'Multiple indices of start and end triggers identified. File {session} has been reduced to data between first instances of start and end triggers')
            else:
                print(f'File {session} excessively large, but multiple indices of start and end triggers not found ')
    
        else:
            print("There is no duplicated data found in file {session}.")
    else:
        print(f"File {session} did not exceed the file size threshold.")
    
    pupil_data = data_tob.data.copy()
    pupil_data['task_session'] = session
    
    # Returns both the self object of the pupil class and a copy of the data that can be used
    return data_tob, pupil_data, blink_ses


def load_pupil_data(participant_str, condition_str, condition_task_sessions, selected_eye = None, plot_pupil_by_task_session=False):
    """
    Function to load and read .txt file from tobii eyetracker. Loads and stores the data for each task session within a given condition, using import_data function within preprocess_eyedata script, created by Florent, Maëva, Aziza and Yannaël. If a file is excessively large (i.e. there is a lot of data after the session has finished), the data stored within the dataframe for all task sessions will be a reduced version of the raw data. Also separates and stores trigger information, session characteristics (e.g. start and end time, with duration calculated) and blink information (based on custom code to identify blinks - see preprocess_eyedata). There is an option to plot the raw eye data for each task session. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - selected_eye: string, defines the eye that is loaded. Options are 'right', 'left' or if None, will use the pick_best_eye function 
    - plot_pupil_by_task_session: boolean, if True, creates subplots the right eye data for each task session across time using Plotly. (default = False)


    Returns:
    - pupil_df: pd.DataFrame, contains the pupil data recorded from each task session
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session
    - pupil_ses_df: pd.DataFrame, contains each task session start time, end time and duration.
    - blink_df: pd.DatFRame, contains the start and end times of each identified blink within a task session. Also contains the duration of each blink, plus the relative and absolute number of blinks, within and across task sessions.      
    """
    # Initialise class
    data_tob = pupil()

    # Create lists to store data for each task session
    pupil_data_ses_list = []
    pupil_trig_ses_list = []
    pupil_ses_list = []
    blink_ses_list = []

    # Loop over sessions/files to extract pupil data
    for session in condition_task_sessions:
        file_dir = f"/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition_str}/eye"
        file_name = f"sub-{participant_str}_ses-{condition_str}_task-AdalearnVNS_run-{session}_eyetracker.txt"
        file_path = f"{file_dir}/{file_name}"

        if selected_eye is None:
            list_subject_selected = [int(participant_str)]
            from_session = condition_task_sessions[0]
            to_session = condition_task_sessions[-1]
            selected_eye = pick_best_eye(file_name, list_subject_selected, from_session, to_session)
        
        # Import data using function within pupil class
        data_tob.import_data(file_name, events_warner='MSG', directory = file_dir, eye=selected_eye, col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='tobii')

        # Extract relevant data from class
        trigger_data = data_tob.events
        nan_ratio = data_tob.nan_ratio
        blink_start = data_tob.blink_start
        blink_end = data_tob.blink_end

        # Create lists to extract data from class objects such that the data can be stored in a dataframe. For example, trigger_data contains a list of dictionaries, which contain keys: 'time' and 'msg', within which values and lists of strings containing the information within that line of the .txt file. 
        outcome_data = []
        session_start = []
        session_end = []
        pupil_end = []

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
            'end': session_end
        })

        # Creates a dataframe to store the extracted blink information for each task session 
        blink_ses = pd.DataFrame({
            'task_session': session,
            'blink_start': blink_start,
            'blink_end': blink_end
        })
        blink_ses['blink_dur'] = (blink_ses['blink_end'] - blink_ses['blink_start']) / 1000000

        # Check that data files are not clearly erroneous 
        data_tob, pupil_data, blink_ses = pupil_data_check(data_tob, blink_ses=blink_ses, pupil_ses=pupil_ses, file_path=file_path, file_size_threshold=2, session=session, plot_original_data=False, plot_reduced_data=False)

        # Append the data to the empty list to store the data for that task session
        pupil_data_ses_list.append(pupil_data)
        pupil_trig_ses_list.append(pupil_trig_ses)
        pupil_ses_list.append(pupil_ses)
        blink_ses_list.append(blink_ses)

    # Concatenate data across taks sessions and store in a single dataframe   
    pupil_df = pd.concat(pupil_data_ses_list, ignore_index=True)

    pupil_trig_df = pd.concat(pupil_trig_ses_list, ignore_index=True)
    pupil_trig_df['outcome_num'] = pupil_trig_df.groupby('task_session').cumcount() 
    pupil_trig_df['cum_outcome_num'] = range(len(pupil_trig_df))
 
    pupil_ses_df = pd.concat(pupil_ses_list)
    pupil_ses_df['duration'] = (pupil_ses_df['end'].astype(int) - pupil_ses_df['start'].astype(int)) / 1000000

    blink_df = pd.concat(blink_ses_list, ignore_index=True)
    blink_df['blink_count'] = blink_df.groupby('task_session').cumcount() 
    blink_df['cum_blink_count'] = range(len(blink_df))

    # Modifying stored data_tob objects so they contain all of the task session data
    data_tob.data = pupil_df.drop('time_diff', axis=1)
    data_tob.blink_start = np.array(blink_df['blink_start'])
    data_tob.blink_end = np.array(blink_df['blink_end'])

    # Option to plot the pupil data
    if plot_pupil_by_task_session:   
        # Defining the unique task sessions within the data file 
        unique_sessions = pupil_df['task_session'].unique()
        num_sessions = len(unique_sessions)

        colors = ['blue', 'green', 'red'] 

        # Create subplots, one for each task session
        fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=False,
                            subplot_titles=[f'Pupil diameter (blue) plus x (green) and y (red) coordinates of the right eye in task session {session}' for session in unique_sessions])

        # Add pupil traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = pupil_df[pupil_df['task_session'] == session]

            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['diameter'], mode='lines', line=dict(color=colors[0]), name=f'{selected_eye} pupil diameter'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['x_pos'], mode='lines', line=dict(color=colors[1]), name=f'{selected_eye} eye x position'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['y_pos'], mode='lines', line=dict(color=colors[2]), name=f'{selected_eye} eye y position'), row=i, col=1)

        fig.update_layout(height=900*num_sessions, 
                          width=1200,
                          title_text=f'Right eye data across task sessions  for participant {participant_str} in condition {condition_str}',
                          xaxis_title='Time (s)',
                          yaxis_title='Pupil Diameter (mm)',
                          showlegend=False)
        
        fig.show()

    
    return pupil_df, pupil_trig_df, pupil_ses_df, blink_df


def process_pupil_data(participant_str, list_subject_selected, condition_str, from_session, to_session, dict_onsets, dict_num, session_list = None, outlier_limit_std = 10, outlier_correct_average_amount = 15,  window_start_before = 0.2, window_end_after = 3, blink_ratio_threshold = 0.20, reject_epochs = False, z_score = True, plot_interpolation_process = False):
    """
    Function to preprocess pupil data into epochs around defined outcomes of interest (e.g. change points, high and low surprise, etc). Preprocessing taken from preprocess_eyedata script, created by Florent, Maëva, Aziza and Yannaël. Calculates the mean and standard deviation across each identified epoch for all task session.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - dict_onsets: dict, contains timestamps of outcomes of interest (e.g. change points, high and low surprise, etc). Keys are task session and values are time stamps within that task session. 
    - dict_nums: dict, contains a cumulative count of the number of outcomes of interest identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session
    - outlier_limit_std: int, defines the threshold of the standard deviation of the derivative when preprocessing the pupil data. Default = 10
    - outlier_correct_average_amount: int, the number of data points to the left and at the right of the artifact which will replace the value of the artifact by the value of the mean of the data around it. Default = 10.
    - window_start_before: int, value defining the start of the epoch window in time (seconds). Default = 1
    - window_end_after: int, value defining the end of the epoch window in time (seconds). Default = 3
    - blink_ratio_threshold:, integer, value determining the threshold criteria for the removal of an epoch due to excessive blink ratio (default = 0.2)
    - reject_epochs: boolean, parameter determining whether epochs with an excessive blink ratio should be removed (default = False)
    - z_score: boolean, option to z_score data before computing epochs. Default = True.
    - plot: boolean, option to plot the mean and standard deviation across epochs using Plotly. Default = True. 

    Returns:
    - epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across epochs and the timestamp. 
    - blink_ratio_epoch_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch
    """
    # Running data through methods within class pupil taken from preprocess_eyedata. 
    # Initialise class
    data_tob = pupil()

    # List to store dataframes containing the epoched data from each task session
    pre_process_data_list = []
    epoched_data_list = []
    blink_ratio_epoch_list = []

    # Determine sessions to iterate over
    if session_list is not None:
        sessions_to_process = session_list
    else:
        sessions_to_process = range(from_session, to_session + 1)

    # Iterate through each task session file to analyse the pupil data
    for session in sessions_to_process:
        session_str = str(session).zfill(2)
        file_dir = f"/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition_str}/eye"
        file_name = f"sub-{participant_str}_ses-{condition_str}_task-AdalearnVNS_run-{session_str}_eyetracker.txt"
        file_path = f"{file_dir}/{file_name}"

    
        #list_best_eye = pick_best_eye(file_path, list_subject_selected, from_session, to_session)
        #print(list_best_eye)
        

        for subject, i_subject in zip(list_subject_selected, range(len(list_subject_selected))):

            # Import the datafile
            data_tob.import_data(filename = file_name,
                             events_warner='MSG', directory = file_dir, eye='right', 
                             col=[0, 1, 2],
                             col_name=['x_pos', 'y_pos', 'diameter'],
                             eyetracker='tobii')
            
            # Extract relevant data from class
            trigger_data = data_tob.events
            nan_ratio = data_tob.nan_ratio
            blink_start = data_tob.blink_start
            blink_end = data_tob.blink_end

            # Create lists to extract data from class objects such that the data can be stored in a dataframe. For example, trigger_data contains a list of dictionaries, which contain keys: 'time' and 'msg', within which values and lists of strings containing the information within that line of the .txt file. 
            outcome_data = []
            session_start = []
            session_end = []
            pupil_end = []

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
                'end': session_end
            })

            # Creates a dataframe to store the extracted blink information for each task session 
            blink_ses = pd.DataFrame({
                'task_session': session,
                'blink_start': blink_start,
                'blink_end': blink_end
            })
            blink_ses['blink_dur'] = (blink_ses['blink_end'] - blink_ses['blink_start']) / 1000000
            
            data_tob, pupil_data, blink_ses = pupil_data_check(data_tob, blink_ses=blink_ses, pupil_ses=pupil_ses, file_path=file_path, file_size_threshold=2, session=session, plot_original_data=False, plot_reduced_data=False)

            # Preprocess the data. If need, can plot the preprocessed pupil time series. To see the blink ratio use data_tob.epoch_blink_ratio. For more complete summary can insert data_tob.epochs_info (provides a data frame with the epoch number, the blink ratio, NaN ratio per epoch and conditions [rejected etc])
            time, diameter, diameter_int, x_pos, x_pos_int, y_pos, y_pos_int = data_tob.preprocess(plot=False, remove_outlier=True, outlier_limit_std=outlier_limit_std, outlier_correct_average_amount=outlier_correct_average_amount)

            # extracting preprocessed data to check interpolation
            processed_data = data_tob.data.copy()
            processed_data['task_session'] = session  
            pre_process_data_list.append(processed_data)
      
            # Epoch the data and apply baseline correction
            data_tob.epoch_tobii(dict_onsets[session_str], variable='diameter_int', decimate=1, reject=reject_epochs, blink_ratio_thd=blink_ratio_threshold, before=window_start_before, after=window_end_after, baseline=[-0.2, 0], conditions={'event_num': dict_num[session_str]}, z_score_data=z_score)

            # Check to ensure the correct number of epochs
            print("For participant n:", subject, " in session = ", session, "there are:", len(data_tob.epochs), "epochs")
            print(data_tob.epochs_info)

            # Appending dataframes containing epoched data to list for each task session
            epoched_data = pd.DataFrame([epoch for epoch in data_tob.epochs])
            epoched_data_list.append(epoched_data)

            # Append list to blink_ratio list 
            blink_ratio_epoch_list.append(data_tob.epoch_blink_ratio)

    # Concatenate pre_processed pupil dataframes across task sessions
    preprocessed_pupil_df = pd.concat(pre_process_data_list, ignore_index=True)

    # Bespoke option to visulaize interpolation process (pupil method 'preprocess' has an option, but I didn't like the visualisation)
    if plot_interpolation_process:
        unique_sessions = preprocessed_pupil_df['task_session'].unique()
        num_sessions = len(unique_sessions)

        # Create subplots, one for each task session
        fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=False,
                            subplot_titles=[f'Preprocessing of pupil diameter for the right eye in task session {session}' for session in unique_sessions])

        # Add pupil traces to each subplot
        for subplot_idx, session in enumerate(unique_sessions, start=1):
            session_data = preprocessed_pupil_df[preprocessed_pupil_df['task_session'] == session]

            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['diameter'], mode='lines', name='Original Pupil Diameter', 
                        line=dict(color='rgba(0, 0, 255, 0.5)')), row=subplot_idx, col=1)
            fig.add_trace(go.Scatter(x=session_data['time_stamp_rel_s'], y=session_data['diameter_int'], mode='lines', name='Interpolated Pupil Data', 
                        line=dict(color='rgba(255, 0, 0, 1)', dash='dot')), row=subplot_idx, col=1) 

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
                                xref=f'x{subplot_idx}',  # Link to specific x-axis
                                yref=f'y{subplot_idx}')  # Link to specific y-axis
                    interpolating = False

        fig.update_layout(height=900*num_sessions, 
                        width=1200,
                        title_text=f'Preprocessed right pupil diameter data across task sessions for participant {participant_str} in condition {condition_str}',
                        xaxis_title='Time (s)',
                        yaxis_title='Pupil Diameter (mm)',
                        showlegend=False)

        fig.show()


    # Generate dataframe with blink_ratio for each epoch vertically stacked
    flattened_array = np.concatenate(blink_ratio_epoch_list)

    blink_ratio_epoch_df = pd.DataFrame(flattened_array, columns=['blink_ratio'])
    blink_ratio_epoch_df['above_threshold'] = blink_ratio_epoch_df['blink_ratio'] >= blink_ratio_threshold

    # Collating the timelocked epoched pupil data to calculate descriptive statistics for plotting
    epoched_pupil_df = pd.concat(epoched_data_list, ignore_index=True).T
    numeric_data = epoched_pupil_df.iloc[:, :]
    epoched_pupil_df['mean'] = numeric_data.mean(axis=1)
    epoched_pupil_df['std'] = numeric_data.std(axis=1)
    epoched_pupil_df['sem'] = epoched_pupil_df['std'] / np.sqrt(numeric_data.shape[1])
    epoched_pupil_df['time'] = data_tob.epochs_times

    return epoched_pupil_df, blink_ratio_epoch_df


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
    

    # Identify triggers in pupil file by aligning with outcomes that signify some event of interest. 3 step process.
    # Change pupil trig datafile task session to integer (removing zero padding)
    pupil_trig_df['task_session'] = pupil_trig_df['task_session'].astype(int)

    # Step 1: Ensure that the same number of unique task session values exist in both the pupil and behavioural event dataframes. If not, signifies that either some data is missing, or data in certain task sessions have been excluded (e.g. there were nor high/low surprising observations in a particular task session):
    pupil_sessions = set(pupil_trig_df['task_session'].unique())
    event_sessions = set(event_df['task_session'].unique())

    if pupil_sessions != event_sessions:
        print(f"ERROR MISMATCH: Pupil data has {len(pupil_sessions)} unique task sessions ({pupil_sessions}), while behavioural events were taken from {len(event_sessions)} unique task sessions ({event_sessions}).")
    else:
        print("MATCH: Task sessions match across both dataframes")

    # Step 2: Check that the length of the pupil trig data equals 75 * the number of unique task sessions. Indicates that all of the pupil data is being included.
    expected_length = 75 * to_session

    if len(pupil_trig_df) != expected_length:
        print(f"ERROR MISMATCH: Pupil data has {len(pupil_trig_df)} rows, but there should be {expected_length} expected rows.")
    else:
        print("MATCH: Pupil data is as expected given the number of task sessions performed.")

    # Step 3: If any of the checks above are violated, proceed with finding the matching rows based on relative task session trigger number
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

    # If checks were not violated, match pupil to behavioural event based on cumulative outcome number across all task sessions
    else:

        pupil_by_event = pupil_trig_df[pupil_trig_df['cum_outcome_num'].isin(event_df['cum_outcome_num'])]

        print(f"Merged dataframe based on cumulative triggers. The number of triggers found was: {pupil_by_event.shape[0]}.")

    # Return the pupil task session to zero-padded so compatible with pupil processing function 
    pupil_by_event['task_session'] = pupil_by_event['task_session'].astype(str).str.zfill(2)

    duplicate_cols = [col for col in pupil_by_event.columns if col.endswith('_x') or col.endswith('_y')]
    if duplicate_cols:
        pupil_by_event = pupil_by_event.drop(columns=duplicate_cols)
   
    return pupil_by_event


def all_outcome_pupil_epochs(participant_str, condition_str, list_subject_selected, from_session, to_session, pupil_trig_df, session_list = None, plot_averaged_pupil_response = True, plot_raster_pupil_response = True):
    """
    Function to preprocess and analysis pupil data around identified change points in the task. Uses process_pupil_data function to run the preporocessing analysis. In addition to analysising the data around change points, there is an option to plot the mean and std derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session.
    - plot_pupil_by_cp: boolean, option to plot the mean and standard deviation of the epoched pupil data around the change points using PLotly. (Default = False).

    Returns:
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across change point defined epochs and the timestamp.
    - blink_ratio_cp_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around change points
    """
    # Creating dictionaries for storage of change point time and number 
    outcome_onsets = {}
    outcome_nums = {}

    # For loop that extracts identified change point times for each task session 
    for session, outcome in pupil_trig_df.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      outcome_onset_values = outcome['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      outcome_num_values = list(range(1, len(outcome) + 1))

      # Define dictionaries used to define epochs
      outcome_onsets[session] = outcome_onset_values
      outcome_nums[session] = outcome_num_values

    # Checking the number of change points identified
    print(f"Condition {condition_str}:")
    for session, events in outcome_nums.items():
      print(f"In task session {session} there are {len(events)} outcome observations") 

    # Analysing pupil data based on identified change point indices
    outcome_epoched_pupil_df, outcome_blink_ratio_df = process_pupil_data(participant_str, list_subject_selected, condition_str, from_session, to_session, outcome_onsets, outcome_nums, session_list, window_start_before=0.2, window_end_after=1.45)

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
            title=f'Baseline corrected right pupil response to each observations for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_pupil_avg.show()


    if plot_raster_pupil_response:
        pupil_epoch_matrix = [outcome_epoched_pupil_df.iloc[:, i].values for i in range(outcome_epoched_pupil_df.shape[1] - 4)]

        fig = px.imshow(pupil_epoch_matrix, 
                        labels=dict(x="Time (s)", y="Outcome number", color="Pupil Diameter (z-score)"),
                        x=outcome_epoched_pupil_df['time'].values,  
                        aspect="auto",  
                        origin="lower"  
                    )

        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(title=f'Epoched pupil diameter time series heatmap across all outcome observations for participant {participant_str} in condition {condition_str}')

        fig.show()

    return outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df


def outcome_colour_pupil_epochs(participant_str, condition_str, list_subject_selected, from_session, to_session, pupil_trig_df, session_list = None, plot_pupil_by_colour_response = True):
    """
    Function to preprocess and analysis pupil data around identified change points in the task. Uses process_pupil_data function to run the preporocessing analysis. In addition to analysising the data around change points, there is an option to plot the mean and std derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session.
    - plot_pupil_by_cp: boolean, option to plot the mean and standard deviation of the epoched pupil data around the change points using PLotly. (Default = False).

    Returns:
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across change point defined epochs and the timestamp.
    - blink_ratio_cp_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around change points
    """
    # Splitting the pupil trig dataframe into two based on outcome 
    pupil_trig_1_df = pupil_trig_df[pupil_trig_df['outcome'] == 1]
    pupil_trig_0_df = pupil_trig_df[pupil_trig_df['outcome'] == 0] 

    # Creating dictionaries for storage of 'blue' outcome time and number 
    outcome_1_onsets = {}
    outcome_1_nums = {}

    # For loop that extracts 'blue' outcomes for each task session 
    for session, outcome in pupil_trig_1_df.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      outcome_1_onset_values = outcome['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      outcome_1_num_values = list(range(1, len(outcome) + 1))

      # Define dictionaries used to define epochs
      outcome_1_onsets[session] = outcome_1_onset_values
      outcome_1_nums[session] = outcome_1_num_values

    # Checking the number of 'blue' outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in outcome_1_nums.items():
      print(f"In task session {session} there are {len(events)} 'blue' outcomes") 

    # Creating dictionaries for storage of 'yellow' outcome time and number 
    outcome_0_onsets = {}
    outcome_0_nums = {}

    # For loop that extracts 'yellow' outcomes for each task session 
    for session, outcome in pupil_trig_0_df.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      outcome_0_onset_values = outcome['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      outcome_0_num_values = list(range(1, len(outcome) + 1))

      # Define dictionaries used to define epochs
      outcome_0_onsets[session] = outcome_0_onset_values
      outcome_0_nums[session] = outcome_0_num_values

    # Checking the number of 'yellow' outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in outcome_0_nums.items():
      print(f"In task session {session} there are {len(events)} 'yellow' outcomes") 

    # Analysing pupil data based on outcome values indices
    outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df = process_pupil_data(participant_str, list_subject_selected, condition_str, from_session, to_session, outcome_1_onsets, outcome_1_nums, session_list, window_start_before=0.2, window_end_after=1.45)

    outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df = process_pupil_data(participant_str, list_subject_selected, condition_str, from_session, to_session, outcome_0_onsets, outcome_0_nums, session_list, window_start_before=0.2, window_end_after=1.45)


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
            title=f'Baseline corrected right pupil response to each outcome colour presentation for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_pupil_avg.show()


    return outcome_1_onsets, outcome_1_nums, outcome_1_epoched_pupil_df, outcome_1_blink_ratio_df, outcome_0_onsets, outcome_0_nums, outcome_0_epoched_pupil_df, outcome_0_blink_ratio_df, 



def change_point_pupil_epochs(participant_str, condition_str, list_subject_selected, from_session, to_session, pupil_trig_df, change_point_indices, session_list = None, plot_pupil_by_cp=False):
    """
    Function to preprocess and analysis pupil data around identified change points in the task. Uses process_pupil_data function to run the preporocessing analysis. In addition to analysising the data around change points, there is an option to plot the mean and std derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - session_list: list, optional, specific list of sessions to process. If not provided, the function will use the range defined by from_session and to_session.
    - plot_pupil_by_cp: boolean, option to plot the mean and standard deviation of the epoched pupil data around the change points using PLotly. (Default = False).

    Returns:
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across change point defined epochs and the timestamp.
    - blink_ratio_cp_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around change points
    """
    pupil_by_change_points = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=change_point_indices, to_session=to_session, print_event_df_shape=True)


    # Creating dictionaries for storage of change point time and number 
    change_point_onsets = {}
    change_point_nums = {}

    # For loop that extracts identified change point times for each task session 
    for session, group in pupil_by_change_points.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      change_point_onset_values = group['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      change_point_num_values = list(range(1, len(group) + 1))

      # Define dictionaries used to define epochs
      change_point_onsets[session] = change_point_onset_values
      change_point_nums[session] = change_point_num_values

    # Checking the number of change points identified
    print(f"Condition {condition_str}:")
    for session, events in change_point_nums.items():
      print(f"In task session {session} there are {len(events)} identified change points") 

    # Analysing pupil data based on identified change point indices
    epoched_pupil_cp_df, blink_ratio_cp_df = process_pupil_data(participant_str, list_subject_selected, condition_str, from_session, to_session, change_point_onsets, change_point_nums, session_list)

    # Plotting the mean (and std) response of pupil response timelocked to change points
    if plot_pupil_by_cp:
        
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='Mean',
            showlegend=False 
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

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title=f'Baseline corrected right pupil response epoched around change points for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_epoch_avg.show()

    return change_point_onsets, change_point_nums, epoched_pupil_cp_df, blink_ratio_cp_df


def surprise_pupil_epochs(participant_str, list_subject_selected, condition_str, from_session, to_session, surprise_df, pupil_trig_df, high_surp_ses_list = None, low_surp_ses_list = None, plot_pupil_by_surprise=True):
    """"
     Function to preprocess and then compute epoched pupil diameter response around modelled "high" and "low" suprising observations. Preprocessing taken from preprocess_eyedata script. Calculates the mean and standard deviation across each identified epoch for all task session. In addition to analysising the data around surprise, there is an option to plot the mean and std derived from all instances across all task sessions. Parameters defining the preprocessing of the pupil data should be modified in the process_pupil_data function directly.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - surprise_df: pd.DataFrame, contains computed surprise values and the categorised values. Used to define timestamps for the categorised surprise values (which are contained in a dictionary, with keys being task sessions and values are time stamps within that task session) and a cumulative count of the number of surprise values categorised as either high or low (again, a dictionary with keys being task sessions and values being the cumulative count within that task session).
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session, extracted from the pupil data files. 
    - high_surp_ses_list: list, optional, specific list of sessions to process for the high surprising observations. If not provided, the function will use the range defined by from_session and to_session.
    - low_surp_ses_list: list, optional, specific list of sessions to process for the low surprising observations. If not provided, the function will use the range defined by from_session and to_session.
    - plot_pupil_by_surprise: boolean, option to plot the mean and standard deviation across epochs using Plotly. (Default = True). 

    Returns:
    - surprise_low_onsets: dict, contains low surprise observation timestamps. Keys are task session and values are time stamps within that task session. 
    - surprise_low_nums: dict, contains a cumulative count of the number of low surprise observations identified within each task session. Keys are task sessions and values are the cumulative count of the number of low surprise observations within that task session.
    - surprise_high_onsets: dict, contains high surprise observation timestamps. Keys are task session and values are time stamps within that task session. 
    - surprise_high_nums: dict, contains a cumulative count of the number of high surprise observations identified within each task session. Keys are task sessions and values are the cumulative count of the number of high surprise observations within that task session.
    - epoched_pupil_low_surprise_df: pd.DataFrame, containing transposed epoched data for the low surprise condition (each column represents one epoch). Variables include the calculated mean and standard deviation epochs and the timestamp. 
    - epoched_pupil_high_surprise_df: pd.DataFrame, containing transposed epoched data for the high surprise condition (each column represents one epoch). Variables include the calculated mean and standard deviation epochs and the timestamp. 
    - low_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around low surprise observations.
    - high_surprise_blink_ratio_df: pd.DataFrame, contains the blink ratio inofrmation for each generated epoch around high surprise observations.
    """
    # Filter for 'below_first_threshold' and 'above_third_threshold' based on surprise category
    filtered_below_first = surprise_df[surprise_df['surprise_cat'] == 'below_first_quartile'].copy()
    filtered_above_third = surprise_df[surprise_df['surprise_cat'] == 'above_third_quartile'].copy()

    pupil_by_surprise_low = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_below_first, to_session=to_session, print_event_df_shape=True)
    pupil_by_surprise_high = align_pupil_to_events(pupil_trig_df=pupil_trig_df, event_df=filtered_above_third, to_session=to_session, print_event_df_shape=True)
   
    # Creating dictionaries for storage of low surprising outcomes: time/onset and number 
    surprise_low_onsets = {}
    surprise_low_nums = {}

    # For loop that extracts times for low surprising outcomes duirng each task session 
    for session, group in pupil_by_surprise_low.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      low_surprise_onset_values = group['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      low_surprise_num_values = list(range(1, len(group) + 1))

      # Define dictionaries used to define epochs
      surprise_low_onsets[session] = low_surprise_onset_values
      surprise_low_nums[session] = low_surprise_num_values

    # Checking the number of low surprising outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in surprise_low_nums.items():
      print(f"In task session {session} there are {len(events)} low surprising outocmes") 

    # Creating dictionaries for storage of high surprising outcomes: time/onset and number 
    surprise_high_onsets = {}
    surprise_high_nums = {}

    # For loop that extracts times for high surprising outcomes during each task session 
    for session, group in pupil_by_surprise_high.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      high_surprise_onset_values = group['time_stamp'].tolist()
    
      # Create a cumulative count list for the current group (session)
      high_surprise_num_values = list(range(1, len(group) + 1))

      # Define dictionaries used to define epochs
      surprise_high_onsets[session] = high_surprise_onset_values
      surprise_high_nums[session] = high_surprise_num_values

    # Checking the number of highly surprising outcomes identified
    print(f"Condition {condition_str}:")
    for session, events in surprise_high_nums.items():
      print(f"In task session {session} there are {len(events)} high surprising observations") 

        
    # Generating pupil response around surprising observations  
    epoched_pupil_low_surprise_df, low_surprise_blink_ratio_df = process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, from_session=from_session, to_session=to_session, dict_onsets=surprise_low_onsets, dict_num=surprise_low_onsets, session_list=low_surp_ses_list)
    epoched_pupil_high_surprise_df, high_surprise_blink_ratio_df = process_pupil_data(participant_str=participant_str, list_subject_selected=list_subject_selected, condition_str=condition_str, from_session=from_session, to_session=to_session, dict_onsets=surprise_high_onsets, dict_num=surprise_high_nums, session_list=high_surp_ses_list)

    # Plotting the mean (and std) response of the pupil diameter timelocked to high and low surprising observations
    if plot_pupil_by_surprise:
              
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_high_surprise_df['time'],
            y=epoched_pupil_high_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='High Surprise'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_high_surprise_df['time'],
            y=epoched_pupil_high_surprise_df['mean'] + epoched_pupil_high_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_high_surprise_df['time'],
            y=epoched_pupil_high_surprise_df['mean'] - epoched_pupil_high_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_low_surprise_df['time'],
            y=epoched_pupil_low_surprise_df['mean'],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 1)'),
            name='Low Surprise'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_low_surprise_df['time'],
            y=epoched_pupil_low_surprise_df['mean'] + epoched_pupil_low_surprise_df['sem'],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_low_surprise_df['time'],
            y=epoched_pupil_low_surprise_df['mean'] - epoched_pupil_low_surprise_df['sem'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 0, 0, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title=f'Baseline corrected right pupil response epoched around observations of high and low surprise for participant {participant_str} in condition {condition_str} (Mean +/- SEM)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter (z-score)'
        )       

        fig_epoch_avg.show()

    return surprise_low_onsets, surprise_low_nums, surprise_high_onsets, surprise_high_nums, epoched_pupil_low_surprise_df, epoched_pupil_high_surprise_df,  low_surprise_blink_ratio_df, high_surprise_blink_ratio_df