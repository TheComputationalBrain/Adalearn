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


def load_pupil_data(participant_str, condition, condition_task_sessions, plot=False):
    """ 
    Function to load and read .txt file from Tobii eyetracker. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. Also separates and stores trigger information taken from .txt file. There is an option to plot the right eye data for each task session. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - plot: boolean, if True, creates subplots the right eye data for each task session across time using Plotly.

    Returns:
    - pupil_df: pd.DataFrame, contains the pupil data recorded from each task session
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session  
    """

    # Set the working directory of the files
    base_dir = f'/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition}'
    os.chdir(base_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Initialize lists for storing parsed data from Tobii eye tracker
    tobii_time_list = []
    RE_x_list = []
    LE_x_list = []
    RE_y_list = []
    LE_y_list = []
    RE_diameter_list = []
    LE_diameter_list = []
    task_session_list = []

    # Initialize lists for storing trigger times inserted into Tobii eye tracker
    trig_time_list = []
    task_sess_trig_list = []

    # Read in .txt file as lines and then for each line, separate the data and store it in an appropriate list.
    for session in condition_task_sessions:
        file_name = f"sub-{participant_str}_ses-{condition}_task-AdalearnVNS_run-{session}_eyetracker.txt"
    
        with open(file_name, 'r') as file:
            lines = file.readlines()

        sessionstarttime = None
        sessionendtime = None

        for nlines in range(len(lines)):
            if 'RE' in lines[nlines]:
                continue
            if 'MSG' in lines[nlines]:
                ind1 = lines[nlines].find('M')
                if 'MSG: Session starts' in lines[nlines] and ind1 != -1:
                    sessionstarttime = float(lines[nlines][0:ind1-1])
                elif 'MSG: Received : stim' in lines[nlines] and ind1 != -1:
                    trig_time_list.append(float(lines[nlines][0:ind1-1]))
                    task_sess_trig_list.append(session)
                elif 'MSG: Session ends' in lines[nlines] and ind1 != -1:
                    sessionendtime = float(lines[nlines][0:ind1-1])
                else:
                    continue
            else:
                indtab = []
                ind1 = lines[nlines].find('\t')
                ind2 = lines[nlines].find(',')
                while ind1 != -1 or ind2 != -1:
                    if ind1 != -1:
                        indtab.append(ind1)
                        ind1 = lines[nlines].find('\t', ind1 + 1)
                    if ind2 != -1:
                        indtab.append(ind2)
                        ind2 = lines[nlines].find(',', ind2 + 1)
            
                tobii_time_list.append(float(lines[nlines][0:indtab[0]-1]))

                if lines[nlines][indtab[0]+3:indtab[1]] == 'nan':
                    RE_x_list.append(np.nan)
                else:
                    RE_x_list.append(float(lines[nlines][indtab[0]+3:indtab[1]]))

                if lines[nlines][indtab[1]+2:indtab[2]-2] == 'nan':
                    RE_y_list.append(np.nan)
                else:
                    RE_y_list.append(float(lines[nlines][indtab[1]+2:indtab[2]-2]))

                if lines[nlines][indtab[2]+3:indtab[3]] == 'nan':
                    LE_x_list.append(np.nan)
                else:
                    LE_x_list.append(float(lines[nlines][indtab[2]+3:indtab[3]]))

                if lines[nlines][indtab[3]+2:indtab[4]-2] == 'nan':
                    LE_y_list.append(np.nan)
                else:
                    LE_y_list.append(float(lines[nlines][indtab[3]+2:indtab[4]-2]))

                if lines[nlines][indtab[4]+2:indtab[5]-1] == 'nan':
                    RE_diameter_list.append(np.nan)
                else:
                    RE_diameter_list.append(float(lines[nlines][indtab[4]+2:indtab[5]-1]))

                if lines[nlines][indtab[5]+2:indtab[6]] == 'nan':
                    LE_diameter_list.append(np.nan)
                else:
                    LE_diameter_list.append(float(lines[nlines][indtab[5]+2:indtab[6]]))

                task_session_list.append(session)

    # Create a DataFrame for the pupil data from storage lists
    pupil_df = pd.DataFrame({
        'tobii_time': tobii_time_list,
        'RE_x': RE_x_list,
        'RE_y': RE_y_list,
        'LE_x': LE_x_list,
        'LE_y': LE_y_list,
        'RE_diameter': RE_diameter_list,
        'LE_diameter': LE_diameter_list,
        'task_session': task_session_list
    })

    # Create relative time variable in pupil_df
    pupil_df['tobii_time_rel'] = pupil_df.groupby('task_session')['tobii_time'].diff().fillna(0).groupby(pupil_df['task_session']).cumsum() / 1000000

    # Create a DataFrame for the trigger data stored in lists
    pupil_trig_df = pd.DataFrame({
        'trigger_time': trig_time_list,
        'task_session': task_sess_trig_list 
    })

    pupil_trig_df['trigger_num'] = range(len(pupil_trig_df))

    # Plotting the right eye pupil data using Plotly if plot=True
    if plot:   
        # Defining the unique task sessions within the data file 
        unique_sessions = pupil_df['task_session'].unique()
        num_sessions = len(unique_sessions)

        colors = ['blue', 'green', 'red'] 

        # Create subplots, one for each task session
        fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=True,
                            subplot_titles=[f'Pupil diameter plus x and y coordinates of the right eye in task session {session}' for session in unique_sessions])

        # Add pupil traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = pupil_df[pupil_df['task_session'] == session]

            fig.add_trace(go.Scatter(x=session_data['tobii_time_rel'], y=session_data['RE_diameter'], mode='lines', line=dict(color=colors[0]), name='Right eye diameter'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['tobii_time_rel'], y=session_data['RE_x'], mode='lines', line=dict(color=colors[1]), name='Right eye x position'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['tobii_time_rel'], y=session_data['RE_y'], mode='lines', line=dict(color=colors[2]), name='Right eye y position'), row=i, col=1)

        # Update layout
        fig.update_layout(height=900*num_sessions, 
                          width=1200,
                          title_text=f'Right eye data across task sessions in condition {condition}',
                          xaxis_title='Relative Time',
                          yaxis_title='Diameter',
                          showlegend=False)
        
        # Show the figure
        fig.show()
    
    return pupil_df, pupil_trig_df


def convert_change_points_to_pupil(pupil_trig_df, change_point_indices, condition):
    """ 
    Function to convert identified change points from the task/behavioural data to corresponding time stamp within tobii .txt files. Stores data within dictionaries that can then be used to define epochs for the analysis of pupil diameter.

    Parameters:
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **

    Returns:
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.  
    """
    # Identify change point based on cumulative trigger number 
    tobii_by_change_points  = pupil_trig_df[pupil_trig_df['trigger_num'].isin(change_point_indices['index'])]

    # Creating dictionaries for storage of change point time and number 
    change_point_onsets = {}
    change_point_nums = {}

    # For loop that extracts identified change point times for each task session 
    for session, group in tobii_by_change_points.groupby('task_session'):
      # Extract the timeflow values for the current group (session)
      change_point_onset_values = group['trigger_time'].tolist()
    
      # Create a cumulative count list for the current group (session)
      change_point_num_values = list(range(1, len(group) + 1))

      change_point_onsets[session] = change_point_onset_values
      change_point_nums[session] = change_point_num_values

    # Checking the number of change points identified
    print(f"Condition {condition}:")
    for session, events in change_point_nums.items():
      print(f"In task session {session} there are {len(events)} identified change points") 

    return change_point_onsets, change_point_nums


def pupil_change_point_analysis(participant_str, list_subject_selected, condition, from_session, to_session, change_point_onsets, change_point_num, outlier_limit_std = 10, outlier_correct_average_amount = 10,  window_start_before = 1, window_end_after = 1.4, z_score = False, plot = True):
    """ 
    Function to preprocess and then compute epoched pupil size around a defined change point. Preprocessing and change point analysis taken from preprocess_eyedata script. Calculates the mean and standard deviation across each identified epoch for all task session.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - list_subject_selected: list, identifier generated from participant number
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - from_session: int, defining the number of the first task session within the condition
    - to_session: int, defining the number of the last task session within the condition
    - change_point_onsets: dict, contains identified change point timestamps. Keys are task session and values are time stamps within that task session. 
    - change_point_nums: dict, contains a cumulative count of the number of change points identified within each task session. Keys are task sessions and values are the cumulative count of the number of change points within that task session.
    - outlier_limit_std: int, defines the threshold of the standard deviation of the derivative when preprocessing the pupil data. Default = 10
    - outlier_correct_average_amount: int, the number of data points to the left and at the right of the artifact which will replace the value of the artifact by the value of the mean of the data around it. Default = 10.
    - window_start_before: int, value defining the start of the epoch window in time (seconds). Default = 1. 
    - window_end_after: int, value defining the end of the epoch window in time (seconds). Default = 1.4.
    - z_score: boolean, option to z_score data before computing epochs. Default = False.
    - plot: boolean, option to plot the mean and standard deviation across epochs using Plotly. Default = True. 

    Returns:
    - epoched_pupil_cp_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean and standard deviation across epochs and the timestamp. 

    """
     # Reset working directory
    base_dir = f'/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition}'
    os.chdir(base_dir)
    print(f"Current working directory: {os.getcwd()}")

    print(condition)

    # Running data through methods within class pupil taken from preprocess_eyedata, created by Florent, Maëva, Aziza and Yannaël. First initialise class
    data_tob = pupil()

    # List to store dataframes containing the epoched data from each task session
    epoched_data_list = []

    # Iterate through each task session file to analyse the pupil data
    for session in range(from_session, to_session +1):
        session_str = str(session).zfill(2)
        file_name = f"sub-{participant_str}_ses-{condition}_task-AdalearnVNS_run-{session_str}_eyetracker.txt"
    
        list_best_eye = pick_best_eye(file_name, list_subject_selected, from_session, to_session)

        for subject, i_subject in zip(list_subject_selected, range(len(list_subject_selected))):

            # Import the datafile
            data_tob.import_data(file_name,
                             events_warner='MSG', eye='right', 
                             col=[0, 1, 2],
                             col_name=['x_pos', 'y_pos', 'diameter'],
                             eyetracker='tobii')

            # Preprocess the data
            time, diameter, diameter_int, x_pos, x_pos_int, y_pos, y_pos_int = data_tob.preprocess(plot=False,    remove_outlier=True, outlier_limit_std=outlier_limit_std, outlier_correct_average_amount=outlier_correct_average_amount)
      
            # Epoch the data and apply baseline correction
            data_tob.epoch_tobii(change_point_onsets[session_str], variable='diameter_int', decimate=1, before=window_start_before, after=window_end_after, baseline=[-1, 0], conditions={'vns_num': change_point_num[session_str]}, z_score_data=z_score)

            # Check to ensure the correct number of epochs
            print("For participant n:", subject, " in session = ", session, "there are:", len(data_tob.epochs), "epochs")

            # Appending dataframes containing epoched data to list for each task session
            epoched_data = pd.DataFrame([epoch for epoch in data_tob.epochs])
            epoched_data_list.append(epoched_data)

    # Collating the timelocked epoched pupil data to calculate descriptive statistics for plotting
    epoched_pupil_cp_df = pd.concat(epoched_data_list, ignore_index=True).T
    numeric_data = epoched_pupil_cp_df.iloc[:, :]
    epoched_pupil_cp_df['mean'] = numeric_data.mean(axis=1)
    epoched_pupil_cp_df['std'] = numeric_data.std(axis=1)
    epoched_pupil_cp_df['time'] = data_tob.epochs_times
    
    # Plotting the mean (and std) response of pupil response timelocked to change points
    if plot:
        
        fig_epoch_avg = go.Figure()

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 1)'),
            name='Mean'
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'] + epoched_pupil_cp_df['std'],
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_trace(go.Scatter(
            x=epoched_pupil_cp_df['time'],
            y=epoched_pupil_cp_df['mean'] - epoched_pupil_cp_df['std'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.1)'),
            showlegend=False
        ))

        fig_epoch_avg.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig_epoch_avg.update_layout(
            title='Baseline corrected right pupil response epoched around change points across all task sessions (Mean +/- Std)',
            xaxis_title='Time (s)',
            yaxis_title='Pupil diameter'
        )       

        fig_epoch_avg.show()

    return epoched_pupil_cp_df