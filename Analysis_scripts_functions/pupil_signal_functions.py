import pandas as pd
import numpy as np
import os
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'


def load_pupil_data(participant_str, condition, condition_task_sessions, plot = False):
    """ 
    Function to load and read .txt file from Tobii eyetracker. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. Also separates and stores trigger information taken from .txt file. There is an option to plot the right eye data for each task session. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in that condition
    - plot: boolean, if True, creates subplots the right eye data for each task session across time using Plotly.

    Returns:
    - pupil_df: DataFrame, contains the pupil data recorded from each task session
    - pupil_trig_df: DataFrame, contains the trigger information (i.e. time, outcome) for each task session  
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

    # Initialise lists for the stroing of trig times inserted into Tobii eye tracker
    trig_time_list = []
    task_sess_trig_list = []

    # Read in .txt file as lines and then for each line, separate the data and store it in an apporpriate list.
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

    # Plotting the right eye pupil data using Plotly if plot=True
    if plot:
        grouped_df = pupil_df.groupby('task_session')

        for group, data in grouped_df:
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=data['tobii_time_rel'], y=data['RE_diameter'], mode='lines', name='Right eye diameter', yaxis='y', showlegend=True))

            fig.add_trace(go.Scatter(x=data['tobii_time_rel'], y=data['RE_x'], mode='lines', name='Right eye x position', yaxis='y2', showlegend=True))

            fig.add_trace(go.Scatter(x=data['tobii_time_rel'], y=data['RE_y'], mode='lines', name='Right eye y position', yaxis='y2', showlegend=True))

            fig.update_layout(title=f'Right eye data for task session {group}',
                          xaxis_title='Relative Time',
                          yaxis_title='Diameter',
                          yaxis=dict(title='Diameter', showgrid=True, zeroline=False),
                          yaxis2=dict(title='Position', overlaying='y', side='right', showgrid=True, zeroline=False))
            
            fig.show()
    
    return pupil_df, pupil_trig_df