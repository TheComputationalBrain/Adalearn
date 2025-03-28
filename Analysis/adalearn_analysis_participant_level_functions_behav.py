import pandas as pd
import numpy as np
import json
import os
from scipy.stats import zscore
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'



def load_behavioural_data(participant_str, condition_str, directory_path, condition_task_sessions, plot_behav_data = False):
    """
    Function to load adalearn .json files. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. Then unpacks parsed_json adalearn files into a dataframe. Returned dataframe contains only selected variables deemed most relevant from the full .json file. Diagnostics identify whether what is expected in the dataframe is in fact there. Options to plot the the behavioural data (participants estimation, ideal observer, true hidden estimates and outcome observations) from each task session, and also the participants estimates against ideal observer across all task sessions.  

    Diagnostics evaluates the following:
    - number of task sessions: Does the number of task sessions match what should be expected?
    - number of trials per task session: Are there 75 outcome presentations per task session?    
    - number of change points per task session: Is there at least 2 change points per task session (2 because the first outcome of each task session is identified as a change point)?
    
    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, path to the participant level directory
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in the condition
    - plot_behav_data: boolean, if true, will plot the raw behavioural data as subplots for each task session (default = False)

    Returns:
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. 
    - computational_var_df: pd.DataFrame, dataframe containing the estimates and select model derived outcomes. To be used to house additional computational variables.
    """
    # Loading all adalearn task session data files
    adalearn_data = []

    # Define possible task names
    task_names = ["AdalearnVNS", "AdalearnPupil"]

    # Iterate across all files and append data to storage list
    for session in condition_task_sessions:
        # Check for the correct file dynamically
        for task in task_names:
            file_path = f'{directory_path}/ses-{condition_str}/behav/sub-{participant_str}_ses-{condition_str}_task-{task}_run-{session}_behav.json'
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as ada_learn_file:
                    parsed_json_con = json.load(ada_learn_file)
                adalearn_data.append(parsed_json_con)
                break  # Exit loop once the correct file is found
        else:
            # This block runs only if no file is found for this session
            print(f"No file found for session {session}")
    
    # Creating data frame to store the unpacked data
    unpack_dfs = []

    # Iterate through each parsed JSON data to unpack data into a compiled data frame
    for parsed_json_file in adalearn_data:
        session_df = pd.DataFrame()
        session_df["time"] = parsed_json_file['trialFrameTime']
        session_df["outcome"] = parsed_json_file['outcome']
        session_df["estimate"] = parsed_json_file['estimate']
        session_df["ideal_observer"] = parsed_json_file['idealObserverEstimate']
        session_df["P1"] = parsed_json_file['p1']
        session_df["change_point"] = parsed_json_file['didP1Change']
        session_df["task_session"] = parsed_json_file['taskSessionIdx']
        session_df["outcome_num"] = parsed_json_file['trialIdx']
        session_df["sequence_id"] = parsed_json_file['sequenceIdx']
        session_df["run_id"] = parsed_json_file['runId']
        session_df["score"] = parsed_json_file['scoreFrac']
        session_df["change_point_probability"] = parsed_json_file['sequencePC']
        session_df["P1min"] = parsed_json_file['sequenceP1Min']
        session_df["P1max"] = parsed_json_file['sequenceP1Max']
        session_df["num_trials"] = parsed_json_file['nTrials']
        session_df["session_start_time"] = parsed_json_file['startTime']
        session_df["session_end_time"] = parsed_json_file['endTime']

        # Append the session DataFrame to the list 
        unpack_dfs.append(session_df)

    # Concatenate all unpacked DataFrames in the list vertically
    behavioural_df = pd.concat(unpack_dfs, ignore_index=True)

    # There was an issue with one participant (sub-09) where a new experiment had to be started during the second half. This means that the task session ids are duplicated between the first and second half of the experiment. This code fixes the data frame to make sure the task session ids are unique.
    duplicated_sessions = behavioural_df.duplicated(subset=['task_session'], keep=False)

    if behavioural_df[duplicated_sessions]['run_id'].nunique() > 1:
        print('Duplicated task sessions found in dataframe creation. Processing to rectify...')

        second_run_id = behavioural_df['run_id'].unique()[-1]

        second_run_df = behavioural_df[behavioural_df['run_id'] == second_run_id]

        duplicated_sessions_second_run = second_run_df['task_session'].duplicated(keep=False)

        behavioural_df.loc[second_run_df.index[duplicated_sessions_second_run], 'task_session'] = behavioural_df.loc[second_run_df.index[duplicated_sessions_second_run], 'task_session'] + 11

    # Calculate first outcome presentation times for each task session
    first_outcome_times = behavioural_df.groupby('task_session')['time'].transform('first')

    # Subtract start times from 'time' column to create 'rel_time' column for each task session
    behavioural_df['rel_time'] = (behavioural_df['time'] - first_outcome_times) / 1000

    # adjust  task session index
    behavioural_df['task_session'] = behavioural_df['task_session'] + 1

    # Behavioural data diagnostics
    # Checking whether the right number of task sessions are in the data frame
    unique_task_sessions = behavioural_df['task_session'].unique()
    number_of_unique_task_sessions = len(unique_task_sessions)

    if number_of_unique_task_sessions == len(condition_task_sessions):
        print(f"There are the expected number of task sessions: {number_of_unique_task_sessions} / {len(condition_task_sessions)}")
    else:
        print(f"There is not the expected number of task sessions: {number_of_unique_task_sessions} / {len(condition_task_sessions)}")
    
    # Checking whether there are 75 outcome presentations per task session
    trial_count_per_session = behavioural_df.groupby('task_session')['outcome_num'].count()

    for task_session, count in trial_count_per_session.items():
        if count == 75:
            print(f"Task session {task_session} has exactly 75 outcome stimuli.")
        else:
            print(f"Task session {task_session} does not have 75 outcome stimuli, it has {count} outcomess.")

    # Creating a computational variable data frame for subsequent model analysis
    computational_var_df = behavioural_df[['outcome', 'estimate', 'ideal_observer', 'outcome_num', 'task_session', 'time', 'rel_time']].copy() 

    # Add session-specific cumulative count
    computational_var_df['cum_outcome_num'] = range(len(computational_var_df))

    # Option to plot the raw behavioural data across each task session separately
    if plot_behav_data:
        # Defining the unique task sessions within the data file 
        unique_sessions = behavioural_df['task_session'].unique()
        num_sessions = len(unique_sessions)

        # Plotting behavioural responses for each task session 
        # Create subplots, one for each task session
        fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=True,
                        subplot_titles=[f'Behavioural estimates (green), ideal observer (red) and hidden task structure (blue) in task session {session}' for session in unique_sessions])

        # Set of colours to be used to designate for each variable
        colors = ['blue', 'green', 'red', 'purple'] 

        # Add learning rate traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = behavioural_df[behavioural_df['task_session'] == session]

            # Scatter for P1 
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=session_data['P1'], mode='lines', line=dict(color=colors[0]), name=f'Hidden proportion of blue'), row=i, col=1)

            # Scatter for participant estimate
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=session_data['estimate'], mode='lines', line=dict(color=colors[1]), name=f'Participant estimate of proportion'), row=i, col=1)

            # Scatter for ideal observer estimate
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=session_data['ideal_observer'], mode='lines', line=dict(color=colors[2]), name=f'Ideal observer estimate of proportion'), row=i, col=1)

            # Scatter for outcome presentation with colour based on 'outcome'
            colours_adalearn = ['blue' if val == 1 else 'yellow' for val in session_data['outcome']]
        
            fig.add_trace(
                go.Scatter(x=session_data['rel_time'], y=[0]*len(session_data['outcome']), mode='markers', marker=dict(color=colours_adalearn), name='Outcomes'), row=i, col=1)

            # Set y-axis range
            fig.update_yaxes(range=[-0.09, 1], row=i, col=1)

        # Update overall layout
        fig.update_layout(height=900*num_sessions, width=1200, title_text=f"Behavioural estimates (green), ideal observer (red) and hidden task structure (blue) in condition {condition_str} for participant {participant_str}",
                      showlegend=False, title_x=0.5) 

        fig.update_xaxes(title_text='Time (s)', row=num_sessions, col=1)

        for i in range(1, num_sessions+1):
            fig.update_yaxes(title_text='Proportion blue', row=i, col=1)

       # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_behav-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()

    return behavioural_df, computational_var_df 



def behavioural_change_point_analysis(participant_str, condition_str, directory_path, behavioural_df, plot_histogram_zscores = False):
    """
    Function to identify change points within each task session from the respective adalearn data file.
    Takes the boolean variable 'change_point' and:
        1. identifies how many change points are in the data frame as a whole. 
        2. identifies how many change points are in each task session. 
        3. identifies the index and relative time stamp (relative to beginning of each respective task session) of each change point.
        4. identifies the hidden value associated with each change point and uses this to categorise the 'obviousness' of the change, based on the criteria defined by Jepma et al (2016). 
    
    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: directory_path: Path, path to the participant level directory
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables.
    - plot_histogram_zscores: boolean, if true, will plot a histogram of the z-score differences between change points (default = False)
        
    Returns:
    - num_change_points: int, the total number of change points in the DataFrame.
    - change_points_per_session: pd.Series, the number of change points in each task session.
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session, and z-score used to categorises the obvious of the change point ('change_point_cat'). (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - behavioural_df: pd.DataFrame, contains a new column 'outcome_dict_to_cp' that specifies the distance of each outcome to the last change point within a task session.
    """
    # Calculate the number of change points
    num_change_points = (behavioural_df['change_point'] == True).sum()
    
    # Calculate the change points per task session
    change_points_per_session = behavioural_df.groupby('task_session')['change_point'].sum()
    
    # Initialize a list to store DataFrames of indices, trial number and relative times
    change_point_indices_list = []

    # Iterate over each group of Task_session
    for task_session, group in behavioural_df.groupby('task_session'):
        indices = group.index[group['change_point']].tolist()
        outcome_num = group.loc[group['change_point'], 'outcome_num'].tolist()
        relative_times = group.loc[group['change_point'], 'rel_time'].tolist()
        hidden_values = group.loc[group['change_point'], 'P1'].tolist()
        
        # Create a DataFrame for the current group and append to the list
        change_point_indices_df = pd.DataFrame({'cum_outcome_num': indices, 'outcome_num': outcome_num,'rel_time': relative_times, 'hidden_value': hidden_values})
        change_point_indices_df['task_session'] = task_session  # Add Task_session for identification
        change_point_indices_list.append(change_point_indices_df)
    
    # Concatenate all the DataFrames in the list into a single DataFrame
    change_point_indices = pd.concat(change_point_indices_list).reset_index(drop=True)

    # Categorising change points into 'non-obvious' and 'obvious' change points based on Jepma et al (2016) criteria
    change_point_indices['z_score_P1'] =  zscore(change_point_indices['hidden_value'])
   
    change_point_indices['z_score_diff'] = change_point_indices['z_score_P1'].diff()

    change_point_indices['change_point_cat'] = None

    change_point_indices.loc[(change_point_indices['z_score_diff'] < 2) & (change_point_indices['z_score_diff'] > -2), 'change_point_cat'] = 'non-obvious'
    change_point_indices.loc[(change_point_indices['z_score_diff'] > 2) | (change_point_indices['z_score_diff'] < -2), 'change_point_cat'] = 'obvious'

    obvious_range_indices = change_point_indices[change_point_indices['change_point_cat'] == 'obvious'].index
    non_obvious_range_indices = change_point_indices[change_point_indices['change_point_cat'] == 'non-obvious'].index

    # Calculating the distance between change points
    behavioural_df = distance_to_change_point(behavioural_df=behavioural_df, change_point_indices=change_point_indices)

    # Identifying the change point numbers that are not associated with the start of a new task session
    change_point_indices = change_point_indices[change_point_indices['outcome_num'] > 1]

    # Create a running count of the number of change points in a condition
    change_point_indices['change_point_count'] = range(1, len(change_point_indices) + 1)
    
    # Print the number of change points identified for each task session 
    print(f"\nIn condition {condition_str} there were {change_point_indices['change_point_count'].max()} change points across task sessions.")
    print(f"Number of change points in each task session:\n{change_point_indices.groupby('task_session')['outcome_num'].count()}")
    value_counts = change_point_indices['change_point_cat'].value_counts()
    print(f"Number of non-obvious change points: {value_counts['non-obvious']}")
    print(f"Number of obvious change points: {value_counts['obvious']}")
    
    # Option to plot the histogram of the z-score differences between change points
    if plot_histogram_zscores:
        fig = px.histogram(change_point_indices, x='z_score_diff', color='change_point_cat', nbins=20, title=f"Histogram of consecutive differences in change point outcome value z-scores for Participant {participant_str} in condition {condition_str}. Colours denote categorisation of 'obvious' and 'non-obvious' change points", labels={'x': 'Z-score difference', 'y': 'Frequency'})

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_histogram-of-z-score-difference-in-change-points.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    return num_change_points, change_points_per_session, change_point_indices, behavioural_df



def behavioural_performance_analysis(participant_str, condition_str, directory_path, behavioural_df, plot_performance):
    """
    Function to compute primary perforamnce metric (the absolute error between the participant's estimate and the true hidden value) for each task session.

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: directory_path: Path, path to the participant level directory
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables.
    - plot_performance: boolean, if true, will plot the average error and the standard deviation in the error for each task session (default = False)

    Returns:
    - mean_abs_error_df: pd.DataFrame, contains the computed mean and standard deviation of the absolute error for each task session.
    """
    abs_error_list = []

    for task_session, row in behavioural_df.groupby('task_session'):
        performance_score = abs(row['estimate'] - row['P1'])
        mean_abs_error = np.mean(performance_score)
        std_abs_error = np.std(performance_score)

        abs_error_df = pd.DataFrame({
            'mean_abs_error': [mean_abs_error], 
            'std_abs_error': [std_abs_error],
            'score': np.mean(row['score']),   
            'task_session': [task_session]                  
        })

        abs_error_list.append(abs_error_df) 

    mean_abs_error_df = pd.concat(abs_error_list, ignore_index=True)

    if plot_performance:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=mean_abs_error_df['task_session'], y=mean_abs_error_df['mean_abs_error'], mode='lines+markers', name='Mean absolute error'))

        fig.add_trace(go.Scatter(x=mean_abs_error_df['task_session'],
            y=mean_abs_error_df['mean_abs_error'] + mean_abs_error_df['std_abs_error'],
            mode='lines',
            line=dict(color='rgba(0, 0, 250, 0.1)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(x=mean_abs_error_df['task_session'],
            y=mean_abs_error_df['mean_abs_error'] - mean_abs_error_df['std_abs_error'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 250, 0.2)',
            line=dict(color='rgba(0, 0, 250, 0.1)'),
            showlegend=False
        ))

        fig.update_layout(title_text=f"Absolute error (and the variation in the absolute error) between subjective estimates and hidden values within each task session for participant {participant_str} in condition {condition_str}",
                          xaxis=dict(
                              tickmode='array',
                              tickvals=mean_abs_error_df['task_session'].unique(),
                              ticktext=[str(s) for s in mean_abs_error_df['task_session'].unique()]
                          ),
                          xaxis_title='Task session',
                          yaxis_title='Absolute error',
                          height=950,
                          width=1800,
                          showlegend=False,
                          margin=dict(l=50, r=50, t=50, b=50)
                          )

       # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_absolute-error-across-task-sessions.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()

    return mean_abs_error_df



def distance_to_change_point(behavioural_df, change_point_indices):
    """
    Funtion to define the distance each outcome is to the last change point (within a task session). 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables.
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session.

    Returns:
    - behavioural_df: pd.DataFrame, contains a new column 'outcome_dict_to_cp' that specifies the distance of each outcome to the last change point within a task session.
    """
    cp_grouped = change_point_indices.groupby('task_session')

    for task_session, group_data in cp_grouped:
        print(task_session)
        
        task_session_outcome_trials = behavioural_df[behavioural_df["task_session"] == task_session]
        n_trials = task_session_outcome_trials.shape[0]  

        # Initialize an array for distance to change points
        dist_to_cp = np.full(n_trials, np.iinfo(np.int32).max, dtype=int)

        # Get the array of change point indices
        cps = group_data['outcome_num'].values  
        n_cps = cps.shape[0] 

        for i_cp in range(n_cps):
            last_cp = cps[i_cp]
            if i_cp > 0:
                print(f"Change point {i_cp}: Distance from last change point: {last_cp}")

            if i_cp < (n_cps - 1):  
                next_cp = cps[i_cp + 1] 
                
                dist_to_last = np.arange(next_cp - last_cp)   
                dist_to_cp[last_cp:next_cp] = dist_to_last   

            else:
                dist_to_last = np.arange(n_trials - last_cp)   
                dist_to_cp[last_cp:n_trials] = dist_to_last  



        behavioural_df.loc[behavioural_df["task_session"] == task_session, "outcome_dist_to_cp"] = dist_to_cp


    return behavioural_df