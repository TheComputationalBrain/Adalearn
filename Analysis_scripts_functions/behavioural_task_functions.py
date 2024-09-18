import pandas as pd
import numpy as np
import json
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

def load_behavioural_data(participant_str, condition, condition_task_sessions):
    """
    Function to load adalearn .json files. Sets the working directory to the relevant file, before loading and storing the data for each task session for a given condition. 
    
    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in the condition

    Returns:
    - adalearn_data: list, contains the dictionaries containing the data from the adalearn files for each task session. 
    """

    # Setting the working directory of the files
    print(os.getcwd())
    os.chdir(f'/home/aaron/Documents/Python/Ada_Learn/VNS Ada_Learn/Data/sub-{participant_str}/ses-{condition}/behav')
    print(os.getcwd())

    # Loading all adalearn task session data files
    adalearn_data = []

    # Iterate across all files and append data to storage list
    for session in condition_task_sessions:
    
        with open(f'sub-{participant_str}_ses-{condition}_task-AdalearnVNS_run-{session}_behav.json', 'rb') as ada_learn_file:
             parsed_json_con = json.load(ada_learn_file)
    
        adalearn_data.append(parsed_json_con)
    
    return adalearn_data

def create_behavioural_dataframe(behavioural_metric_list):
    """
    Function to unpack parsed_json adalearn files into a dataframe. Returned dataframe contains only selected variables deemed most relevant from the full .json file.  

    Parameters:
    - behavioural_metric_list: list, contains the dictionaries containing the data from the adalearn files for each task session. 

    Returns:
    - adalearn_df: DataFrame, behavioural task data across each task session stacked vertically (Note*: 'relative_time' calculated based on the presentation of the first outcome (rather than task session start time). This was chosen for easier translation to pupil/biopac trigger data) 
    """
    # Creating data frame to store the unpacked
    unpack_dfs = []

    # Iterate through each parsed JSON data to unpack data into a compiled data frame
    for parsed_json_file in behavioural_metric_list:
        session_df = pd.DataFrame()
        session_df["time"] = parsed_json_file['trialFrameTime']
        session_df["outcome"] = parsed_json_file['outcome']
        session_df["estimate"] = parsed_json_file['estimate']
        session_df["ideal_observer"] = parsed_json_file['idealObserverEstimate']
        session_df["P1"] = parsed_json_file['p1']
        session_df["change_point"] = parsed_json_file['didP1Change']
        session_df["task_session"] = parsed_json_file['taskSessionIdx']
        session_df["trial_number"] = parsed_json_file['trialIdx']
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
    adalearn_df = pd.concat(unpack_dfs, ignore_index=True)

    # Calculate first outcome presentation times for each task session
    first_outcome_times = adalearn_df.groupby('task_session')['time'].transform('first')

    # Subtract start times from 'time' column to create 'relative_time' column for each task session
    adalearn_df['relative_time'] = (adalearn_df['time'] - first_outcome_times) / 1000

    # adjust trial and task session index
    adalearn_df['task_session'] = adalearn_df['task_session'] + 1
    adalearn_df['trial_number'] = adalearn_df['trial_number'] + 1

    return adalearn_df

def behavioural_data_diagnostics(behavioural_df, condition_task_sessions):
    """
    Function to evaluate whether the behavioural data that should be in the dataframe is in fact there. Prints outcome of analysis. This should include:
    - number of task sessions: Does the number of task sessions match what should be expected?
    - number of trials per task session: Are there 75 outcome presentations per task session?    
    - number of change points per task session: Is there at least 2 change points per task session (2 because the first outcome of each task session is identified as a change point)?

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables. 
    - condition_task_sessions: list, contains string variables (set to 2 numbers) denoting the individual task sessions completed by the participant in the condition

    Returns:
    - Does not return an object. Prints outcomes of evaluation  
    """

    # Checking whether the right number of task sessions are in the data frame
    unique_task_sessions = behavioural_df['task_session'].unique()
    number_of_unique_task_sessions = len(unique_task_sessions)
    print(f'The task sessions in the data file are: {unique_task_sessions}')

    if number_of_unique_task_sessions == len(condition_task_sessions):
        print(f"There are the expected number of task sessions: {number_of_unique_task_sessions} / {len(condition_task_sessions)}")
    else:
        print(f"There is not the expected number of task sessions: {number_of_unique_task_sessions} / {len(condition_task_sessions)}")
    
    # Checking whether there are 75 outcome presentations per task session
    trial_count_per_session = behavioural_df.groupby('task_session')['trial_number'].count()

    for task_session, count in trial_count_per_session.items():
        if count == 75:
            print(f"Task session {task_session} has exactly 75 trials.")
        else:
            print(f"Task session {task_session} does not have 75 trials, it has {count} trials.")

    # Checking that there are sufficient change points within each task session data file
    change_point_count_per_session = behavioural_df.groupby('task_session')['change_point'].sum()
    print(f"Number of True values in 'change_point' for each task session:\n{change_point_count_per_session}")

    # Remove the True value associated with trial number 1 in each task session then group by 'task_session' and count the number of True values in 'change_point' again
    filtered_cp_df = behavioural_df[~((behavioural_df['trial_number'] == 1) & (behavioural_df['change_point'] == True))]

    filtered_change_point_count_per_session = filtered_cp_df.groupby('task_session')['change_point'].sum()
    print(f"\nNumber of True values in 'change_point' for each task session after removing True value for trial number 1:\n{filtered_change_point_count_per_session}")

def behavioural_change_point_analysis(behavioural_df):
    """
    Function to identify change points within each task session from the respective adalearn data file.
    Takes the boolean variable 'change_point' and:
        1. identifies how many change points are in the data frame as a whole. 
        2. identifies how many change points are in each task session. 
        3. identifies the index and relative time stamp (relative to beginning of each respective task session) of each change point.
    
    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables. 
        
    Returns:
    - num_change_points: int, the total number of change points in the DataFrame.
    - change_points_per_session: pd.Series, the number of change points in each task session.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point)
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
        trial_number = group.loc[group['change_point'], 'trial_number'].tolist()
        relative_times = group.loc[group['change_point'], 'relative_time'].tolist()
        
        # Create a DataFrame for the current group and append to the list
        change_point_indices_df = pd.DataFrame({'index': indices, 'trial_number': trial_number,'relative_time': relative_times})
        change_point_indices_df['task_session'] = task_session  # Add Task_session for identification
        change_point_indices_list.append(change_point_indices_df)
    
    # Concatenate all the DataFrames in the list into a single DataFrame
    change_point_indices = pd.concat(change_point_indices_list).reset_index(drop=True)

    # Identifying the change point numbers that are not associated with the start of a new task session
    change_point_indices = change_point_indices[change_point_indices['trial_number'] > 1]
    
    return num_change_points, change_points_per_session, change_point_indices

def plot_behavioural_data(behavioural_df, participant, condition, estimate_v_io_plot = False):
    """ 
    Function to plot the behavioural, ideal observer, hidden estimates and outcome observations from each task session. Optionally, can plot subjective estimates against ideal observer across all task sessions (estimate_v_io_plot; default is False).

    Parameters:
    - participant: string, identifier of participant number
    - condition: string, identifier of condition order (set to 2 numbers)
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables.
        - estimate_v_io_plot: boolean,, if True, will plot scatter graph of subjective estimates against ideal observer for all task sessions (plotly enables you to view individual task sessions by clicking on the respective label in the legend to activate/deactivate that series) 

    Returns:
    - Figure containing subplots opened within a browser.
    """
    # Defining the unique task sessions within the data file 
    unique_sessions = behavioural_df['task_session'].unique()
    num_sessions = len(unique_sessions)

    # Plotting behavioural responses for each task session 
    # Create subplots, one for each task session
    fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=True,
                        subplot_titles=[f'Behavioural estimates, ideal observer and task characteristics in task session {session}' for session in unique_sessions])

    # Set of colours to be used to designate for each variable
    colors = ['blue', 'green', 'red', 'purple'] 

    # Add learning rate traces to each subplot
    for i, session in enumerate(unique_sessions, start=1):
        session_data = behavioural_df[behavioural_df['task_session'] == session]

        # Scatter for P1 
        fig.add_trace(
            go.Scatter(x=session_data['relative_time'], y=session_data['P1'], mode='lines', line=dict(color=colors[0]), name=f'Hidden proportion of blue'), row=i, col=1)

        # Scatter for participant estimate
        fig.add_trace(
            go.Scatter(x=session_data['relative_time'], y=session_data['estimate'], mode='lines', line=dict(color=colors[1]), name=f'Participant estimate of proportion'), row=i, col=1)

        # Scatter for ideal observer estimate
        fig.add_trace(
            go.Scatter(x=session_data['relative_time'], y=session_data['ideal_observer'], mode='lines', line=dict(color=colors[2]), name=f'Ideal observer estimate of proportion'), row=i, col=1)

        # Scatter for outcome presentation with colour based on 'outcome'
        colours_adalearn = ['blue' if val == 1 else 'yellow' for val in session_data['outcome']]
        fig.add_trace(
            go.Scatter(x=session_data['relative_time'], y=[0]*len(session_data['outcome']), mode='markers', marker=dict(color=colours_adalearn), name='Outcomes'), row=i, col=1)

        # Set y-axis range
        fig.update_yaxes(range=[-0.09, 1], row=i, col=1)

    # Update overall layout
    fig.update_layout(height=900*num_sessions, width=1200, title_text=f"Behavioural estimates, ideal observer and task characteristics in condition {condition} for participant {participant}",
                      showlegend=False, title_x=0.5) 

    # Add axis titles
    fig.update_xaxes(title_text='Time (s)', row=num_sessions, col=1)

    for i in range(1, num_sessions+1):
        fig.update_yaxes(title_text='Proportion blue', row=i, col=1)

    # Show plot
    fig.show()

    # Plotting subjective estimates versus ideal observer
    if estimate_v_io_plot:
        df= behavioural_df.copy()
        df['task_session'] = df['task_session'].astype('category')

        fig = px.scatter(df, x="ideal_observer", y="estimate", color="task_session", symbol="task_session")

        fig.update_layout(
        title_text="Ideal observer by subjective estimate for each task session",
        showlegend=True,
        xaxis_title="Ideal Observer",
        yaxis_title="Estimate"
        )

        fig.show()

def calculate_behavioural_learning_rate(behavioural_df):
    """ 
    Function to calculate the behavioural learning rate for each outcome presentation. Learning rate (αt) is derived from the delta rule (Bt+1 = Bt + αt * t ): 

                        αt = (vt - vt-1) / (xt - vt-1)

    where vt is the current estimate, vt-1 is the previous estimate and xt is the outcome (here, either 1 for blue and 0 for yellow). Returns dataframe with new learning rate variable created. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables. 

    Returns:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable.     
    """
    # Create learning rate variable within dataframe
    behavioural_df['learning_rate'] = float('nan')

    # Defining learning rate equation
    def compute_learning_rate(group):
        for i in range(1, len(group)):
            vt = group.iloc[i]['estimate']
            vt_minus_1 = group.iloc[i-1]['estimate']
            xt = group.iloc[i]['outcome']
        
            group.at[group.index[i], 'learning_rate'] = (vt - vt_minus_1) / (xt - vt_minus_1)
            
        return group    

    # Implementing learning rate equation
    behavioural_df = behavioural_df.groupby('task_session').apply(compute_learning_rate).reset_index(drop=True) 

    return behavioural_df

def plotting_behavioural_learning_rate(behavioural_df, condition):
    """ 
    Function to plot the behavioural learning rate for each task session. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable.
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **

    Returns:
    - Figure containing subplots opened within a browser.
    """

    # Create subplots, one for each task session
    unique_sessions = behavioural_df['task_session'].unique()
    num_sessions = len(unique_sessions)

    fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=True,
                    subplot_titles=[f'Beahvioural learning rate in task session {session}' for session in unique_sessions])

    # Add learning rate traces to each subplot
    for i, session in enumerate(unique_sessions, start=1):
        session_data = behavioural_df[behavioural_df['task_session'] == session]
        fig.add_trace(
            go.Scatter(x=session_data['trial_number'], y=session_data['learning_rate'], mode='lines+markers', name=f'Beahvioural learning rate in task session {session}'), row=i, col=1)

    # Update layout
    fig.update_layout(height=300*num_sessions, width=800, title_text=f"Behavioural learning rate across task sessions in condition {condition}",
                  showlegend=False, title_x=0.5)

    # Add axis titles
    fig.update_xaxes(title_text='Trial Number', row=num_sessions, col=1)

    for i in range(1, num_sessions+1):
        fig.update_yaxes(title_text='Learning Rate', row=i, col=1)

    # Show plot
    fig.show()

def learning_rate_by_change_point(behavioural_df, change_point_indices, participant, condition, pre=2, post=10):
    """ 
    Function to plot the average response around change points. Uses identified change point indices to derive learning rate dynamics before (2 outcomes) and after (10 outcomes). Stores responses across all task sessions and then plots the average plus standard deviation in a web browser. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', 
      containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
    - participant: string, identifier of participant number
    - condition: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - pre: integer, number of outcomes before the change point. Predefined to incorporate in plotting the data.
    - post: integer, number of outcomes after the change point. Predefined to incorporate in plotting the data.

    Returns:
    - Figure containing subplots opened within a browser.
    - filtered_learning_rate_by_cp: pd.DataFrame, contains behavioural data within defined bounds around change points.
    - averaged_learning_rate_by_cp: pd.DataFrame, contains calculated mean, std, count and sem of learning rate at each relative trial position for each identified change point. 
    """
  
    # Creating dataframe to store data around defined   
    expanded_change_points = []

    for idx, row in change_point_indices.iterrows():
        task_session = row['task_session']
        trial_number = row['trial_number']

        for i in range(-pre, post+1):  # Predefined to be 2 trials before to 10 trials after identified change points. Can be changed to suit. 
            expanded_change_points.append({
                'task_session': task_session,
                'trial_number': trial_number,
                'relative_position': i,
                'trial_number_rel': trial_number + i
            })

    expanded_change_points_df = pd.DataFrame(expanded_change_points)
    expanded_change_points_df['relative_position'] = expanded_change_points_df['relative_position'] +1                      

    # Filter main behavioural dataframe to only contain values in defined range around change point  
    filtered_learning_rate_by_cp = pd.merge(expanded_change_points_df, behavioural_df, 
                                            left_on=['task_session', 'trial_number_rel'], 
                                            right_on=['task_session', 'trial_number'], 
                                            how='left', suffixes=('', '_y'))

    # Drop unnecessary columns
    filtered_learning_rate_by_cp.drop(columns=['relative_time', 'trial_number_y'], inplace=True)

    # Calculate the average, standard deviation and count of the behavioural learning rate across the different relative positions to the change point
    averaged_learning_rate_by_cp = filtered_learning_rate_by_cp.groupby('relative_position')['learning_rate'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate SEM
    averaged_learning_rate_by_cp['sem'] = averaged_learning_rate_by_cp['std'] / np.sqrt(averaged_learning_rate_by_cp['count'])

    # Convert 'relative_position' to a numeric range
    averaged_learning_rate_by_cp['relative_position'] = averaged_learning_rate_by_cp['relative_position'].astype(int)
    averaged_learning_rate_by_cp = averaged_learning_rate_by_cp.sort_values('relative_position')

    # Plotting the behavioural learning rate using subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Individual Learning Rates", "Average Learning Rate"))

    num_traces = 0

    # Add individual learning rates
    for idx, row in change_point_indices.iterrows():
        session_data = filtered_learning_rate_by_cp[(filtered_learning_rate_by_cp['task_session'] == row['task_session']) & 
                                                    (filtered_learning_rate_by_cp['trial_number'] == row['trial_number'])]

        if len(session_data) == (pre + post + 1):
            num_traces += 1
            fig.add_trace(go.Scatter(
                x=session_data['relative_position'],
                y=session_data['learning_rate'],
                mode='lines',
                name=f'Trace {num_traces}',
                line=dict(width=1),
                opacity=0.5,
                showlegend = False
            ), row=1, col=1)

    print(f"Total number of traces added to the individual plot: {num_traces}")

    # Add mean learning rate 
    fig.add_trace(go.Scatter(
        x=averaged_learning_rate_by_cp['relative_position'],
        y=averaged_learning_rate_by_cp['mean'],
        mode='lines',
        name='Mean Learning Rate',
        line=dict(color='blue')
    ), row=2, col=1)

    # Add standard error of the mean as a shaded area around the mean
    fig.add_trace(go.Scatter(
        x=list(averaged_learning_rate_by_cp['relative_position']) + list(averaged_learning_rate_by_cp['relative_position'][::-1]),
        y=list(averaged_learning_rate_by_cp['mean'] + averaged_learning_rate_by_cp['sem']) + list((averaged_learning_rate_by_cp['mean'] - averaged_learning_rate_by_cp['sem'])[::-1]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=2, col=1)

    # Add dashed line at change point
    for i in [1, 2]:
        fig.add_shape(type="line",
            x0=0.5, y0=-5, x1=0.5, y1=5,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
            row=i, col=1
        )

    # Update layout
    fig.update_layout(
        title=f"Behavioural Learning Rate Relative to Change Points for Participant {participant} in Condition {condition}",
        xaxis2_title="Trial Number After Change Point",
        yaxis1_title="Learning Rate (Individual)",
        yaxis2_title="Learning Rate (Average)",
        xaxis1=dict(
            tickmode='linear',
            dtick=1,
            tickvals=[i for i in range(1, post + 1)]
        ),
        xaxis2=dict(
            tickmode='linear',
            dtick=1,
            tickvals=[i for i in range(1, post + 1)]
        ),
        yaxis2=dict(
            range=[averaged_learning_rate_by_cp['mean'].min() - averaged_learning_rate_by_cp['sem'].max(), averaged_learning_rate_by_cp['mean'].max() + averaged_learning_rate_by_cp['sem'].max()]
        ),
    )

    fig.show()

    return filtered_learning_rate_by_cp, averaged_learning_rate_by_cp