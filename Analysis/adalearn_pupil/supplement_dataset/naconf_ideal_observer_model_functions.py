import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import base_dir, directory_path, selected_participant, task_sessions

sys.path.append('/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference
from model_learner import ideal_learner_prob_estimates_from_outcomes, ideal_learner_prob_confidence_from_outcomes, ideal_learner_prob_cpp



def load_task_behavioural_data(directory_path, selected_participant, task_sessions):
    """
    Function to load the task behavioural data from the .csv files for the naconf study. Creates a pandas dataframe with the extracted data for each session for a given participant.

    Parameters:
    - directory_path: str, path to the directory containing the data files
    - selected_participant: str, participant identifier (single digits for 1-9, double digits for 10-99)
    - task_sessions: list, list of session numbers to load data for

    Returns:
    - task_session_df: pandas DataFrame, concatenated dataframe containing the task data for the specified participant and sessions
    """
    task_session_df = []
    # Load the data
    for session in task_sessions:
        df = pd.read_csv(Path(f'{directory_path}/ProbLearnTask_{selected_participant}_sess_{session}.csv'), skiprows=1)

        # Remove rows with NaN in 'trial_ID'
        df = df[~df['trial_ID'].isna()]

        task_session_df.append(df)
    
    # Concatenate all dataframes
    task_session_df = pd.concat(task_session_df, ignore_index=True)

    return task_session_df



def load_constants(directory_path, task_sessions):
    """
    Function to load the constants used in the task across each session. These are read from a pickle file and stored in a dataframe.

    Parameters:
    - directory_path: str, path to the directory containing the data files
    - task_sessions: list, list of session numbers to load data for

    Returns:
    - constants: dict, dictionary containing the constants
    """
    constants_list = []
    # Load the constants
    for session in task_sessions:
        with open(Path(f'{directory_path}/constants_session_{session}.pickle'), 'rb') as f:
            data = pickle.load(f)
            constants_df = pd.DataFrame([data])
            constants_list.append(constants_df)

    # Concatenate all constants dataframes
    constants = pd.concat(constants_list, ignore_index=True)

    # Extracting task parameters
    p_c_values = constants['pJump'].values
    if np.all(p_c_values == p_c_values[0]):
         print('pJump is constant across sessions')
         P_C = p_c_values[0]
    else:
         print('pJump is not constant across sessions')
         P_C = None

    p_min_values = constants['pMin'].values
    if np.all(p_min_values == p_min_values[0]):
         print('pMin is constant across sessions')
         P_MIN = p_min_values[0]
    else:
         print('pMin is not constant across sessions')
         P_MIN = None

    p_max_values = constants['pMax'].values
    if np.all(p_max_values == p_max_values[0]):
         print('pMax is constant across sessions')
         P_MAX = p_max_values[0]
    else:
         print('pMax is not constant across sessions')
         P_MAX = None
        
    return constants, P_C, P_MIN, P_MAX



def change_point_analysis(participant_str, directory_path, task_session_df,):
    """
    Function to identify change points within each task session 
    Creates the boolean variable 'change_point' and:
        1. identifies how many change points are in the data frame as a whole. 
        2. identifies how many change points are in each task session. 
        3. identifies the index and relative time stamp (relative to beginning of each respective task session) of each change point.
    
    Parameters:
    - participant_str: string, participant identifier (single digits for 1-9, double digits for 10-99)
    - directory_path: directory_path: Path, path to the participant level directory
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables.

    Returns:
    - num_change_points: int, the total number of change points in the DataFrame.
    - change_points_per_session: pd.Series, the number of change points in each task session.
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session, and z-score used to categorises the obvious of the change point ('change_point_cat'). (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - behavioural_df: pd.DataFrame, contains a new column 'outcome_dict_to_cp' that specifies the distance of each outcome to the last change point within a task session.
    """
    # Initialize change point column 
    task_session_df['change_point'] = False

    # Loop over each session
    for session in task_session_df['session'].unique():
        mask = task_session_df['session'] == session
        data = task_session_df.loc[mask]

        # Identify change points: when p1 is the same as the previous value
        change_points = data['p1'] != data['p1'].shift(1)

        # Assign back to the original dataframe using .loc
        task_session_df.loc[mask, 'change_point'] = change_points.fillna(False)

    # Calculate the number of change points
    num_change_points = (task_session_df['change_point'] == True).sum()
    
    # Calculate the change points per task session
    change_points_per_session = task_session_df.groupby('session')['change_point'].sum()
    
    # Initialize a list to store DataFrames of indices, trial number and relative times
    change_point_indices_list = []

    # Iterate over each group of Task_session
    for task_session, group in task_session_df.groupby('session'):
        indices = group.index[group['change_point']].tolist()
        outcome_num = group.loc[group['change_point'], 'trial_ID'].tolist()
        hidden_values = group.loc[group['change_point'], 'p1'].tolist()
        
        # Create a DataFrame for the current group and append to the list
        change_point_indices_df = pd.DataFrame({
            'cum_outcome_num': indices, 
            'outcome_num': outcome_num,
            'hidden_value': hidden_values}
            )
        
        change_point_indices_df['session'] = task_session  # Add Task_session for identification
        change_point_indices_list.append(change_point_indices_df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    change_point_indices = pd.concat(change_point_indices_list).reset_index(drop=True)
    
    # Identifying the change point numbers that are not associated with the start of a new task session
    change_point_indices = change_point_indices[change_point_indices['outcome_num'] > 1]

    # Create a running count of the number of change points in a condition
    change_point_indices['change_point_count'] = range(1, len(change_point_indices) + 1)
    
    # Print the number of change points identified for each task session 
    print(f"\nThere were {change_point_indices['change_point_count'].max()} change points across task sessions.")
    print(f"Number of change points in each task session:\n{change_point_indices.groupby('session')['outcome_num'].count()}")


    return num_change_points, change_points_per_session, change_point_indices, task_session_df




def compute_surprise_from_observations(participant_str, directory_path, task_session_df, change_point_indices_df, p_c, p_min, p_max, plot_surprise = False):
    """
    Function computes the level of surprise of each set of task session observations. Achieves this using the "run_inference" function (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis) to generate posterior inferences of generative probabilities of the data sequence (seq). Surprise is calculated as:

                                                                       S(x) =  -log2 P(x)

    The surprise of an event (x) is therefore the product of the negative logarithm to base 2 and the probability of the event (x). In the code, surprise is calculated as:

                                                       surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    
    So the surprise of observation at time t is the product of the negative logarithm to base 2 and the estimated probability of that observation given the last observation. 

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a dataframe with the level of surprise included. For some basic analysis, the level of surprise is categorised as either "high" and "low" based on the interquartile range of the surprise values from all task sessions. The updated posterior distribution characteristics (mean and SD) in the face of each new observation are also returned.

    
    Parameters:
    - participant_str: str, participant identifier (single digits for 1-9, double digits for 10-99)  
    - directory_path: Path, directory path to the participant level of the data storage
    - task_session_df: pd.DataFrame, contains data extracted from .csv data file to be used in/or provide context to model analysis.
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The identified outcome number within each task session will be used in plot visualisation.
    - plot_surprise: boolean, if true, will plot the change in surprise per observation across each task session (default = False).
    
    Parameters in run_inference function:
    - seq: np.array (connverted from the 'stim' column in task_session_df to a binary sequence of observations (1 for 'G', 0 for 'A')
    - options: dict, keys:
        'p_c': a priori volatility (for hmm)
        'resol': number of bins used for discretization (for hmm)
        'p1_min': the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
        'p1_max': the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)

     Returns:
    - task_session_df: pd.DataFrame, contains computed variables from the ideal observer model.  With this function, calculated 'surprise' columns are added for each task session. Also contains model calculated posterior distribution parameters (mean and SD of the posterior).
    """
    # Define options for "run inference" function
    options = {'p_c': p_c, 'resol': 20, 'p_min': p_min, 'p_max': p_max}

    task_session_surprise_list = []  # List to store session DataFrames

    for task_session, session_data in task_session_df.groupby('session'):
        seq = np.array(session_data['stim'])
        binary_seq = np.where(seq == 'G', 1, 0)

        # Run inference to compute surprise for the sequence of observations
        output = run_inference(binary_seq, options=options)

        # Obtaining variables of interest
        surprise_values = output['surprise']
        model_post_prob_1 = output[(1,)]['mean']
        model_post_prob_0 = output[(0,)]['mean']
        model_post_prob_var = output[(1,)]['SD']

        # Add new columns to computational_var_df
        session_indices = session_data.index
        task_session_df.loc[session_indices, 'surprise'] = surprise_values
        task_session_df.loc[session_indices, 'model_post_prob_1'] = model_post_prob_1
        task_session_df.loc[session_indices, 'model_post_prob_0'] = model_post_prob_0
        task_session_df.loc[session_indices, 'model_post_prob_var'] = model_post_prob_var


    # Removing null values from the start of each task session and replacing with a surprise value of 1
    null_values = task_session_df['surprise'].isna()

    indices = []

    for i, val in enumerate(null_values):
        if val: 
            indices.append(i)

    print(f"Number of NaN values in surprise column: {len(indices)}")
    print("List of indices with NaN values:", indices)

    task_session_df.loc[null_values, 'surprise'] = 1

    # Option to plot the data for surprise and posterior distribution parameters
    if plot_surprise:
        unique_sessions = task_session_df['session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=False,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = task_session_df[task_session_df['session'] == session]
            fig.add_trace(go.Scatter(x=session_data['trial_ID'], y=session_data['surprise'], mode='lines', line=dict(color='blue'), name=f'Surprise - Session {session}', showlegend=False), row=i, col=1)

            outcome_colors = ['red' if val == 'A' else 'green' for val in task_session_df['stim']]

            fig.add_trace(
                go.Scatter(
                    x=session_data['trial_ID'],
                    y=[-0.5] * len(session_data['stim']),
                    mode='markers',
                    marker=dict(color=outcome_colors, size = 4),
                    name='Outcomes', 
                    showlegend=False
                ), row=i, col=1)
            
            change_points = change_point_indices_df[change_point_indices_df['session'] == session]['outcome_num']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=session_data['surprise'].max(), 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Surprise across task sessions for participant {participant_str}", showlegend=False)

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_surprise-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()


    return task_session_df



def compute_prior_conf_and_cpp_from_observations(participant_str, directory_path, task_session_df, change_point_indices_df, p_c, p_min, p_max, estimate_type, conf_type, unit = 'confidence', plot_prior_conf = False, plot_cpp = False):
    """
    Function computes the "confidence" in the prior and change point probability from each set of task session observations. Computes "confidence" and change point probability using functions within "model_learner" (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis). "Confidence" is the -log(sd) of the prior distribution, while change point probability is calculated from the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring, divided by the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring plus the conditional probability of an observation (at time t) given no change point has occurred multiplied by the true probability that a change point did not occur:

                                    p(cp | X_t) = p(X_t | cp) p(cp) / (p(X_t | cp) p(cp) + p(X_t | no cp) (1-p(cp)))

    This function takes the 'outcome' data, converts the observations to a sequence for each task session, before estimating prior "confidence" and change point probability and returning the modified dataframe.

    Parameters:
    - participant_str: string, participant identifier (single digits for 1-9, double digits for 10-99)  
    - directory_path: Path, directory path to the participant level of the data storage
    - task_session_df: pd.DataFrame, contains behavioural variables plus computed variables from the ideal observer model.
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The outcome number within each task session will be used in plot visualisation.
    - 'p_c': float, true change point probability 
    - 'p_min': float, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) 
    - 'p_max': float, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) 
    - estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options account for volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected (only working with "mean" at the minute. Cant get "mean_pred" to work)
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 
    - unit: string, parameter specifying the units in which prior variability should be presented. Four options available: 'confidence', 'uncertaity_sd', 'uncertainty_log_sd', and 'uncertainty_sd_cpp'.
    - 'plot_prior_conf: boolean, if true will plot the prior "confidence" of the probability distribution (default = False)
    - 'plot_cpp: boolean, if true will plot the change point probability (default = False)

    
    Parameters in functions taken from "model_learner":
    - 'outcomes': np.array, outcome observations for each task session
    - 'p_c': interger, true change point probability  
    - 'p_min': float, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'p_max': float, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options accountfor volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected.
    - 'keys': dictionary keys, defines ideal_learner_prob_inference_from_outcomes function, specifying the parameters of the distribution to be computed (['mean', 'SD', 'dist'])
    - 'resol': float, number of bins used for discretization (for hmm). Defines ideal_learner_prob_inference_from_outcomes function. Set to 20. 
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 

     Returns:
    -  computational_var_df: pd.DataFrame, contains computed variables from the ideal observer model. With this function, 'prior_dist' (the prior distribution mean), 'prior_conf' (the variance in the prior distribution) and the 'cpp' (change point probability) are added as columns for each task session. Also contains a variable that categorises the prior_conf and cpp based on the interquartile range across all task sessions (used for basic analysis).
    """
    for task_session, session_data in task_session_df.groupby('session'):
        seq = np.array(session_data['stim'])
        binary_seq = np.where(seq == 'G', 1, 0)

        # Run functions to compute confidence and change point probability. Add data to existing data frame computational_var_df
        prior_dist = ideal_learner_prob_estimates_from_outcomes(
            binary_seq, p_c, estimate_type=estimate_type, p1_min=p_min, p1_max=p_max)  
        
        prior_conf = ideal_learner_prob_confidence_from_outcomes(
            binary_seq, p_c, conf_type=conf_type, p1_min=p_min, p1_max=p_max)
        
        cpp = ideal_learner_prob_cpp(binary_seq, p_c, p1_min=p_min, p1_max=p_max)

        # Option to modify the units prior confidence is presented in
        if unit == 'uncertainty_sd':
            prior_conf = np.exp(-prior_conf)
        elif unit == 'uncertainty_log_sd':
            prior_conf = -prior_conf
        elif unit == 'uncertainty_sd_cpp':
            prior_conf = np.exp(-prior_conf) * (1 - cpp)
        elif unit == 'confidence':
            pass
        else:
             raise ValueError('Correct unit for prior variability needs to be specified. ''Options are: "confidence", "uncertainty_sd", "uncertainty_log_sd", "uncertainty_sd_cpp".')

        # Add new columns to task_session_df
        session_indices = session_data.index
        task_session_df.loc[session_indices, 'prior_conf'] = prior_conf
        task_session_df.loc[session_indices, 'cpp'] = cpp

    # Option to plot the data for prior confidence and change point probability
    if plot_prior_conf:
        unique_sessions = task_session_df['session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=False,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = task_session_df[task_session_df['session'] == session]
            fig.add_trace(go.Scatter(x=session_data['trial_ID'], y=session_data['prior_conf'], mode='lines', line=dict(color='blue'), name=f'Prior confidence - Session {session}'), row=i, col=1)

            outcome_colors = ['red' if val == 'A' else 'green' for val in task_session_df['stim']]

            fig.add_trace(
                go.Scatter(
                    x=session_data['trial_ID'],
                    y=[0] * len(session_data['stim']),
                    mode='markers',
                    marker=dict(color=outcome_colors, size = 4),
                    name='Outcomes', 
                    showlegend=False
                ), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['session'] == session]['outcome_num']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=session_data['prior_conf'].max(), 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Prior confidence across task sessions for participant {participant_str}", showlegend=False)

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_prior-conf-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    if plot_cpp:
        unique_sessions = task_session_df['session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=False,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = task_session_df[task_session_df['session'] == session]
            fig.add_trace(go.Scatter(x=session_data['trial_ID'], y=session_data['cpp'], mode='lines', line=dict(color='blue'), name=f'Change point probability - Session {session}'), row=i, col=1)

            outcome_colors = ['red' if val == 'A' else 'green' for val in task_session_df['stim']]

            fig.add_trace(
                go.Scatter(
                    x=session_data['trial_ID'],
                    y=[0] * len(session_data['stim']),
                    mode='markers',
                    marker=dict(color=outcome_colors, size = 4),
                    name='Outcomes', 
                    showlegend=False
                ), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['session'] == session]['outcome_num']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=session_data['cpp'].max(), 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Change point probability across task sessions for participant {participant_str}", showlegend=False)

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_cpp-by-task-session.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    return task_session_df