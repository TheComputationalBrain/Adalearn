import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis')
print(sys.path)  

try:
    from main_functions import run_inference
    print("Imported run_inference successfully")
except ImportError as e:
    print(f"Error importing run_inference: {e}")




def compute_surprise_of_observations (participant_str, condition_str, adalearn_data_df, plot_surprise = True, plot_posterior_conf = True, plot_posterior_dist_means = True):
    """
    Computes the level of surprise of each observation. Achieves this using the "run_inference" function (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis) to generate posterior inferences of generative probabilities of the data sequence (seq). Surprise is calculated as:

                                                                S(x) =  -log2 P(x)

    The surprise of an event (x) is therefore the product of the negative logarithm to base 2 and the probability of the event (x). In the code, surprise is calculated as:
                                                    surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    
    So the surprise of observation at time t is the product of the negative logarithm to base 2 and the estimated probability of that observation given the last observation. 

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a dataframe with the level of surprise included. The updated posterior distribution characteristics (mean and SD) in the face of each new observation are also returned.

    
    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - adalearn_data_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 
    - plot_surprise: boolean, if true, will plot the change in surprise per observation across each task session (default is set to True).
    - plot_posterior_conf: boolean, if true, will plot the change in posterior distribution SD (confidence) per observation across each task session (default is set to True). 
    - plot_posterior_dist_means: boolean, if true, will plot the change in posterior distribution means for both the blue and yellow distributions per observation across each task session (default is set to True).

    Parameters in run_inference function:
    - seq: np.array, 
    - options: dict, keys:
        'p_c': a priori volatility (for hmm)
        'resol': number of bins used for discretization (for hmm)
        'p1_min': the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
        'p1_max': the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
        'do_inference_on_current_trial': -----

     Returns:
    - surprise_df: pd.DataFrame, containing 'task_session' and 'surprise' columns for each task session.
    """

    # Define options for "run inference" function
    options = {'p_c': 0.05, 'resol': 20, 'p1_min': 0.1, 'p1_max': 0.9}

    # Initialize lists to store task session data
    task_sessions = []
    task_sess_outcome_0 = []
    task_sess_outcome_1 = []
    task_session_surprise = []
    task_session_surprise_df = []

    # Loop over task sessions and convert the outcome column to a sequence (np.array)
    for task_session, session_data in adalearn_data_df.groupby('task_session'):
        seq = np.array(session_data['outcome'])

        # Run inference to compute surprise for the sequence
        output = run_inference(seq, options=options)
    
        task_sessions.append(task_session)
        task_sess_outcome_0.append(output[(0,)])
        task_sess_outcome_1.append(output[(1,)])
        task_session_surprise.append(output['surprise'])

        # Creating dataframe to store for that task session
        surprise_values = output['surprise']
        surprise_df = pd.DataFrame({'surprise': surprise_values})
        surprise_df['task_session'] = task_session
        surprise_df['task_sess_count'] = surprise_df.groupby('task_session').cumcount() + 1 
        task_session_surprise_df.append(surprise_df)
         
    surprise_df = pd.concat(task_session_surprise_df)

    # Extracting posterior distribution values for outcomes 0 and 1.
    # Outcome 0 
    task_sess_outcome_0_df_list = []

    for i, outcome in enumerate(task_sess_outcome_0):
        outcome_df = pd.DataFrame({
            'mean_0': np.ravel(outcome['mean']),
            'SD_0': np.ravel(outcome['SD'])
        })
    
        outcome_df['task_session'] = task_sessions[i]
    
        task_sess_outcome_0_df_list.append(outcome_df)

    # Outcome 1
    task_sess_outcome_1_df_list = []

    for i, outcome in enumerate(task_sess_outcome_1):
        outcome_df = pd.DataFrame({
            'mean_1': np.ravel(outcome['mean']),
            'SD_1': np.ravel(outcome['SD'])
        })
    
        outcome_df['task_session'] = task_sessions[i]
    
        task_sess_outcome_1_df_list.append(outcome_df)

    # Creating 
    task_sess_post_dis_0 = pd.concat(task_sess_outcome_0_df_list, ignore_index=True)
    task_sess_post_dis_1 = pd.concat(task_sess_outcome_1_df_list, ignore_index=True)
    task_sess_post_dis_df = pd.concat([task_sess_post_dis_0, task_sess_post_dis_1], axis=1)
    task_sess_post_dis_df = task_sess_post_dis_df.loc[:, ~task_sess_post_dis_df.columns.duplicated()]
    task_sess_post_dis_df['task_sess_count'] = task_sess_post_dis_df.groupby('task_session').cumcount() + 1 


    if plot_surprise:
        unique_sessions = surprise_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = surprise_df[surprise_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['surprise'], mode='lines', name=f'Surprise - Session {session}'), row=i, col=1)

        fig.update_layout(height=300 * len(unique_sessions), title_text="Surprise across task sessions", showlegend=False)

        fig.show()

    if plot_posterior_conf:
        unique_task_sessions = task_sess_post_dis_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = task_sess_post_dis_df[task_sess_post_dis_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['SD_1'], mode='lines', name=f'Posterior distribution SD (confidence) - Session {session}'), row=i, col=1)

        fig.update_layout(height=300 * len(unique_task_sessions), title_text="Posterior distribution SD ('confidence') across task sessions", showlegend=False)

        fig.show()

    if plot_posterior_dist_means:
        unique_task_sessions = task_sess_post_dis_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = task_sess_post_dis_df[task_sess_post_dis_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['mean_0'], mode='lines', name=f'Posterior distribution mean (P(0)) - Session {session}', line_color = 'yellow'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['mean_1'], mode='lines', name=f'Posterior distribution mean (P(1)) - Session {session}', line_color = 'blue'), row=i, col=1)

        fig.update_layout(height=300 * len(unique_task_sessions), title_text="Posterior distribution mean across task sessions", showlegend=False)

        fig.show()

    return surprise_df, task_sess_post_dis_df