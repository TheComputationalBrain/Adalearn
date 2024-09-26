import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis')
print(sys.path)  

try:
    from main_functions import run_inference
    print("Imported run_inference successfully")
except ImportError as e:
    print(f"Error importing run_inference: {e}")


def compute_surprise_of_observations (participant_str, condition_str, adalearn_data_df):
    """
    Computes the level of surprise of each observation. Achieves this using the "run_inference" function (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis) to generate posterior inferences of generative probabilities of the data sequence (seq). Surprise is calculated as:

                                                                S(x) =  -log2 P(x)

    The surprise of an event (x) is therefore the product of the negative logarithm to base 2 and the probability of the event (x). In the code, surprise is calculated as:
                                                    surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    
    So the surprise of observation at time t is the product of the negative logarithm to base 2 and the estimated probability of that observation given the last observation. 

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a dataframe with the level of surprise included. 

    
    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - adalearn_data_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 

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

        surprise_values = output['surprise']
        surprise_df = pd.DataFrame(surprise_values)
        surprise_df['session'] = task_session
        task_session_surprise_df.append(surprise_df)
         
    surprise_df = pd.concat(task_session_surprise_df)
    print(surprise_df.head())

    # Create a DataFrame to return the results
    #surprise_output_df = pd.DataFrame({
    #    'task_session': task_sessions,
    #    '(0,)': task_sess_outcome_0,
    #    '(1,)': task_sess_outcome_1,
    #    'surprise': task_session_surprise
    #})

    return surprise_df