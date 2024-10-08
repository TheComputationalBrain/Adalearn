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
    print("Imported successfully")
except ImportError as e:
    print(f"Error importing: {e}")

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference
import data_analysis_utils as dana
from data_analysis_utils import get_prev_estimates, get_values_with_prior
from model_learner import ideal_learner_prob_estimates_from_outcomes, ideal_learner_prob_inference_from_outcomes, ideal_learner_prob_confidence_from_outcomes, ideal_learner_prob_cpp


def compute_surprise_from_observations(participant_str, condition_str, adalearn_data_df, change_point_indices_df, plot_surprise = True, plot_posterior_conf = True, plot_posterior_dist_means = True):
    """
    Function computes the level of surprise of each set of task session observations. Achieves this using the "run_inference" function (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis) to generate posterior inferences of generative probabilities of the data sequence (seq). Surprise is calculated as:

                                                                                                                     S(x) =  -log2 P(x)

    The surprise of an event (x) is therefore the product of the negative logarithm to base 2 and the probability of the event (x). In the code, surprise is calculated as:

                                                                                                    surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    
    So the surprise of observation at time t is the product of the negative logarithm to base 2 and the estimated probability of that observation given the last observation. 

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a dataframe with the level of surprise included. For some basic analysis, the level of surprise is categorised as either "high" and "low" based on the interquartile range of the surprise values from all task sessions. The updated posterior distribution characteristics (mean and SD) in the face of each new observation are also returned.

    
    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - adalearn_data_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The identified trial number within each task session will be used in plot visualisation.
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

     Returns:
    - surprise_df: pd.DataFrame, containing 'task_session' and 'surprise' columns for each task session.
    - task_sess_post_dist_df: pd.DataFrame, contains the model calculated posterior distribution. Includes the mean and SD of the posterior. 
    - surprise_cat_df: pd.DataFrame, copy of surprise_df but contains a variable that categorises the level of surprise per observation based on the interquartile range across all task sessions. 
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

        # Run inference to compute surprise for the sequence of observations
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
         
    surprise_df = pd.concat(task_session_surprise_df, ignore_index=True)
    surprise_df['cum_trigger_count'] = surprise_df.index
    (print(surprise_df))

    # Categorizing the level of surprise for basic analysis
    surprise_cat_df = surprise_df.copy()
    surprise_cat_df['surprise'] = pd.to_numeric(surprise_cat_df['surprise'], errors='coerce')
    surprise_cat_df = surprise_cat_df.dropna(subset=['surprise'])
    first_quartile = surprise_cat_df['surprise'].quantile(0.25)
    third_quartile = surprise_cat_df['surprise'].quantile(0.75)

    surprise_cat_df['surprise_cat'] = 'between_quartiles'

    surprise_cat_df.loc[surprise_cat_df['surprise'] < first_quartile, 'surprise_cat'] = 'below_first_quartile'
    surprise_cat_df.loc[surprise_cat_df['surprise'] > third_quartile, 'surprise_cat'] = 'above_third_quartile'


    # Extracting posterior distribution values for outcomes 0 and 1.
    # Outcome 0 - 'Yellow'
    task_sess_outcome_0_df_list = []

    for i, outcome in enumerate(task_sess_outcome_0):
        outcome_df = pd.DataFrame({
            'mean_0': np.ravel(outcome['mean']),
            'SD_0': np.ravel(outcome['SD'])
        })
    
        outcome_df['task_session'] = task_sessions[i]
    
        task_sess_outcome_0_df_list.append(outcome_df)

    # Outcome 1 - 'Blue'
    task_sess_outcome_1_df_list = []

    for i, outcome in enumerate(task_sess_outcome_1):
        outcome_df = pd.DataFrame({
            'mean_1': np.ravel(outcome['mean']),
            'SD_1': np.ravel(outcome['SD'])
        })
    
        outcome_df['task_session'] = task_sessions[i]
    
        task_sess_outcome_1_df_list.append(outcome_df)

    # Creating dataframe for posterior distribution characteristics
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

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=5, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Surprise across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()

    if plot_posterior_conf:
        unique_task_sessions = task_sess_post_dis_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = task_sess_post_dis_df[task_sess_post_dis_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['SD_1'], mode='lines', name=f'Posterior distribution SD (confidence) - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=0.5, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_task_sessions), title_text=f"Posterior distribution SD ('confidence') across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()

    if plot_posterior_dist_means:
        unique_task_sessions = task_sess_post_dis_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = task_sess_post_dis_df[task_sess_post_dis_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['mean_0'], mode='lines', name=f'Posterior distribution mean (P(0)) - Session {session}', line_color = 'yellow'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['mean_1'], mode='lines', name=f'Posterior distribution mean (P(1)) - Session {session}', line_color = 'blue'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=1, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_task_sessions), title_text=f"Posterior distribution mean across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()

    return surprise_df, task_sess_post_dis_df, surprise_cat_df



def compute_prior_conf_and_cpp_from_observations(participant_str, condition_str, adalearn_data_df, change_point_indices_df, p_c, p1_min, p1_max, estimate_type, conf_type, plot_distribution_mean = True, plot_prior_conf = True, plot_cpp = True):
    """
    Function computes the "confidence" in the prior and change point probability from each set of task session observations. Computes "confidence" and change point probability using functions within "model_learner" (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis). "Confidence" is the -log(sd) of the prior distribution, while change point probability is calculated from the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring, divided by the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring plus the conditional probability of an observation (at time t) given no change point has occurred multiplied by the true probability that a change point did not occur:

                                                                                                    p(cp | X_t) = p(X_t | cp) p(cp) / (p(X_t | cp) p(cp) + p(X_t | no cp) (1-p(cp)))

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a separate dataframe for prior "confidence" and change point probability.

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - adalearn_data_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The identified trial number within each task session will be used in plot visualisation.
    - 'p_c': interger, true change point probability (should be 0.05)
    - 'p1_min': interger, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) (should be 0.1)
    - 'p1_max': interger, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) (should be 0.9)
    - estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options accountfor volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected (only working with "mean" at the minute. Cant get "mean_pred" to work)
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 
    - 'plot_distribution_mean: boolean, if true will plot the prior mean of the probability distribution (default set to true)
    - 'plot_prior_conf: boolean, if true will plot the prior "confidence" of the probability distribution (default set to true)
    - 'plot_cpp: boolean, if true will plot the change point probability (default set to true)

    
    Parameters in functions taken from "model_learner":
    - 'outcomes': np.array, outcome observations for each task session
    - 'p_c': interger, true change point probability  
    - 'p1_min': interger, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'p1_max': interger, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options accountfor volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected.
    - 'keys': dictionary keys, defines ideal_learner_prob_inference_from_outcomes function, specifying the parameters of the distribution to be computed (['mean', 'SD', 'dist'])
    - 'resol': interger, number of bins used for discretization (for hmm). Defines ideal_learner_prob_inference_from_outcomes function. Set to 20. 
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 

     Returns:
    - prior_outcome_df: pd.DataFrame, contains the selected inferred parameters of the prior distribution (typically, 'mean-pred', 'SD', 'dist') for each task session
    - prior_conf_df: pd.DataFrame, contains the -log(SD) - "confidence"/precision - of prior distribution. Typically choose confidence type "prev-conf-pred" to account for volatility. "Confidence" is calculated for each task session
    - cpp_df: pd.DataFrame, contains the computed change point probability for each task session
    """
    # Lists to store computed parameters for each task session
    prior_dist_df_list = []
    prior_conf_df_list =[]
    cpp_df_list =[]

    for task_session, session_data in adalearn_data_df.groupby('task_session'):
        seq = np.array(session_data['outcome'])

        # Run functions to compute prior parameters, confidence and change point probability. Create data frames for each variable and then save the data frame for each task session in a list

        outcome_inference = ideal_learner_prob_inference_from_outcomes(seq, p_c, keys=['mean', 'SD', 'dist'], p1_min=p1_min, p1_max=p1_max, do_inference_on_current_trial=True, resol=20)

        prior_dist = ideal_learner_prob_estimates_from_outcomes(seq, p_c, estimate_type = estimate_type, p1_min = p1_min, p1_max = p1_max)
        mean_values = prior_dist
        prior_dist_df = pd.DataFrame({
            'mean': mean_values,
        })
        prior_dist_df['task_session'] = task_session
        prior_dist_df['task_sess_count'] = prior_dist_df.groupby('task_session').cumcount() + 1
        prior_dist_df_list.append(prior_dist_df)
        

        prior_conf = ideal_learner_prob_confidence_from_outcomes(seq, p_c, conf_type = conf_type, p1_min = p1_min, p1_max = p1_max)
        conf_values = prior_conf
        prior_conf_df = pd.DataFrame({
            'prior_conf': conf_values
        })
        prior_conf_df['task_session'] = task_session
        prior_conf_df['task_sess_count'] = prior_conf_df.groupby('task_session').cumcount() + 1
        prior_conf_df_list.append(prior_conf_df)


        cpp =  ideal_learner_prob_cpp(seq, p_c, p1_min = p1_min, p1_max = p1_max)
        cpp_values = cpp
        cpp_df = pd.DataFrame({
            'cpp': cpp_values
        })
        cpp_df['task_session'] = task_session
        cpp_df['task_sess_count'] = cpp_df.groupby('task_session').cumcount() + 1
        cpp_df_list.append(cpp_df)


    prior_distribution_df = pd.concat(prior_dist_df_list, ignore_index=True)
    prior_distribution_df['cum_trigger_count'] = prior_distribution_df.index

    prior_confidence_df = pd.concat(prior_conf_df_list, ignore_index=True)
    prior_confidence_df['cum_trigger_count'] = prior_confidence_df.index

    change_point_prob_df = pd.concat(cpp_df_list, ignore_index=True)
    change_point_prob_df['cum_trigger_count'] = change_point_prob_df.index


    if plot_distribution_mean:
        unique_sessions = prior_distribution_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = prior_distribution_df[prior_distribution_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['mean'], mode='lines', name=f'Prior distribution mean - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=1, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Prior distribution mean across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()


    if plot_prior_conf:
        unique_sessions = prior_confidence_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = prior_confidence_df[prior_confidence_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['prior_conf'], mode='lines', name=f'Prior confidence - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=2.25, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Prior confidence across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()

    if plot_cpp:
        unique_sessions = change_point_prob_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = change_point_prob_df[change_point_prob_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['task_sess_count'], y=session_data['cpp'], mode='lines', name=f'Change point probability - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['trial_number']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=0, y1=0.5, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        fig.update_layout(height=300 * len(unique_sessions), title_text=f"Change point probability across task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)

        fig.show()

    return prior_distribution_df, prior_confidence_df, change_point_prob_df