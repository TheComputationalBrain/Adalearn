import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import sem
from scipy.stats import linregress
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference
from model_learner import ideal_learner_prob_estimates_from_outcomes, ideal_learner_prob_inference_from_outcomes, ideal_learner_prob_confidence_from_outcomes, ideal_learner_prob_cpp



def compute_learning_rate(computational_var_df, estimate_parameter=None, estimate_prior=None):
    """
    Function to compute learning rate for each outcome presentation across groups defined 
    by a specific parameter (e.g., task session). 
    Learning rate (αt) is derived from the delta rule:

                            αt = (vt - vt-1) / (xt - vt-1)

    where:
        - vt is the current estimate at time t
        - vt-1 is the previous estimate at time t-1
        - xt is the observed outcome at time t (e.g., 1 for blue and 0 for yellow).

    The function groups takes a dataframe which contains and 'outcome' and some estimate parameter. If there is a grouping variable column (e.g., 'task_session'), the function computes the learning rate for each group based on the provided estimate parameter. Returns the updated dataframe with a new 'learning_rate' column and associated estimate update and error columns.

    Parameters:
    - computational_var_df: pd.DataFrame, the input dataframe containing the estimates and select model derived outcomes. To be used to house additional computational variables.    
    - estimate_parameter: string, the column name representing the estimates from which the learning rate is to be derived. It can be a participant's estimate or the ideal observer model estimate.
    - estimate_prior: float, the prior estimate to be used in the learning rate calculation. Reflects the value where the estimate slider is positioned at the start of each session (i.e. 0.5)

    Returns:
    - computational_var_df: pd.DataFrame, the input dataframe with the following columns added for the respective estimate parameter used (i.e. participant's estimates or ideal observer estimates): 'learning_rate' (calculated learning rate indicating the size of the update as a proportion of the estimate error), 'estimate_updates' (the change in estimate at trial t from trial t-1), 'estimate_error' (difference between the outcome and subject's estimate at trial t-1) and 'was_updated' (boolean variable indicateing that the estimate_update differs from 0 with some tolerance).
    """
    learning_rate_col = f'learning_rate_{estimate_parameter}'
    estimate_updates_col = f'{estimate_parameter}_updates'
    estimate_errors_col = f'{estimate_parameter}_errors'
    was_update_cat = f'was_updated_{estimate_parameter}'

    # Group the DataFrame by the task session
    grouped_lr_df = computational_var_df.groupby('task_session')

    # Iterate through each task session to compute the learning rate
    for group, data in grouped_lr_df:

        # Prepend the prior estimate to the 'estimate' column
        estimates_with_prior = np.insert(data[estimate_parameter], 0, estimate_prior, axis=0)

        # Compute the estimate updates and errors for learning rate calculation (estimate_updates = vt - vt-1; estimate_errors = xt - vt-1)
        estimate_updates = estimates_with_prior[1:] - estimates_with_prior[:-1]

        estimate_errors = data['outcome'] - estimates_with_prior[:-1]
        
        # Compute the raw learning rates (delta rule)
        learning_rate = estimate_updates / estimate_errors

        # Computing mask to check if an update occurred (tolerance around 0: relative = 1e-05; absolute = 1e-08)
        was_updated = ~np.isclose(estimate_updates, 0.)  
        
        # Assign computed learning rates to the original DataFrame rows (except first row)
        computational_var_df.loc[data.index, learning_rate_col] = learning_rate
        computational_var_df.loc[data.index, estimate_updates_col] = estimate_updates
        computational_var_df.loc[data.index, estimate_errors_col] = estimate_errors
        computational_var_df.loc[data.index, was_update_cat] = was_updated

    return computational_var_df



def behavioural_learning_rate_analysis(computational_var_df, change_point_indices_df, participant_str, condition_str, estimate_prior = None, plot_learning_rate_by_ses = False):
    """ 
    Function to calculate the behavioural learning rate based on participants estimates following each outcome presentation. Returns behavioural dataframe with new learning rate variable created. Option to plot learning rate across each task session. 

    Parameters:
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - change_point_indices_df: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. Used to mark the change points on plots (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - estimate_prior: float, the prior estimate to be used in the learning rate calculation. Reflects the value where the estimate slider is positioned at the start of each session (i.e. 0.5).
    - plot_learning_rate_by_ses: boolean, if true, will plot the calculated (and filtered) learning rate across each task session. Change points added as dotted horizontal lines for visualisation (default = false).

    Returns:
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable. Final data frame contains a "cleaned learning rate", which removes values that exceed set thresholds (-0.6 and +1.3, which were determined a priori from a pilot study conducted before Foucault & Meyniel, 2024. Thresholding used to to avoid aberrant learning rate values due to very small denominators (error: estimate - outcome)).      
    """
    # Implementing learning rate equation
    computational_var_df = compute_learning_rate(computational_var_df, estimate_parameter='estimate', estimate_prior=estimate_prior) 

    computational_var_df['cleaned_learning_rate_estimate'] = computational_var_df['learning_rate_estimate'].where((computational_var_df['learning_rate_estimate'] >= -0.6) & (computational_var_df['learning_rate_estimate'] < 1.3), float('nan'))

    removed_data_points = computational_var_df['learning_rate_estimate'].notna().sum() - computational_var_df['cleaned_learning_rate_estimate'].notna().sum()
    print(f"Number of data points removed in cleaning step: {removed_data_points}")

    # Plotting learning rate data
    # Create subplots, one for each task session
    if plot_learning_rate_by_ses:
        unique_sessions = computational_var_df['task_session'].unique()
    
        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                        subplot_titles=[f'Beahvioural learning rate in task session {session}' for session in unique_sessions])

        # Add learning rate traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            
            fig.add_trace(
                go.Scatter(x=session_data['outcome_num'], y=session_data['cleaned_learning_rate_estimate'], mode='lines+markers', name=f'Beahvioural learning rate in task session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=-0.8, y1=1.5, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        # Update layout
        fig.update_layout(height=300*len(unique_sessions), width=800, title_text=f"Behavioural learning rate across task sessions for participant {participant_str} in condition {condition_str}",
                  showlegend=False, title_x=0.5)

        fig.update_xaxes(title_text='Trial Number', row=len(unique_sessions), col=1)

        for i in range(1, len(unique_sessions)+1):
            fig.update_yaxes(title_text='Learning Rate', row=i, col=1)

        fig.show()

    
    return computational_var_df



def ideal_observer_learning_rate_analysis(computational_var_df, change_point_indices_df, participant_str, condition_str, estimate_prior = None, plot_learning_rate_by_ses = False):
    """ 
    Function to calculate the learning rate based on the ideal observer model estimates following each outcome presentation. Returns behavioural dataframe with new learning rate variable created. Option to plot learning rate across each task session. 

    Parameters:
    -  computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - change_point_indices_df: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. Used to mark the change points on plots (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - estimate_prior: float, the prior estimate to be used in the learning rate calculation. Reflects the value where the estimate slider is positioned at the start of each session (i.e. 0.5).
    - plot_learning_rate_by_ses: boolean, if true, will plot the calculated (and filtered) learning rate across each task session. Change points added as dotted horizontal lines for visualisation (default = false).

    Returns:
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable. Final data frame contains a "cleaned learning rate", which removes values that exceed set thresholds (-0.6 and +1.3, which were determined a priori from a pilot study conducted before Foucault & Meyniel, 2024. Thresholding used to to avoid aberrant learning rate values due to very small denominators (error: estimate - outcome)).      
    """
    # Implementing learning rate equation
    computational_var_df = compute_learning_rate(computational_var_df, estimate_parameter='ideal_observer', estimate_prior=estimate_prior) 

    computational_var_df['cleaned_learning_rate_ideal_observer'] = computational_var_df['learning_rate_ideal_observer'].where((computational_var_df['learning_rate_ideal_observer'] >= -0.6) & (computational_var_df['learning_rate_ideal_observer'] < 1.3), float('nan'))

    removed_data_points = computational_var_df['learning_rate_ideal_observer'].notna().sum() - computational_var_df['cleaned_learning_rate_ideal_observer'].notna().sum()
    print(f"Number of data points removed in cleaning step: {removed_data_points}")

    # Plotting learning rate data
    # Create subplots, one for each task session
    if plot_learning_rate_by_ses:
        unique_sessions = computational_var_df['task_session'].unique()
    
        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                        subplot_titles=[f'Ideal observer model learning rate in task session {session}' for session in unique_sessions])

        # Add learning rate traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(
                go.Scatter(x=session_data['outcome_num'], y=session_data['cleaned_learning_rate_ideal_observer'], mode='lines+markers', name=f'Ideal observer model learning rate in task session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
            for cp in change_points:
                fig.add_shape(
                    type="line", 
                    x0=cp, x1=cp, 
                    y0=-0.8, y1=1.5, 
                    line=dict(color="grey", width=2, dash="dash"),
                    xref=f'x{i}', yref=f'y{i}'
                )  

        # Update layout
        fig.update_layout(height=300*len(unique_sessions), width=800, title_text=f"Ideal observer model learning rate across task sessions for participant {participant_str} in condition {condition_str}",
                  showlegend=False, title_x=0.5)

        fig.update_xaxes(title_text='Trial Number', row=len(unique_sessions), col=1)

        for i in range(1, len(unique_sessions)+1):
            fig.update_yaxes(title_text='Learning Rate', row=i, col=1)

        fig.show()

    
    return computational_var_df



def learning_rate_by_change_point(computational_var_df, change_point_indices,participant_str, condition_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = False):
    """ 
    Function to calculate the average learning rate around change points. Uses identified change point indices to derive learning rate dynamics before and after a change in task structure. Stores responses across all task sessions and then plots the average (plus standard deviation) learning rate. 

    Parameters:
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
    - participant_str: string, identifier of participant number
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - learning_rate_parameter: string, column to be used to calculate learning rate around change points. Can be 'estimate' or 'ideal_observer'. Default = 'estimate'. 
    - pre: float, number of outcomes before the change point. Predefined to incorporate in plotting the data.
    - post: float, number of outcomes after the change point. Predefined to incorporate in plotting the data.
    - plot_learning_rate_by_cp: boolean, if true, will plot individual traces as well as averaged learning rate around change points (defined by pre and post variables) (default = False)

    Returns:
    - filtered_learning_rate_by_cp: pd.DataFrame, contains learning rate data within defined bounds around change points.
    - averaged_learning_rate_by_cp: pd.DataFrame, contains calculated mean, std, count and sem of learning rate at each relative trial position for each identified change point. 
    """
  
    # Creating dataframe to store data around defined   
    expanded_change_points = []

    for idx, row in change_point_indices.iterrows():
        task_session = row['task_session']
        outcome_num = row['outcome_num']

        for i in range(-pre, post+1):  # Predefined to be 2 trials before to 10 trials after identified change points. Can be changed to suit. 
            expanded_change_points.append({
                'task_session': task_session,
                'outcome_num': outcome_num,
                'relative_position': i,
                'outcome_num_rel_pos': outcome_num + i
            })

    expanded_change_points_df = pd.DataFrame(expanded_change_points)
    expanded_change_points_df['relative_position'] = expanded_change_points_df['relative_position'] +1                      

    # Filter main behavioural dataframe to only contain values in defined range around change point  
    filtered_learning_rate_by_cp = pd.merge(expanded_change_points_df, computational_var_df, 
                                            left_on=['task_session', 'outcome_num_rel_pos'], 
                                            right_on=['task_session', 'outcome_num'], 
                                            how='left', suffixes=('', '_y'))

    # Drop unnecessary columns
    filtered_learning_rate_by_cp.drop(columns=['rel_time', 'outcome_num_y'], inplace=True)


    # Get the parameter of interest
    learning_rate_col = f'cleaned_learning_rate_{learning_rate_parameter}'

    # Calculate the average, standard deviation and count of the behavioural learning rate across the different relative positions to the change point
    averaged_learning_rate_by_cp = filtered_learning_rate_by_cp.groupby('relative_position')[learning_rate_col].agg(['mean', 'std', 'count']).reset_index()

    # Calculate SEM
    averaged_learning_rate_by_cp['sem'] = averaged_learning_rate_by_cp['std'] / np.sqrt(averaged_learning_rate_by_cp['count'])

    # Convert 'relative_position' to a numeric range
    averaged_learning_rate_by_cp['relative_position'] = averaged_learning_rate_by_cp['relative_position'].astype(int)
    averaged_learning_rate_by_cp = averaged_learning_rate_by_cp.sort_values('relative_position')

    # Option to plot averaged behavioural learning rate in relation to change points 
    if plot_learning_rate_by_cp:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Individual Learning Rates", "Average Learning Rate"))

        num_traces = 0

        # Add individual learning rates
        for idx, row in change_point_indices.iterrows():
            session_data = filtered_learning_rate_by_cp[(filtered_learning_rate_by_cp['task_session'] == row['task_session']) & 
                                                    (filtered_learning_rate_by_cp['outcome_num'] == row['outcome_num'])]

            if len(session_data) == (pre + post + 1):
                num_traces += 1
                fig.add_trace(go.Scatter(
                    x=session_data['relative_position'],
                    y=session_data[learning_rate_col],
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
            name=learning_rate_col,
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
                x0=0, y0=-5, x1=0, y1=5,
                line=dict(
                    color="Red",
                    width=2,
                    dash="dash",
                ),
                row=i, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"Learning rate relative to change points for participant {participant_str} in condition {condition_str}",
            xaxis2_title="Trial number after change point",
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




def compute_surprise_from_observations(participant_str, condition_str, computational_var_df, change_point_indices_df, plot_surprise = False, plot_posterior_conf = False, plot_posterior_dist_means = False):
    """
    Function computes the level of surprise of each set of task session observations. Achieves this using the "run_inference" function (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis) to generate posterior inferences of generative probabilities of the data sequence (seq). Surprise is calculated as:

                                                                       S(x) =  -log2 P(x)

    The surprise of an event (x) is therefore the product of the negative logarithm to base 2 and the probability of the event (x). In the code, surprise is calculated as:

                                                       surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    
    So the surprise of observation at time t is the product of the negative logarithm to base 2 and the estimated probability of that observation given the last observation. 

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a dataframe with the level of surprise included. For some basic analysis, the level of surprise is categorised as either "high" and "low" based on the interquartile range of the surprise values from all task sessions. The updated posterior distribution characteristics (mean and SD) in the face of each new observation are also returned.

    
    Parameters:
    - participant_str: string, participant identifier (set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The identified outcome number within each task session will be used in plot visualisation.
    - plot_surprise: boolean, if true, will plot the change in surprise per observation across each task session (default = False).
    - plot_posterior_conf: boolean, if true, will plot the change in posterior distribution SD (confidence) per observation across each task session (default = False). 
    - plot_posterior_dist_means: boolean, if true, will plot the change in posterior distribution means for both the blue and yellow distributions per observation across each task session (default = False).

    Parameters in run_inference function:
    - seq: np.array, 
    - options: dict, keys:
        'p_c': a priori volatility (for hmm)
        'resol': number of bins used for discretization (for hmm)
        'p1_min': the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
        'p1_max': the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)

     Returns:
    - computational_var_df: pd.DataFrame, contains computed variables from the ideal observer model.    With this function, 'task_session' and 'surprise' columns are added for each task session. Also contains a variable that categorises the level of surprise per observation based on the interquartile range across all task sessions (used for basic analysis) and model calculated posterior distribution parameters (mean and SD of the posterior).
    """

    # Define options for "run inference" function
    options = {'p_c': 0.05, 'resol': 20, 'p1_min': 0.1, 'p1_max': 0.9}

    task_session_surprise_list = []  # List to store session DataFrames

    for task_session, session_data in computational_var_df.groupby('task_session'):
        seq = np.array(session_data['outcome'])

        # Run inference to compute surprise for the sequence of observations
        output = run_inference(seq, options=options)

        # Obtaining variables of interest
        surprise_values = output['surprise']
        model_post_prob_1 = output[(1,)]['mean']
        model_post_prob_0 = output[(0,)]['mean']
        model_post_prob_var = output[(1,)]['SD']

        # Add new columns to computational_var_df
        session_indices = session_data.index
        computational_var_df.loc[session_indices, 'surprise'] = surprise_values
        computational_var_df.loc[session_indices, 'model_post_prob_1'] = model_post_prob_1
        computational_var_df.loc[session_indices, 'model_post_prob_0'] = model_post_prob_0
        computational_var_df.loc[session_indices, 'model_post_prob_var'] = model_post_prob_var


    # Removing null values from the start of each task session and replacing with a surprise value of 1
    null_values = computational_var_df['surprise'].isna()

    indices = []

    for i, val in enumerate(null_values):
        if val: 
            indices.append(i)

    print(f"Number of NaN values in surprise column: {len(indices)}")
    print("List of indices with NaN values:", indices)

    computational_var_df.loc[null_values, 'surprise'] = 1
 
    # Categorizing the level of surprise for basic analysis
    percent_25 = computational_var_df['surprise'].quantile(0.25)
    percent_50 = computational_var_df['surprise'].quantile(0.50)  
    percent_75 = computational_var_df['surprise'].quantile(0.75)

    # Initialize a default category
    computational_var_df['surprise_cat'] = None

    # Assign categories based on quartile boundaries
    computational_var_df.loc[computational_var_df['surprise'] < percent_25, 'surprise_cat'] = 'first_quartile'
    computational_var_df.loc[
        (computational_var_df['surprise'] >= percent_25) & (computational_var_df['surprise'] < percent_50),
        'surprise_cat'
    ] = 'second_quartile'
    computational_var_df.loc[
        (computational_var_df['surprise'] >= percent_50) & (computational_var_df['surprise'] < percent_75),
        'surprise_cat'
    ] = 'third_quartile'
    computational_var_df.loc[computational_var_df['surprise'] > percent_75, 'surprise_cat'] = 'fourth_quartile'

    surprise_cat_counts = computational_var_df.groupby(['task_session', 'surprise_cat']).size().reset_index(name='counts')
    surprise_cat_df = surprise_cat_counts.pivot(index='task_session', columns='surprise_cat', values='counts').fillna(0)

    # Option to plot the data for surprise and posterior distribution parameters
    if plot_surprise:
        unique_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['surprise'], mode='lines', name=f'Surprise - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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
        unique_task_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['model_post_prob_var'], mode='lines', name=f'Posterior distribution SD (confidence) - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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
        unique_task_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_task_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_task_sessions])

        for i, session in enumerate(unique_task_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['model_post_prob_0'], mode='lines', name=f'Posterior distribution mean (P(0)) - Session {session}', line_color = 'yellow'), row=i, col=1)
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['model_post_prob_1'], mode='lines', name=f'Posterior distribution mean (P(1)) - Session {session}', line_color = 'blue'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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

    return computational_var_df, surprise_cat_df



def compute_prior_conf_and_cpp_from_observations(participant_str, condition_str, computational_var_df, change_point_indices_df, p_c, p1_min, p1_max, estimate_type, conf_type, plot_distribution_mean = False, plot_prior_conf = False, plot_cpp = False):
    """
    Function computes the "confidence" in the prior and change point probability from each set of task session observations. Computes "confidence" and change point probability using functions within "model_learner" (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis). "Confidence" is the -log(sd) of the prior distribution, while change point probability is calculated from the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring, divided by the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring plus the conditional probability of an observation (at time t) given no change point has occurred multiplied by the true probability that a change point did not occur:

                                    p(cp | X_t) = p(X_t | cp) p(cp) / (p(X_t | cp) p(cp) + p(X_t | no cp) (1-p(cp)))

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a separate dataframe for prior "confidence" and change point probability.

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - computational_var_df: pd.DataFrame, contains computed variables from the ideal observer model.
    - change_point_indices_df: pd.DataFrame, contains outputs related to the presence of change points in the hidden task structure. The outcome number within each task session will be used in plot visualisation.
    - 'p_c': float, true change point probability (should be 0.05)
    - 'p1_min': float, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) (should be 0.1)
    - 'p1_max': float, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution) (should be 0.9)
    - estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options accountfor volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected (only working with "mean" at the minute. Cant get "mean_pred" to work)
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 
    - 'plot_distribution_mean: boolean, if true will plot the prior mean of the probability distribution (default = False)
    - 'plot_prior_conf: boolean, if true will plot the prior "confidence" of the probability distribution (default = False)
    - 'plot_cpp: boolean, if true will plot the change point probability (default = False)

    
    Parameters in functions taken from "model_learner":
    - 'outcomes': np.array, outcome observations for each task session
    - 'p_c': interger, true change point probability  
    - 'p1_min': float, the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'p1_max': float, the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    - 'estimate_type': string, parameter defining ideal_learner_prob_estimates_from_outcomes function. Options are "mean", "mean-pred", "median", "median-pred". The "-pred" options accountfor volatility in the estimate. Function returns the standard deviation and distribution with each distribution parameter selected.
    - 'keys': dictionary keys, defines ideal_learner_prob_inference_from_outcomes function, specifying the parameters of the distribution to be computed (['mean', 'SD', 'dist'])
    - 'resol': float, number of bins used for discretization (for hmm). Defines ideal_learner_prob_inference_from_outcomes function. Set to 20. 
    - 'conf_type': string, parameter defining ideal_learner_prob_conf_from_outcomes function. Options are "prev-conf", "prev-conf-pred", "new-conf", "new-conf-pred". The "-pred" options accountfor volatility in the estimate. "Prev-conf" refers to prior variance while "new-conf" refers to posterior variance. 

     Returns:
    -  computational_var_df: pd.DataFrame, contains computed variables from the ideal observer model. With this function, 'prior_dist' (the prior distribution mean), 'prior_conf' (the variance in the prior distribution) and the 'cpp' (change point probability) are added as columns for each task session. Also contains a variable that categorises the prior_conf and cpp based on the interquartile range across all task sessions (used for basic analysis).
    """
    for task_session, session_data in computational_var_df.groupby('task_session'):
        seq = np.array(session_data['outcome'])

        # Run functions to compute prior parameters, confidence and change point probability. Add data to existing data frame computational_var_df  
        prior_dist = ideal_learner_prob_estimates_from_outcomes(
            seq, p_c, estimate_type=estimate_type, p1_min=p1_min, p1_max=p1_max
        )
        prior_conf = ideal_learner_prob_confidence_from_outcomes(
            seq, p_c, conf_type=conf_type, p1_min=p1_min, p1_max=p1_max
        )
        cpp = ideal_learner_prob_cpp(seq, p_c, p1_min=p1_min, p1_max=p1_max)

        # Add new columns to computational_var_df
        session_indices = session_data.index
        computational_var_df.loc[session_indices, 'prior_dist'] = prior_dist
        computational_var_df.loc[session_indices, 'prior_conf'] = prior_conf
        computational_var_df.loc[session_indices, 'cpp'] = cpp

    # Categorizing prior confidence and change point probability into quartiles for basic analysis
    # Prior confidence
    conf_percent_25 = computational_var_df['prior_conf'].quantile(0.25)
    conf_percent_50 = computational_var_df['prior_conf'].quantile(0.50)  
    conf_percent_75 = computational_var_df['prior_conf'].quantile(0.75)

    # Initialize a default category
    computational_var_df['prior_conf_cat'] = None

    # Assign categories based on quartile boundaries
    computational_var_df.loc[computational_var_df['prior_conf'] < conf_percent_25, 'prior_conf_cat'] = 'first_quartile'
    computational_var_df.loc[
        (computational_var_df['prior_conf'] >= conf_percent_25) & (computational_var_df['prior_conf'] < conf_percent_50),
        'prior_conf_cat'
    ] = 'second_quartile'
    computational_var_df.loc[
        (computational_var_df['prior_conf'] >= conf_percent_50) & (computational_var_df['prior_conf'] < conf_percent_75),
        'prior_conf_cat'
    ] = 'third_quartile'
    computational_var_df.loc[computational_var_df['prior_conf'] > conf_percent_75, 'prior_conf_cat'] = 'fourth_quartile'

    # Change point probability
    cpp_percent_25 = computational_var_df['cpp'].quantile(0.25)
    cpp_percent_50 = computational_var_df['cpp'].quantile(0.50)  
    cpp_percent_75 = computational_var_df['cpp'].quantile(0.75)

    # Initialize a default category
    computational_var_df['cpp_cat'] = None

    # Assign categories based on quartile boundaries
    computational_var_df.loc[computational_var_df['cpp'] < cpp_percent_25, 'cpp_cat'] = 'first_quartile'
    computational_var_df.loc[
        (computational_var_df['cpp'] >= cpp_percent_25) & (computational_var_df['cpp'] < cpp_percent_50),
        'cpp_cat'
    ] = 'second_quartile'
    computational_var_df.loc[
        (computational_var_df['cpp'] >= cpp_percent_50) & (computational_var_df['cpp'] < cpp_percent_75),
        'cpp_cat'
    ] = 'third_quartile'
    computational_var_df.loc[computational_var_df['cpp'] > cpp_percent_75, 'cpp_cat'] = 'fourth_quartile'

    # Options to plots the the prior mean, prior confidence and change point probability
    if plot_distribution_mean:
        unique_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['prior_dist'], mode='lines', name=f'Prior distribution mean - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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
        unique_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['prior_conf'], mode='lines', name=f'Prior confidence - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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
        unique_sessions = computational_var_df['task_session'].unique()

        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                    subplot_titles=[f'Task Session {session}' for session in unique_sessions])

        for i, session in enumerate(unique_sessions, start=1):
            session_data = computational_var_df[computational_var_df['task_session'] == session]
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['cpp'], mode='lines', name=f'Change point probability - Session {session}'), row=i, col=1)

            change_points = change_point_indices_df[change_point_indices_df['task_session'] == session]['outcome_num']
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

    return computational_var_df



def behav_v_model_linear_analysis(computational_var_df, participant_str, condition_str, model_var, behav_var, exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=False):
    """
    Function to compute the correlation and optionally perform regression analysis 
    between two variables in a DataFrame.

    Parameters:
    - computational_var_df: pd.DataFrame, the dataset containing the variables to be analysed.
    - participant_str: string, participant identifier.
    - condition_str: string, identifier of the condition.
    - model_var: string, name of the column to be used as the predictor variable (x-axis).
    - behav_var: string, name of the column to be used as the outcome variable (y-axis).
    - exclude_zero: boolean, if true, will exclude rows where x_col is 0.
    - plot_behav_v_model_scatter: boolean, if true, will plot the scattergram between the two variables.
    - add_regression: boolean, if true, will compute and plot the regression line.

    Returns:
    - behav_v_model_corr: float, correlation coefficient between the two specified behavioual/model variables.
    - behav_v_model_regress: dict, containing slope, intercept, r value, p value, and std error of the regression analysis between behavioural and model variables (if add_regression is True).
    """
    # Get initial number of rows
    initial_rows = len(computational_var_df)

    # Drop rows with missing values in the columns of interest
    cleaned_df = computational_var_df[[model_var, behav_var, "task_session"]].dropna()

    # Print the number of rows removed due to missing values
    removed_missing = initial_rows - len(cleaned_df)
    print(f"Removed {removed_missing} rows due to missing values.")

    # Optionally exclude rows where the behavioural variable shows no update (i.e.) value is 0).
    if exclude_zero:
        before_zero_filter = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df[behav_var] != 0]
        removed_zero = before_zero_filter - len(cleaned_df)
        print(f"Removed {removed_zero} rows where {behav_var} was 0.")
    
    # Compute the correlation coefficient
    behav_v_model_corr = cleaned_df[model_var].corr(cleaned_df[behav_var])
    print(f"The correlation coefficient between {model_var} and {behav_var} for participant {participant_str} is: {behav_v_model_corr}")
    
    behav_v_model_regress = None

    # Optionally perform regression
    if add_regression:
        regression = linregress(cleaned_df[model_var], cleaned_df[behav_var])
        behav_v_model_regress = {
            "slope": regression.slope,
            "intercept": regression.intercept,
            "r_value": regression.rvalue,
            "p_value": regression.pvalue,
            "std_err": regression.stderr
        }
        print(f"Linear regression results: {behav_v_model_regress}")
    
    # Optionally plot
    if plot_behav_v_model_scatter:
        df= cleaned_df.copy()
        df['task_session'] = df['task_session'].astype('category')

        fig = px.scatter(df, x=model_var, y=behav_var, color="task_session", symbol="task_session")

        fig.update_layout(
        title_text=f"Ideal observer model by behavioural response for participant {participant_str} across all task sessions in condition {condition_str}.",
        showlegend=True,
        xaxis_title=model_var,
        yaxis_title=behav_var
        )

        if add_regression:
            x_vals = np.linspace(cleaned_df[model_var].min(), cleaned_df[model_var].max(), 100)
            y_vals = behav_v_model_regress["slope"] * x_vals + behav_v_model_regress["intercept"]
            fig.add_scatter(x=x_vals, y=y_vals, mode="lines", name="Regression Line")

        fig.update_layout(
            title_text=f"Scatterplot of {behav_var} derived from behavioural and ideal observer for participant {participant_str} in condition {condition_str}",
            xaxis_title=model_var,
            yaxis_title=behav_var,
            showlegend=True
        )
        fig.show()
    
    return behav_v_model_corr, behav_v_model_regress



def uncertainty_on_learning_rate_participant_level(participant_str, condition_str, computational_var_df, n_bins = 6, parameter_estimate = 'estimate', exclude_zero = True, plot_corr_subplots = True):
    """
    Function to compute correlations and regression parameters for prior confidence and change point probability on learning rate. Returns analysis outputs and model parameters. 

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - n_bins: float, number of bins the prior confidence variable shuld be split into. 
    - parameter_estimate: string, column identifier of which parameter to use (can be 'estimate' for participants' estimates, or 'ideal_observer' for model estimates).
    - exclude_zero: boolean, if true, will remove row in which no updates made (learning rate = 0) 
    - plot_corr_subplots: boolean, if true, will plot the output of the correlations and linear regression weights assessing the association between parameters of uncertainty and learning rate. 

    Returns:
    - learning_rate_analysis_dict: dict, contains the a cleaned dataframe used for the analysis (with z-scores calculated for variables of interest), binned output and output coefficients of correlation and regression analysis.
    """
    # Calculating zscore for computational parameters used in analysis
    scaler = StandardScaler()

    computational_var_df[f'lr_zscore_{parameter_estimate}'] = scaler.fit_transform(computational_var_df[[f'cleaned_learning_rate_{parameter_estimate}']]).flatten()
    computational_var_df['conf_zscore'] = scaler.fit_transform(computational_var_df[['prior_conf']]).flatten()

    computational_var_df['cpp_zscore'] = scaler.fit_transform(computational_var_df[['cpp']]).flatten()

    # Removing empty rows due to cleaning of learning rate
    df_clean = computational_var_df.copy() 
    df_clean = df_clean.dropna(subset=[f'lr_zscore_{parameter_estimate}'])

    # Remove rows where no update was present
    if exclude_zero:
        before_zero_filter = len(df_clean)
        df_clean = df_clean[df_clean[f'cleaned_learning_rate_{parameter_estimate}'] != 0]
        removed_zero = before_zero_filter - len(df_clean)
        print(f"Removed {removed_zero} rows where 'cleaned_learning_rate_{parameter_estimate}' was 0.")

    # Correlation between parameters and learning rate
    conf_lr_corr, _ = pearsonr(x = df_clean['conf_zscore'], y = df_clean[f'lr_zscore_{parameter_estimate}'])
    cpp_lr_corr, _ = pearsonr(x = df_clean['cpp_zscore'], y = df_clean[f'lr_zscore_{parameter_estimate}'])

    # Binned asscoiation between prior confidence and learning rate
    n_bins = 6
    _, bin_edges = pd.qcut(df_clean['conf_zscore'], n_bins, retbins=True)

    binned_data = {'conf_avg': [], 'lr_avg': [], 'lr_sem': []}

    for i in range(len(bin_edges) - 1):
        mask = (df_clean['conf_zscore'] >= bin_edges[i]) & (df_clean['conf_zscore'] < bin_edges[i + 1])
        bin_conf = df_clean['conf_zscore'][mask]
        bin_lrs = df_clean[f'lr_zscore_{parameter_estimate}'][mask]
            
        # Calculate averages and SEM
        if len(bin_conf) > 0:
            binned_data['conf_avg'].append(np.mean(bin_conf))
            binned_data['lr_avg'].append(np.mean(bin_lrs))
            binned_data['lr_sem'].append(sem(bin_lrs))
        
    # Linear regression with prior confidence and change point probability predicting learning rate  
    X = df_clean[['conf_zscore', 'cpp_zscore']]
    y = df_clean[f'lr_zscore_{parameter_estimate}']

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    #  Extract Beta estimates (coefficients) and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Estimate odel fit (R²)
    r_squared = model.score(X, y)

    # Calculate Mean Squared Error (MSE) and RMSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Storing data in dictionary
    learning_rate_analysis_dict = {'uncert_lr_part_df': df_clean, 'binned_conf_lr_dict': binned_data, 'conf_lr_corr': conf_lr_corr, 'cpp_lr_corr': cpp_lr_corr, 'reg_coefs': coefficients, 'reg_intercept': intercept, 'reg_rsquared': r_squared, 'reg_mse': mse, 'reg_rsme': rmse}

    # Option to data across subplots
    if plot_corr_subplots:
        fig = make_subplots(
            rows=2, 
            cols=2, 
            shared_xaxes=False, 
            vertical_spacing=0.1, 
            subplot_titles=(
                "Correlation between learning rate and change point probability",
                "Correlation between learning rate and prior confidence", 
                "Regression weights for prior confidence and change point probability on learning rate", 
                "Binned relationship between learning rate and prior confidence"
            )
        )

        # Subplot 1.
        fig.add_trace(go.Scatter(x=df_clean['cpp_zscore'], y=df_clean[f'lr_zscore_{parameter_estimate}'], mode='markers'), row=1, col=1
        )

        # Subplot 2.  
        fig.add_trace(go.Scatter(x=df_clean['conf_zscore'], y=df_clean[f'lr_zscore_{parameter_estimate}'], mode='markers'), row=1, col=2)

        # Subplot 3.
        parameter_names = ['Prior Confidence (z-score)', 'Change Point Probability (z-score)']
        fig.add_trace(go.Bar(x=parameter_names, y=coefficients), row=2, col=1)

        # Subplot 4. 
        fig.add_trace(go.Scatter(
            x=binned_data['conf_avg'],
            y=binned_data['lr_avg'],
            mode='markers',
            name='Average learning rate',
            error_y=dict(
                type='data',
                array=binned_data['lr_sem'],
                visible=True
            )
        ), row=2, col=2)

        fig.update_layout(
            title=f"Association between computationally-derived parameters and learning rate for participant {participant_str} in condition {condition_str}",
            title_x=0.5,
            title_font=dict(size=16, family="Arial"),
            height=950,
            width=1800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="closest"
        )

        fig.update_xaxes(title_text="Change Point Probability (z-score)", row=1, col=1)
        fig.update_xaxes(title_text="Prior Confidence (z-score)", row=1, col=2)
        fig.update_xaxes(title_text="Prior Confidence (z-score)", row=2, col=2)

        fig.update_yaxes(title_text="Learning Rate (z-score)", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate (z-score)", row=1, col=2)
        fig.update_yaxes(title_text="Regression Coefficient", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate Average (z-score)", row=2, col=2)

        fig.show()

    return learning_rate_analysis_dict