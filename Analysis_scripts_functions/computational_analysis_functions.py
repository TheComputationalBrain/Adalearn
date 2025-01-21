import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference
from model_learner import ideal_learner_prob_estimates_from_outcomes, ideal_learner_prob_inference_from_outcomes, ideal_learner_prob_confidence_from_outcomes, ideal_learner_prob_cpp



def compute_learning_rate(behavioural_df, groupby_parameter = None, estimate_parameter=None):
    """
    Function to compute learning rate for each outcome presentation across groups defined 
    by a specific parameter (e.g., task session). 
    Learning rate (αt) is derived from the delta rule:

                            αt = (vt - vt-1) / (xt - vt-1)

    where:
        - vt is the current estimate at time t
        - vt-1 is the previous estimate at time t-1
        - xt is the observed outcome at time t (e.g., 1 for blue and 0 for yellow).

    The function groups takes a dataframe which contains and 'outcome' and some estimate parameter. If there is a grouping variable column (e.g., 'task_session'), the function computes the learning rate for each group based on the provided estimate parameter. Returns the updated dataframe with a new 'learning_rate' column.

    Parameters:
    - behavioural_df: pd.DataFrame, the input dataframe containing the estimates and outcomes.
    - groupby_parameter: string, the column name by which to group the dataframe (e.g., 'task_session').        
    - estimate_parameter: string, the column name representing the estimates from which the learning rate is to be derived. It can be a participant's estimate or the ideal observer model estimate.

    Returns:
    - behavioural_df: pd.DataFrame, the input dataframe with an added 'learning_rate' column, containing the computed learning rates for each outcome presentation (within each group).
    """
    learning_rate_col = f'learning_rate_{estimate_parameter}'

    # Initialize the learning rate column with NaN
    behavioural_df[learning_rate_col] = float('nan')
    
    # Check if grouping is needed. If yes, group the dataframe by the specified parameter
    if groupby_parameter and groupby_parameter in behavioural_df.columns:
        grouped = behavioural_df.groupby(groupby_parameter)

        # Iterate through each group to compute the learning rate
        for _, group in grouped:
            for i in range(1, len(group)):
                vt = group.iloc[i][estimate_parameter]
                vt_minus_1 = group.iloc[i-1][estimate_parameter]
                xt = group.iloc[i]['outcome']
                                
                behavioural_df.at[group.index[i], learning_rate_col] = (vt - vt_minus_1) / (xt - vt_minus_1)
            
    else:
        # If no grouping is required, compute learning rate across the entire DataFrame
        for i in range(1, len(behavioural_df)):
            vt = behavioural_df.iloc[i][estimate_parameter]
            vt_minus_1 = behavioural_df.iloc[i-1][estimate_parameter]
            xt = behavioural_df.iloc[i]['outcome']
            
                
            # Assign the learning rate
            behavioural_df.at[behavioural_df.index[i], learning_rate_col] = (vt - vt_minus_1) / (xt - vt_minus_1)

    return behavioural_df



def behavioural_learning_rate_analysis(behavioural_df, change_point_indices_df, participant_str, condition_str, plot_learning_rate_by_ses = False):
    """ 
    Function to calculate the behavioural learning rate based on participants estimates following each outcome presentation. Returns behavioural dataframe with new learning rate variable created. Option to plot learning rate across each task session. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables.
    - change_point_indices_df: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. Used to mark the change points on plots (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - plot_learning_rate_by_ses: boolean, if true, will plot the calculated (and filtered) learning rate across each task session. Change points added as dotted horizontal lines for visualisation (default = false).

    Returns:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable. Final data frame contains a "cleaned learning rate", which removes values that exceed set thresholds (-0.6 and +1.3, which were determined a priori from a pilot study conducted before Foucault & Meyniel, 2024. Thresholding used to to avoid aberrant learning rate values due to very small denominators (error: estimate - outcome)).      
    """
    # Implementing learning rate equation
    behavioural_df = compute_learning_rate(behavioural_df, groupby_parameter = 'task_session', estimate_parameter='estimate') 

    behavioural_df['cleaned_learning_rate_estimate'] = behavioural_df['learning_rate_estimate'].where((behavioural_df['learning_rate_estimate'] >= -0.6) & (behavioural_df['learning_rate_estimate'] < 1.3), float('nan'))

    removed_data_points = behavioural_df['learning_rate_estimate'].notna().sum() - behavioural_df['cleaned_learning_rate_estimate'].notna().sum()
    print(f"Number of data points removed in cleaning step: {removed_data_points}")

    # Plotting learning rate data
    # Create subplots, one for each task session
    if plot_learning_rate_by_ses:
        unique_sessions = behavioural_df['task_session'].unique()
    
        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                        subplot_titles=[f'Beahvioural learning rate in task session {session}' for session in unique_sessions])

        # Add learning rate traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = behavioural_df[behavioural_df['task_session'] == session]
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

    
    return behavioural_df



def ideal_observer_learning_rate_analysis(behavioural_df, change_point_indices_df, participant_str, condition_str, plot_learning_rate_by_ses = False):
    """ 
    Function to calculate the learning rate based on the ideal observer model estimates following each outcome presentation. Returns behavioural dataframe with new learning rate variable created. Option to plot learning rate across each task session. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables.
    - change_point_indices_df: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session. Used to mark the change points on plots (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - plot_learning_rate_by_ses: boolean, if true, will plot the calculated (and filtered) learning rate across each task session. Change points added as dotted horizontal lines for visualisation (default = false).

    Returns:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable. Final data frame contains a "cleaned learning rate", which removes values that exceed set thresholds (-0.6 and +1.3, which were determined a priori from a pilot study conducted before Foucault & Meyniel, 2024. Thresholding used to to avoid aberrant learning rate values due to very small denominators (error: estimate - outcome)).      
    """
    # Implementing learning rate equation
    behavioural_df = compute_learning_rate(behavioural_df, groupby_parameter = 'task_session', estimate_parameter='ideal_observer') 

    behavioural_df['cleaned_learning_rate_ideal_observer'] = behavioural_df['learning_rate_ideal_observer'].where((behavioural_df['learning_rate_ideal_observer'] >= -0.6) & (behavioural_df['learning_rate_ideal_observer'] < 1.3), float('nan'))

    removed_data_points = behavioural_df['learning_rate_ideal_observer'].notna().sum() - behavioural_df['cleaned_learning_rate_ideal_observer'].notna().sum()
    print(f"Number of data points removed in cleaning step: {removed_data_points}")

    # Plotting learning rate data
    # Create subplots, one for each task session
    if plot_learning_rate_by_ses:
        unique_sessions = behavioural_df['task_session'].unique()
    
        fig = make_subplots(rows=len(unique_sessions), cols=1, shared_xaxes=True,
                        subplot_titles=[f'Ideal observer model learning rate in task session {session}' for session in unique_sessions])

        # Add learning rate traces to each subplot
        for i, session in enumerate(unique_sessions, start=1):
            session_data = behavioural_df[behavioural_df['task_session'] == session]
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

    
    return behavioural_df



def learning_rate_by_change_point(behavioural_df, change_point_indices,participant_str, condition_str, learning_rate_parameter = 'estimate', pre=2, post=10, plot_learning_rate_by_cp = False):
    """ 
    Function to calculate the average learning rate around change points. Uses identified change point indices to derive learning rate dynamics before and after a change in task structure. Stores responses across all task sessions and then plots the average (plus standard deviation) learning rate. 

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables with the new addition of learning rate variable.
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
    filtered_learning_rate_by_cp = pd.merge(expanded_change_points_df, behavioural_df, 
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





def compute_surprise_from_observations(participant_str, condition_str, behavioural_df, change_point_indices_df, plot_surprise = False, plot_posterior_conf = False, plot_posterior_dist_means = False):
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
    - behavioural_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 
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

    # Initialize lists to store task session data
    task_session_surprise_df = []

    # Loop over task sessions and convert the outcome column to a sequence (np.array)
    for task_session, session_data in behavioural_df.groupby('task_session'):
        seq = np.array(session_data['outcome'])

        # Run inference to compute surprise for the sequence of observations
        output = run_inference(seq, options=options)

        # Creating dataframe to store for that task session
        surprise_values = output['surprise']
        session_surprise_df = pd.DataFrame({'surprise': surprise_values})
        session_surprise_df['task_session'] = task_session
        session_surprise_df['outcome'] = seq
        session_surprise_df['outcome_num'] = np.arange(0, len(session_surprise_df))
        session_surprise_df['model_post_prob_1'] = output[(1,)]['mean']
        session_surprise_df['model_post_prob_0'] = output[(0,)]['mean']
        session_surprise_df['model_post_prob_var'] = output[(1,)]['SD']
        task_session_surprise_df.append(session_surprise_df)
         
    computational_var_df = pd.concat(task_session_surprise_df, ignore_index=True)
    computational_var_df['cum_outcome_num'] = computational_var_df.index

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



def compute_prior_conf_and_cpp_from_observations(participant_str, condition_str, computational_var_df, behavioural_df, change_point_indices_df, p_c, p1_min, p1_max, estimate_type, conf_type, plot_distribution_mean = False, plot_prior_conf = False, plot_cpp = False):
    """
    Function computes the "confidence" in the prior and change point probability from each set of task session observations. Computes "confidence" and change point probability using functions within "model_learner" (created by Cedric Foucault and Florent Meyniel: https://github.com/florentmeyniel/Ada-Learn/blob/main/analysis). "Confidence" is the -log(sd) of the prior distribution, while change point probability is calculated from the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring, divided by the conditional probability of an observation (at time t) given a change point has occurred multiplied by the true probability of a change point occurring plus the conditional probability of an observation (at time t) given no change point has occurred multiplied by the true probability that a change point did not occur:

                                    p(cp | X_t) = p(X_t | cp) p(cp) / (p(X_t | cp) p(cp) + p(X_t | no cp) (1-p(cp)))

    This function takes the ada_learn data, converts the observations to a sequence for each task session, before returning a separate dataframe for prior "confidence" and change point probability.

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)  
    - condition: string, identifier of condition order (set to 2 digits) ** May need to change to denote actual condition in Adalearn VNS study **
    - computational_var_df: pd.DataFrame, contains computed variables from the ideal observer model.
    - behavioural_df: pd.DataFrame, contains the behavioural outputs of the adalearn data files for a given condition. 
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
    # Lists to store computed parameters for each task session
    prior_dist_df_list = []
    prior_conf_df_list =[]
    cpp_df_list =[]

    for task_session, session_data in behavioural_df.groupby('task_session'):
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
            fig.add_trace(go.Scatter(x=session_data['outcome_num'], y=session_data['mean'], mode='lines', name=f'Prior distribution mean - Session {session}'), row=i, col=1)

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