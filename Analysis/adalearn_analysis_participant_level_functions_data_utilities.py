import pandas as pd
import numpy as np
import sys
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'



def check_trig_sync_between_files(participant_str, condition_str, directory_path, behavioural_df, pupil_trig_df, physiological_triggers_df, plot_outcomes_between_files = True):
    """
    Function to check that the number and order of triggers inserted matches across the different files.
    It aligns the triggers by 'task_session' and handles missing data by inserting NaN where necessary.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - directory_path: Path, directory path to the participant level of the data storage.
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. Variable of interest is 'outcome'. 
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session from pupil .txt file. Variable of interest is 'outcome'
    - physiological_triggers_df: pd.DataFrame, contains trigger data, associated timestamp, and data file number obtained from .edf file. Variable of interest is 'triggers'


    ##### Currently doesn't match due to mismatch in task sessions in participant 3 (due to skip of the first task session file due to size). Also need to convert bipoac triggers to binary based on channels triggers taken from ######
    """
    filtered_physiological_df = physiological_triggers_df.groupby('task_session').apply(lambda x: x.iloc[1:-1] if len(x) > 2 else x).reset_index(drop=True)

    combined_outcome_df = pd.DataFrame({
        'behavioural': behavioural_df['outcome'].reset_index(drop=True),
        'pupil': pupil_trig_df['outcome'].reset_index(drop=True),
        'physiological': filtered_physiological_df['outcome'].reset_index(drop=True),
        'task_session': behavioural_df['task_session'].reset_index(drop=True),
        'outcome_num': behavioural_df['outcome_num'].reset_index(drop=True)
    })
    combined_outcome_df['cum_outcome_num'] = combined_outcome_df.index
    
    combined_outcome_df['difference_flag'] = combined_outcome_df.apply(lambda x: len(set([x['behavioural'], x['pupil'], x['physiological']])) > 1, axis=1)

    if combined_outcome_df['difference_flag'].any():
        print(f"ERROR: Mismatch in outcomes derived from different files")
    else:
        print("MATCHED: Outcomes derived from different files are matched")
    
    combined_outcome_df_filled = combined_outcome_df.copy().fillna(-1)

    # Plot outcomes if requested
    if plot_outcomes_between_files:
        task_sessions = combined_outcome_df_filled['task_session'].unique()
        num_task_sessions = len(task_sessions)
        
        # Create subplots
        fig = make_subplots(rows=3, cols=num_task_sessions, subplot_titles=[f"Session {i+1}" for i in range(num_task_sessions)])

        color_map = {1: 'blue', 0: 'yellow', -1: 'red'}
        symbol_map = {1: 'circle', 0: 'circle', -1: 'cross'}

        # Plot each channel and task session
        for i, task_session in enumerate(task_sessions):
            session_data = combined_outcome_df_filled[combined_outcome_df_filled['task_session'] == task_session]

            for j, (col, label) in enumerate(zip(['behavioural', 'pupil', 'physiological'], ['Behavioural', 'Pupil', 'Physiological'])):
                valid_session_data = session_data[pd.notna(session_data[col])]
                colors = [color_map[val] for val in valid_session_data[col]]
                symbols = [symbol_map[val] for val in session_data[col]]
                
                fig.add_trace(
                    go.Scatter(
                        x=valid_session_data['cum_outcome_num'],
                        y=[0] * len(valid_session_data),  
                        mode='markers',
                        marker=dict(color=colors, symbol=symbols, size=10),
                        name=f"{label} {task_session}",
                    ),
                    row=j+1, col=i+1
                )

        fig.update_layout(height=900, width=900 * num_task_sessions, title_text=f"Outcome comparisons across files and task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)
        
        fig.update_yaxes(title_text="Behavioural Outcome", row=1, col=1)
        fig.update_yaxes(title_text="Pupil Outcome", row=2, col=1)
        fig.update_yaxes(title_text="Physiological Outcome", row=3, col=1)

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_triggers-across-different-file-types.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
        

        fig.show()

    return combined_outcome_df



def plot_session_behav_comp(participant_str, condition_str, directory_path, behavioural_df, computational_df, change_point_indices, task_session=1, turn_on_plotting = False):
    """
    Function to plot the behavioural and computational parameters from data of selected task sessions. 

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - directory_path: Path, directory path to the participant level of the data storage.
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables.
    - computational_df: pd.DataFrame, contains the computational-derived parameters of interest for each task session.
    - change_point_indices: pd.DataFrame, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session, and z-score used to categorises the obvious of the change point ('change_point_cat'). (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - task_session: float, the specific task session to analyze (default: 1)
    - turn_on_plotting: boolean, if True, will plot the subplots. Provides an option to turn off the plotting when collating the data across scripts.

    Returns:
    - None
    - Displays a Plotly figure with behavioural and computational subplots
    """
    # Select a task session from the dataframes
    task_session_behav_df = behavioural_df[behavioural_df['task_session'] == task_session].copy()
    task_session_comp_df = computational_df[computational_df['task_session'] == task_session].copy()
    task_session_cp = change_point_indices[change_point_indices['task_session'] == task_session].copy()

    if turn_on_plotting:
        fig = make_subplots(
            rows=3, 
            cols=2, 
            shared_xaxes=False, 
            vertical_spacing=0.1, 
            subplot_titles=(
                "Behavioural estimates (green), ideal observer estimates (red), hidden values (blue), and outcomes",
                "Behavioural learning rate", 
                "Ideal observer learning rate", 
                "Change point probability", 
                "Prior confidence",
                "Surprise"
            )
        )

        # Subplot 1: Behavioural and computational estimates
        fig.add_trace(
            go.Scatter(
                x=task_session_behav_df['outcome_num'],
                y=task_session_behav_df['P1'],
                mode='lines',
                line=dict(color='blue'),
                name='Hidden proportion'
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=task_session_behav_df['outcome_num'],
                y=task_session_behav_df['estimate'],
                mode='lines',
                line=dict(color='green'),
                name='Participant estimate'
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=task_session_behav_df['outcome_num'],
                y=task_session_behav_df['ideal_observer'],
                mode='lines',
                line=dict(color='red'),
                name='Ideal observer estimate'
            ), row=1, col=1
        )
        outcome_colors = ['blue' if val == 1 else 'yellow' for val in task_session_behav_df['outcome']]
        fig.add_trace(
            go.Scatter(
                x=task_session_behav_df['outcome_num'],
                y=[0] * len(task_session_behav_df['outcome']),
                mode='markers',
                marker=dict(color=outcome_colors),
                name='Outcomes'
            ), row=1, col=1
        )

        # Subplot 2: Behavioural learning rate
        fig.add_trace(
            go.Scatter(
                x=task_session_comp_df['outcome_num'], 
                y=task_session_comp_df['cleaned_learning_rate_estimate'], 
                mode='lines+markers', 
                name='Behavioural learning rate'
            ), row=1, col=2
        )
        for cp in task_session_cp['outcome_num']:
            fig.add_shape(
                type="line", 
                x0=cp, x1=cp, 
                y0=task_session_comp_df['cleaned_learning_rate_estimate'].min(), y1=task_session_comp_df['cleaned_learning_rate_estimate'].max(), 
                line=dict(color="grey", width=2, dash="dash"),
                xref='x2', yref='y2'
            )

        # Subplot 3: Ideal observer learning rate
        fig.add_trace(
            go.Scatter(
                x=task_session_comp_df['outcome_num'], 
                y=task_session_comp_df['cleaned_learning_rate_ideal_observer'], 
                mode='lines+markers', 
                name='Ideal observer learning rate'
            ), row=2, col=1
        )
        for cp in task_session_cp['outcome_num']:
            fig.add_shape(
                type="line", 
                x0=cp, x1=cp, 
                y0=task_session_comp_df['cleaned_learning_rate_ideal_observer'].min(), y1=task_session_comp_df['cleaned_learning_rate_ideal_observer'].max(), 
                line=dict(color="grey", width=2, dash="dash"),
                xref='x3', yref='y3'
            )

        # Subplot 4: Change point probability
        fig.add_trace(
            go.Scatter(
                x=task_session_comp_df['outcome_num'], 
                y=task_session_comp_df['cpp'], 
                mode='lines+markers', 
                name='Change point probability'
            ), row=2, col=2
        )
        for cp in task_session_cp['outcome_num']:
            fig.add_shape(
                type="line", 
                x0=cp, x1=cp, 
                y0=0, y1=task_session_comp_df['cpp'].max(), 
                line=dict(color="grey", width=2, dash="dash"),
                xref='x4', yref='y4'
            )

        # Subplot 5: Prior confidence
        fig.add_trace(
            go.Scatter(
                x=task_session_comp_df['outcome_num'], 
                y=task_session_comp_df['prior_conf'], 
                mode='lines+markers', 
                name='Prior confidence'
            ), row=3, col=1
        )
        for cp in task_session_cp['outcome_num']:
            fig.add_shape(
                type="line", 
                x0=cp, x1=cp, 
                y0=0, y1=task_session_comp_df['prior_conf'].max(), 
                line=dict(color="grey", width=2, dash="dash"),
                xref='x5', yref='y5'
            )

        # Subplot 6: Surprise
        fig.add_trace(
            go.Scatter(
                x=task_session_comp_df['outcome_num'], 
                y=task_session_comp_df['surprise'], 
                mode='lines+markers', 
                name='Surprise'
            ), row=3, col=2
        )
        for cp in task_session_cp['outcome_num']:
            fig.add_shape(
                type="line", 
                x0=cp, x1=cp, 
                y0=0, y1=task_session_comp_df['surprise'].max(), 
                line=dict(color="grey", width=2, dash="dash"),
                xref='x6', yref='y6'
            )

        # Update layout
        fig.update_layout(
            title=f"Behavioural and computationally-derived parameters obtained from task session {task_session} for participant {participant_str} in condition {condition_str}",
            title_x=0.5,
            title_font=dict(size=16, family="Arial"),
            height=950,
            width=1800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="closest"
        )

        # Add axis titles
        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_xaxes(title_text="Trial Number", row=1, col=2)
        fig.update_xaxes(title_text="Trial Number", row=2, col=1)
        fig.update_xaxes(title_text="Trial Number", row=2, col=2)
        fig.update_xaxes(title_text="Trial Number", row=3, col=1)
        fig.update_xaxes(title_text="Trial Number", row=3, col=2)
        fig.update_yaxes(title_text="Hidden Values / Estimates", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="Change Point Probability", row=2, col=2)
        fig.update_yaxes(title_text="Prior Confidence", row=3, col=1)
        fig.update_yaxes(title_text="Surprise", row=3, col=2)

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_task-session-behav-model-variables.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)


        fig.show()



def compute_avg_learning_rate_by_categories(computational_var_df, category_col, parameter_estimate):
    """
    Function to compute the average learning rate within computational parameters split into quartiles. 

    Parameters:
    - computational_var_df: pd.DataFrame, contains the computational-derived parameters of interest for each task session. Must have columns related to categories created for parameters of interest (e.g. 'surprise_cat').
    - category_col: string, column name of category of interest (e.g. 'surprise_cat').
    - parameter_estimate: string, options are 'estimate' or 'ideal_observer'. If 'estimate', the learning rate estimates from the participant are used. If 'ideal_observer', the learning rate estimates from the ideal observer are used.
    
    Returns:
    - avg_lr_dict: dictionary, contains the average learning rate for each quartile.
    """
    quartiles = ['first_quartile', 'second_quartile', 'third_quartile', 'fourth_quartile']

    avg_lr_dict = {}  
    
    for quartile in quartiles:
        subset = computational_var_df[computational_var_df[category_col] == quartile]
        avg_lr = np.nanmean(subset[f'cleaned_learning_rate_{parameter_estimate}'])
        avg_lr_dict[quartile] = avg_lr 
    
    return avg_lr_dict 



def evaluation_of_learning_rate(participant_str, condition_str, directory_path, behavioural_df, computational_var_df, change_point_indices, quantile_baseline = 0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation = True):
    """
    Function to compute various metrics that enable closer inspection of the learning rate estimates in response to change points and latent parameters of uncertainty.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - directory_path: Path, directory path to the participant level of the data storage.
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. Must have column related to distance from change point ('outcome_dist_to_cp'). 
    - computational_var_df: pd.DataFrame, contains the computational-derived parameters of interest for each task session. Must have columns related to learning rate estimates (e.g. 'cleaned_learning_rate_estimate') and whether updates occured (i.e. 'was_updated_estimate').
    - change_point_indices: pd.Dataframe, with columns 'cum_outcome_num', 'rel_time', and 'task_session', containing the indices and relative times of the change points for each session, and z-score used to categorises the obvious of the change point ('change_point_cat'). (*Note: Have removed identification of the first outcome of each task session as a new change point)
    - parameter_estimate: string, options are 'estimate' or 'ideal_observer'. If 'estimate', the learning rate estimates from the participant are used. If 'ideal_observer', the learning rate estimates from the ideal observer are used.
    - baseline_method: string, options are quantile_over_sessions or quantile_within_sessions. If quantile_over_sessions, the threshold distance from the change point is computed over all task sessions. If quantile_within_sessions, the threshold distance is computed within each task session.
    - plot_lr_evaluation: boolean, if true, will plot subplots depicting the average learning computed across various parameters. 

    Returns:
    - lr_evaluation_dict: dictionary, contains average learning rate values derived across various parameters (i.e. distance from change point, computational markers of uncertainty)
    """
    # Definitions of comparators for average difference computations
    close = [0, 7]
    far = [8, 15]
    before = [-2, -1]
    
    # Compute average update frequency and learning rate across task sessions
    avg_lr_task_sess_df = pd.DataFrame()

    for group, data in computational_var_df.groupby('task_session'):
        update_freq = np.mean(data['was_updated_estimate'])
        avg_lr = np.mean(data[f'cleaned_learning_rate_{parameter_estimate}'])

        evaluation_df = pd.DataFrame({'task_session': [group], 'update_freq': [update_freq], 'avg_lr': [avg_lr]})

        avg_lr_task_sess_df = pd.concat([avg_lr_task_sess_df, evaluation_df], ignore_index=True)


    update_freq_over_sessions = avg_lr_task_sess_df['update_freq'].mean()

    avg_lr_over_sessions = avg_lr_task_sess_df['avg_lr'].mean()

    # Computing the baseline trials in those defined as the quartile-furthest away from a change point
    if baseline_method == "quantile_over_sessions":
        thresh_dist = np.quantile(behavioural_df['outcome_dist_to_cp'], (1 - quantile_baseline))
        
        baseline_trial_indices = np.nonzero(behavioural_df['outcome_dist_to_cp'] > thresh_dist)[0]
       
        # Compute average learning rate for over_sessions method
        baseline_learning_rates = computational_var_df.iloc[baseline_trial_indices][f'cleaned_learning_rate_{parameter_estimate}']

        # Compute the mean
        baseline_lr = np.mean(baseline_learning_rates)

    if baseline_method == "quantile_within_sessions":
        baseline_trial_indices_list = []
        avg_lr_within_sessions_list = []

        for group, data in behavioural_df.groupby('task_session'):

            thresh_dist = np.quantile(data['outcome_dist_to_cp'], (1 - quantile_baseline))

            baseline_trial_indices = np.nonzero(data['outcome_dist_to_cp'] > thresh_dist)[0]
            
            baseline_trial_indices_list.append(baseline_trial_indices)
            
            # Compute average learning rate within each session and store
            avg_lr_within_sessions_list.append(computational_var_df.iloc[baseline_trial_indices][f'cleaned_learning_rate_{parameter_estimate}'].mean())

        baseline_lr = np.mean(avg_lr_within_sessions_list)

    # Compute the average learning rate for the close and far trials around change points
    # Identify change points
    change_indices = behavioural_df[behavioural_df['outcome_dist_to_cp'] == 0].index

    # Identify the first index of each task session
    first_indices_per_session = behavioural_df.groupby('task_session').apply(lambda x: x.index[0]).values

    # Filter out change points that are at the start of a task session
    filtered_change_indices = [idx for idx in change_indices if idx not in first_indices_per_session]

    # Store means per change point
    close_means = []
    far_means = []
    before_means = []

    for idx in filtered_change_indices:
        # Define window ranges
        close_window = computational_var_df[(idx <= computational_var_df.index) & (computational_var_df.index <= idx + close[1])]
        far_window = computational_var_df[(idx + far[0] <= computational_var_df.index) & (computational_var_df.index <= idx + far[1])]
        before_window = computational_var_df[(idx + before[0] <= computational_var_df.index) & (computational_var_df.index <= idx + before[1])]
        
        # Compute mean learning rates within each window
        close_mean = close_window[f'cleaned_learning_rate_{parameter_estimate}'].mean()
        far_mean = far_window[f'cleaned_learning_rate_{parameter_estimate}'].mean()
        before_mean = before_window[f'cleaned_learning_rate_{parameter_estimate}'].mean()
        
        # Store results
        close_means.append(close_mean)
        far_means.append(far_mean)
        before_means.append(before_mean)

    # Compute overall means
    overall_close_mean = np.nanmean(close_means)

    overall_far_mean = np.nanmean(far_means)

    overall_before_mean = np.nanmean(before_means)

    # Computing mean learning rate across quartiles of computational parameters 
    # Surprise
    surprise_avg_lr_dict = compute_avg_learning_rate_by_categories(computational_var_df=computational_var_df, category_col='surprise_cat', parameter_estimate=parameter_estimate)

    # Prior confidence
    conf_avg_lr_dict = compute_avg_learning_rate_by_categories(computational_var_df=computational_var_df, category_col='prior_conf_cat', parameter_estimate=parameter_estimate)

    # Change point probability
    cpp_avg_lr_dict = compute_avg_learning_rate_by_categories(computational_var_df=computational_var_df, category_col='cpp_cat', parameter_estimate=parameter_estimate)

    # Calculating average for obvious and non-obvious change points
    category = ['obvious', 'non-obvious']
    cp_cat_avg_lr_dict = {}

    for item in category:
        # Get relevant indices for the category
        relevant_indices = change_point_indices.loc[change_point_indices['change_point_cat'] == item].index

        learning_rates = []

        for idx in relevant_indices:
            # Get index range: current index + next 7 indices
            index_range = list(range(idx + close[0], idx + close[1]))

            # Extract learning rates from computational_var_df
            subset = computational_var_df.loc[computational_var_df.index.isin(index_range), f'cleaned_learning_rate_{parameter_estimate}']

            # Append valid learning rates
            learning_rates.extend(subset.dropna().values)

        # Compute and store the average learning rate for this category
        cp_cat_avg_lr_dict[item] = np.nanmean(learning_rates)


    # Storing all averages within a dictionary
    lr_evaluation_dict = {'avg_lr_session_df': avg_lr_task_sess_df, 'update_freq_mean': update_freq_over_sessions, 'lr_mean': avg_lr_over_sessions, 'baseline_lr': baseline_lr, 'lr_close_cp_mean': overall_close_mean, 'lr_far_cp_mean': overall_far_mean, 'lr_before_cp_mean': overall_before_mean, 'surprise_avg_lr_dict': surprise_avg_lr_dict, 'conf_avg_lr_dict': conf_avg_lr_dict, 'cpp_avg_lr_dict': cpp_avg_lr_dict, 'cp_cat_avg_lr_dict': cp_cat_avg_lr_dict}
    
    # Option to plot average learning rate across different parameters
    if plot_lr_evaluation:
        fig = make_subplots(
            rows=2, 
            cols=2, 
            shared_xaxes=False, 
            vertical_spacing=0.1, 
            subplot_titles=(
                "Average learning rate across all sessions and based on relative distance to change points",
                "Average learning rate across change point categories (Jepma et al., 2016)", 
                "Average learning rate across change point probability split by quartiles", 
                "Average learning rate across prior confidence split by quartiles"
            )
        )

        list_quartiles = ['First quartile', 'Second quartile', 'Third quartile', 'Fourth quartile']
        list_position_names = ['Average', 'Before CP', 'Close to CP', 'Far from CP', 'Baseline (q = 20%)']
        list_position_values = [
            lr_evaluation_dict.get('lr_mean', 0),
            lr_evaluation_dict.get('lr_before_cp_mean', 0),
            lr_evaluation_dict.get('lr_close_cp_mean', 0),
            lr_evaluation_dict.get('lr_far_cp_mean', 0),
            lr_evaluation_dict.get('baseline_lr', 0)
        ]

        # Subplot 1.
        fig.add_trace(go.Bar(x=list_position_names, y=list_position_values), row=1, col=1)

        # Subplot 2.
        fig.add_trace(go.Bar(x=category, y=tuple(lr_evaluation_dict['cp_cat_avg_lr_dict'].values())), row=1, col=2)
                      
        # Subplot 3. 
        fig.add_trace(go.Bar(x=list_quartiles, y=tuple(lr_evaluation_dict['cpp_avg_lr_dict'].values())), row=2, col=1)

        # Subplot 4.     
        fig.add_trace(go.Bar(x=list_quartiles, y=tuple(lr_evaluation_dict['conf_avg_lr_dict'].values())), row=2, col=2)

        fig.update_layout(
            title="Average learning rate across different parameters",
            title_x=0.5,
            title_font=dict(size=16, family="Arial"),
            height=950,
            width=1800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="closest"
        )

         # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_evaluation-of-learning-rate.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)



        fig.show()


    return lr_evaluation_dict





def save_dfs_to_csv(participant_str, condition_str,  directory_path, task = 'pupil', analysis_type = ['all']):
    """
    Function to save dataframes to .csv files to be loaded and used for group level analysis.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, path to the participant level directory
    - task: string, defines the task being analysed
    - analysis_type: list of strings, specifies which analyses to save ('pupil', 'computational', 'behavioural', etc.). If you want to save all you can simply put 'all'

    Returns:
    - Saves dataframes as dedicated .csv files in the participant level data directory
    """
    # Ensure the save directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Dictionary mapping variable names to their corresponding DataFrames
    dataframes = {
        # Behavioural analyses
        'behavioural': { 
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behavioural-df": behavioural_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mean-abs-error-df": mean_abs_error_df,
        },
        # Computational analyses
        'computational': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_computational-var-df": computational_var_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp": filtered_behav_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp": averaged_behav_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-model-lr-by-cp": filtered_model_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-model-lr-by-cp": averaged_model_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-estimate": behav_v_model_estimate_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp-zero-excluded": filtered_behav_learning_rate_by_cp_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp-zero-excluded": averaged_behav_learning_rate_by_cp_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-estimate": behav_v_model_estimate_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp": change_point_avg_lrs_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-corr": lr_around_cp_corr_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-zero-excluded": change_point_avg_lrs_df_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-corr-zero-excluded": lr_around_cp_corr_df_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-lr": behav_v_model_lr_df, 
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-lr-zero-excluded": behav_v_model_lr_df_zero_exclude, 
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-by-session": lr_evaluation_dict['avg_lr_session_df'],
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-df": learning_rate_analysis_dict['uncertainty_v_lr_df'], 
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-df-zero-excluded": learning_rate_analysis_dict_exclude_zero['uncertainty_v_lr_df'],
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-model-df": learning_rate_model_analysis_dict['uncertainty_v_lr_df'],
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_dkl-regression-analysis": dkl_regression_df,
        },
        # Pupil analyses
        'pupil': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-trig-df": pupil_trig_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df": outcome_epoched_pupil_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df": outcome_blink_ratio_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df-nobc": outcome_epoched_pupil_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df-nobc": outcome_blink_ratio_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-amplitude-around-cp": pupil_amp_around_cp_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-average-around-cp": pupil_avg_around_cp_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-baseline-around-cp": pupil_baseline_around_cp_df,
        },
        # Pupil v computational variables regression analyses
        'pupil_comp_regression': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef": conf_cpp_coef,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df": pupil_regress_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef-nobc": conf_cpp_coef_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df-nobc": pupil_regress_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-amplitude": mediation_output,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-average": mediation_output_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-amplitude-zero-excluded": mediation_output_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-average-zero-excluded": mediation_output_nobc_zero_excluded,
        },
    }

    # Select the correct analysis type
    if analysis_type == "all":
        selected_dataframes = {k: v for category in dataframes.values() for k, v in category.items()}
    elif analysis_type in dataframes:
        selected_dataframes = dataframes[analysis_type]
    else:
        print(f"Invalid analysis_type '{analysis_type}'. Choose from 'pupil', 'computational', 'behavioural', etc. or 'all'.")
        return

    # Save each DataFrame to a CSV file
    save_dir = os.path.join(directory_path, f"ses-{condition_str}", "csv_files")

    os.makedirs(save_dir, exist_ok=True)

    for name, df in selected_dataframes.items():
        if isinstance(df, dict):
            print(f"Skipping {name} (dict object)")
            continue
        file_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")