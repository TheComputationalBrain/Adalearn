import pandas as pd
import os
import numpy as np
from scipy import stats
import pingouin
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'



def safe_adjust_index(idx, max_idx):
    """
    Function to safely adjust a 1-based index to 0-based, ensuring it doesn't become negative or exceed bounds.

    Parameters:
    - idx: int, the index to adjust, typically from cluster permutation output (may be 1-based or include 0).
    - max_idx: int, the maximum valid index for the target data (e.g., length of a DataFrame).

    Returns:

    - int: A valid 0-based index within the range [0, max_idx - 1].

    """
    if idx == 0:
        return 0
    
    return min(idx - 1, max_idx - 1)



def extract_cluster_times(cluster_df, time_series, alpha=0.05):
    """
    A function to extract the start and end times for the longest significant cluster from a cluster permutation result. Filters clusters by statistical significance, computes their durations, selects the longest one, and returns the corresponding time values from the time series.

    Parameters:
    
    - cluster_df: pd.DataFrame, DataFrame containing cluster information. Must include 'cluster_start', 'cluster_end', and 'p_value' columns.
    - time_series: pd.Series, time values associated with the rows of the data used in the analysis.
    - alpha: float, optional significance threshold for filtering clusters (default is 0.05).

    Returns:

    - tuple of (float, float) or None: The (start_time, end_time) of the longest significant cluster. Returns None if no clusters are significant.
    """
    # Filter for significant clusters
    significant_clusters = cluster_df[cluster_df['p_value'] < alpha].copy()
    if significant_clusters.empty:
        return None  # or raise warning

    # Calculate duration and select longest cluster
    significant_clusters['duration'] = significant_clusters['cluster_end'] - significant_clusters['cluster_start']
    longest_cluster = significant_clusters.loc[significant_clusters['duration'].idxmax()]

    max_idx = len(time_series)

    # Adjust indices safely
    start_idx = safe_adjust_index(longest_cluster['cluster_start'], max_idx)
    end_idx = safe_adjust_index(longest_cluster['cluster_end'], max_idx)

    return time_series.iloc[start_idx], time_series.iloc[end_idx]



def compute_peristim_mean_sem_by_count(df, value_cols, time_col='time', group_col='count', sort_col='participant'):
    """
    Function to calculate the mean and SEM for specified columns following peristimulus regression of pupil epochs,  grouped by a counter variable.

    Parameters: 

    - pupil_regress_coef_df : pd.DataFrame, input DataFrame with regression coefficients across the epoch for each participant.
    - value_cols : list of strings, columns for which to compute the mean and SEM.
    - time_col : str, (optional) name of the column representing time (default is 'time').
    - group_col : str, (optional) column to group by for epoch-level averaging (default is 'count').
    - sort_col : str, (optional) column to sort by within each group (default is 'participant').

    Returns:
    
    - pd.DataFrame: A DataFrame containing the group-level means and SEMs for each value column, along with average time.
    """
    result = {col: [] for col in value_cols}
    result.update({f"{col}_error": [] for col in value_cols})
    result['time'] = []

    for count, group in df.groupby(group_col):
        group = group.sort_values(by=sort_col)

        for col in value_cols:
            result[col].append(group[col].mean())
            result[f"{col}_error"].append(stats.sem(group[col], nan_policy='omit'))

        result['time'].append(group[time_col].mean())

    return pd.DataFrame(result)



def pupil_response_around_change_point(directory_path, participant_str, condition_str, outcome_epoched_pupil_df, change_point_indices, baseline_correct_window = [], nobc_window = [], before = 6, after = 6, estimate_type = 'mean', plot_pupil_around_change_point = True):
    """
    Function to plot pupil responses to stimuli around change points. Pupil responses are the mean of the baseline corrected data (the mean is calculated across the entire epoch), the amplitude of the baseline corrected data (the amplitude is calculated as the max - min after stimulus presentation), and the baseline average of the uncorrected data (the average before stimulus presentation). CAUTION: Requires attention that the correct data is matched to the appropriate estimate type. Returns a dataframe with the mean (and variance; std, sem) of the response around the change in the task hidden structure.

    Parameters:
    - directory_path: Path, path to the participant level directory
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - outcome_epoched_pupil_df: pd.DataFrame, containing transposed epoched data (each column represents one epoch), the calculated mean, standard deviation and sem across all outcome defined epochs. Includes the timestamp.
    - change_point_indices: pd.DataFrame, with columns 'index', 'relative_time', and 'task_session', containing the indices and relative times of the change points for each session. (*Note: Have removed identification of the first outcome of each task session as a new change point).
    - baseline_correct_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the baseline corrected pupil signal.
    - nobc_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the nonbaseline corrected pupil signal.  
    - before: float, defines the number of outcomes examined before the change point
    - after: float, defines the number of outcomes examined after the change point
    - estimate_type: string: can be 'baseline', 'mean', 'amplitude', 'baseline_corrected_effect". Defines the function performed on the pupil data. Attention should be paid that the appropriate function is matched to the examined data (i.e. baseline vs. non-baseline corrected)
    - plot_around_change_point: boolean, if True, will plot the calculated pupil response around change points. 

    Returns:
    - pupil_around_cp_df: pd.DataFrame, contains calculated mean and estimates of variance for each outcome around the change points for each change point analysed.
   
    """
    # Reduce pupil epochs to the raw data and insert time as the index 
    outcome_epoched_pupil_df = outcome_epoched_pupil_df.copy()

    # Create a empty dictionary to store averages
    session_averages = {}

    # For loop, which takes the cumulative change point number and finds the corresponding column (and surrounding window) in the raw pupil data
    for session in change_point_indices['task_session'].unique():
        print(f"Task session: {session}")
        points = change_point_indices.loc[change_point_indices['task_session'] == session, 'cum_outcome_num']
        print(f'There are {len(points)} change points in session {session}. The change for session {session} are: {points.values}')

        point_ints = set(points.astype(int).values)  
        matching_cols = [col for col in outcome_epoched_pupil_df.columns if col in point_ints]
        print(f"Matching columns found: {matching_cols}")

        if matching_cols:
            for col in matching_cols:
                print(f"Analysisng column: {col}")
                
                col_idx = outcome_epoched_pupil_df.columns.get_loc(col)

                # Find the index of the *next* change point within the same session
                col_num = int(col)  # convert to outcome number for numeric comparison
                next_cp = points[points > col_num].min()  # find next change point (if any)

                # Compute the max post limit (either 'after' or before the next change point)
                if not pd.isna(next_cp):
                    # calculate how many trials until next CP
                    next_cp_distance = int(next_cp - col_num - 1)
                    max_after = min(after, next_cp_distance)
                else:
                    max_after = after  # no later change point → full window allowed

                # Define window bounds with this adjusted range
                start_idx = max(0, col_idx - before)
                end_idx = min(len(outcome_epoched_pupil_df.columns), col_idx + max_after + 1)

                # Subset the window
                window_columns = outcome_epoched_pupil_df.columns[start_idx:end_idx]

                window_data = outcome_epoched_pupil_df[window_columns]

                new_column_names = [f"{i - col_idx}" for i in range(start_idx, end_idx)]
                
                window_data.columns = new_column_names

                for column in window_data.columns:
                    if estimate_type == 'nonbaseline_corrected_effect':
                        filtered_pupil_data = window_data[column].loc[(window_data.index >= nobc_window[0]) & (window_data.index <= 0)]

                        pupil_variable = filtered_pupil_data.mean()

                        title_append = 'nonbaseline_corrected_effect_window'

                        if column not in session_averages:
                            session_averages[column] = []

                        session_averages[column].append(pupil_variable)

                    elif estimate_type == 'mean':
                        pupil_variable = window_data[column].mean()

                        title_append = 'mean'

                        if column not in session_averages:
                            session_averages[column] = []

                        session_averages[column].append(pupil_variable)

                    elif estimate_type == 'amplitude':
                        filtered_pupil_data = window_data[column].loc[(window_data.index >= 0.0) & (window_data.index <= 1.4)]

                        pupil_variable = filtered_pupil_data.mean()

                        title_append = f"amplitude"

                        if column not in session_averages:
                            session_averages[column] = []

                        session_averages[column].append(pupil_variable)

                    elif estimate_type == 'baseline_corrected_effect':
                        filtered_pupil_data = window_data[column].loc[(window_data.index >= baseline_correct_window[0]) & (window_data.index <= baseline_correct_window[1])]

                        pupil_variable = filtered_pupil_data.mean()

                        title_append = f"baseline_corrected_effect_window"

                        if column not in session_averages:
                            session_averages[column] = []

                        session_averages[column].append(pupil_variable)

                    else:
                        print(f'ERROR: mean, amplitude, baseline or baseline_corrected_effect must be specified as the estimate type for the analysis') 

                        title_append = ""

    # After processing all sessions, calculates the overall average, standard deviation and standard error of the mean for each column/stimulus around change point
    overall_averages = {col: np.mean(session_averages[col]) for col in session_averages}
    overall_stds = {col: np.std(session_averages[col], ddof=1) for col in session_averages}
    overall_sems = {col: overall_stds[col] / np.sqrt(len(session_averages[col])) for col in session_averages}

    # Store data in a dataframe
    pupil_around_cp_df = pd.DataFrame({
        "stimulus_rel_cp": list(overall_averages.keys()),  
        "mean": list(overall_averages.values()),
        "std": list(overall_stds.values()),
        "sem": list(overall_sems.values())
    })

    pupil_around_cp_df["stimulus_rel_cp"] = pd.to_numeric(pupil_around_cp_df["stimulus_rel_cp"])

    pupil_around_cp_df.sort_values("stimulus_rel_cp", inplace=True)

    # Option to plot the mean and variance for each stimulus around change point 
    if plot_pupil_around_change_point:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pupil_around_cp_df["stimulus_rel_cp"], 
            y=pupil_around_cp_df["mean"], 
            mode='lines+markers',
            name="Average pupil diameter mean"
        ))

        fig.add_trace(go.Scatter(x=pupil_around_cp_df["stimulus_rel_cp"],
            y=pupil_around_cp_df["mean"] + pupil_around_cp_df["sem"],
            mode='lines',
            line=dict(color='rgba(0, 0, 250, 0.1)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(x=pupil_around_cp_df["stimulus_rel_cp"],
            y=pupil_around_cp_df["mean"] - pupil_around_cp_df["sem"],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 0, 250, 0.2)',
            line=dict(color='rgba(0, 0, 250, 0.1)'),
            showlegend=False
        ))

        fig.update_layout(
            title=f"Pupil {title_append} response around change point for participant {participant_str} in condition {condition_str}",
            xaxis_title="Outcomes relative to change point",
            yaxis_title="Pupil diameter", 
            showlegend = False
        )

        # Save the figure to the partcipant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-{title_append}-response-around-cp.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)



        fig.show()

    return pupil_around_cp_df



def pupil_on_learning_rate_regression_analysis(participant_str, condition_str, directory_path, computational_var_df, outcome_epoched_pupil_df, outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove, baseline_correct_window = [], nobc_window = [], parameter_estimate = 'estimate', exclude_zero = False, plot_pupil_lr_regress = True):
    """
    Function to conduct regression analysis for baseline corrected and non-corrected pupil metrics on learning rate. Returns analysis outputs and model parameters. 

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, directory path to the participant level of the data storage.
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs for the baseline corrected pupil signal.
    - outcome_epoched_pupil_df_nobc: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs for the non-baseline corrected pupil signal.
    - baseline_correct_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the baseline corrected pupil signal.
    - nobc_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the nonbaseline corrected pupil signal.  
    - parameter_estimate = string, identifier specifiy whether participants' or model learning rates are to be used (default is participants = 'estimate'). String should be 'estimate' or 'ideal_observer' 
    - exclude_zero: boolean, if true, will remove row in which no updates made (learning rate = 0) (default = False)
    - plot_pupil_lr_regress: boolean, if true, will plot the output of the correlations and linear regression weights assessing the association between parameters of uncertainty and learning rate (default = True). 

    Returns:
    - pupil_v_lr_df: pd.DataFrame, contains the regression coefficients 
    """
    # Calculating zscore for computational parameters used in analysis

    computational_var_df[f'lr_zscore_{parameter_estimate}'] = zscore(computational_var_df[f'cleaned_learning_rate_{parameter_estimate}'], nan_policy='omit')
    
    df_clean = computational_var_df.copy() 

    # Based on the epochs containing too much interpolated data, remove respective rows                
    if outcome_epochs_to_remove is not None:
        outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
        df_clean.drop(index=outcomes_to_remove, inplace=True)
   
    # Preparing pupil data
    pupil_data_bc = outcome_epoched_pupil_df.copy()
    pupil_data_bc = pupil_data_bc.set_index('time')
    pupil_data_bc = pupil_data_bc.drop(columns=['mean', 'std', 'sem'])

    # Baseline corrected specify pupil data to be examined (the average after an outcome within the window of the peristimulus effect is used as a marker of phasic pupil activity.) 
    filtered_pupil_data_bc = pupil_data_bc.loc[(pupil_data_bc.index >= baseline_correct_window[0]) & (pupil_data_bc.index <= baseline_correct_window[1])]
    pupil_variable_bc = filtered_pupil_data_bc.mean()

    # The average of the non-baseline corrected pupil response before the stimulus (used as a marker of tonic activity)
    pupil_data_nobc = outcome_epoched_pupil_df_nobc.copy()
    pupil_data_nobc = pupil_data_nobc.set_index('time')
    pupil_data_nobc = pupil_data_nobc.drop(columns=['mean', 'std', 'sem'])

    filtered_pupil_data_nobc = pupil_data_nobc.loc[(pupil_data_nobc.index >= nobc_window[0]) & (pupil_data_nobc.index <= 0)]
    pupil_variable_nobc = filtered_pupil_data_nobc.mean()

    # Creating dataframe for regression 
    lr_regress_df = pd.DataFrame()
    lr_regress_df['learning_rate_zscore'] = df_clean[f'lr_zscore_{parameter_estimate}']

    lr_regress_df['pupil_response_bc'] = pupil_variable_bc.values 
    lr_regress_df['pupil_response_bc_zscore'] = zscore(lr_regress_df['pupil_response_bc'], nan_policy='omit')

    lr_regress_df['pupil_response_nobc'] = pupil_variable_nobc.values 
    lr_regress_df['pupil_response_nobc_zscore'] = zscore(lr_regress_df['pupil_response_nobc'], nan_policy='omit')

    title_addon = ''
    title = ''

    # Excluding zero update trials
    if exclude_zero:
        before_zero_filter = len(lr_regress_df)
        lr_regress_df['cleaned_learning_rate_estimate'] = df_clean['cleaned_learning_rate_estimate']
        lr_regress_df = lr_regress_df[lr_regress_df['cleaned_learning_rate_estimate'] != 0]
        removed_zero = before_zero_filter - len(lr_regress_df)
        print(f"Removed {removed_zero} rows where learning rate was 0.")
        # Re z-scoring the data after zero updates removed
        lr_regress_df['learning_rate_zscore'] = zscore(lr_regress_df['cleaned_learning_rate_estimate'], nan_policy='omit')

        title_addon = '-(zero-excluded)'
        title = '(zero excluded)'

    # Removing the null values in the cleaning of learning rate
    lr_regress_df = lr_regress_df.dropna(subset=['learning_rate_zscore'])

    # Setting variables for regression analysis
    X = lr_regress_df[['pupil_response_bc_zscore', 'pupil_response_nobc_zscore']].apply(pd.to_numeric, errors='coerce')  
    y = pd.to_numeric(lr_regress_df['learning_rate_zscore'], errors='coerce')

    # Add intercept term
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Extract coefficients and model fit statistics
    coefficients = model.params.iloc[1:].values  
    intercept = model.params.iloc[0]  
    r_squared = model.rsquared
    p_values = model.pvalues.values
    std_err_mean = model.bse.values  
    conf_int = model.conf_int().values  

    # Calculate MSE and RMSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Create a DataFrame for regression and correlation results
    pupil_v_lr_df = pd.DataFrame({
        "variable": ["Intercept"] + X.columns[1:].tolist(),  # Include predictor names
        "coefficient": [intercept] + coefficients.tolist(),
        "p_value": p_values.tolist(),
        "std_err": std_err_mean.tolist(),
        "ci_lower": conf_int[:, 0].tolist(),
        "ci_upper": conf_int[:, 1].tolist(),
    })

    # Plot regression weights for pupil metrics on learning rate
    if plot_pupil_lr_regress:
        fig = go.Figure()

        parameter_names = ['Pupil (corrected)', 'Pupil (non-corrected)']

        fig.add_trace(go.Bar(x = parameter_names, y=pupil_v_lr_df['coefficient'][1:], error_y=dict(type='data', array=pupil_v_lr_df['std_err'][1:])))

        fig.add_hline(y=0, line_dash='dot')

        fig.update_traces(
                    marker_color='rgba(0, 130, 150, 0.4)', 
                    marker_line_color='rgba(0, 130, 150, 0.8)', 
                    marker_line_width=1.0)

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black',  title_text="Regression weights")

        fig.update_layout(
                title=f'Regression weights for pupil metrics on learning rate {title}',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                font=dict(size=16),
                height=950,
                width=1800)
        
        # Save the figure to the participant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-on-learning-rate-regression{title_addon}.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)

        fig.show()


    return pupil_v_lr_df  



def pupil_uncertainty_lr_mediation_analysis(participant_str, condition_str, directory_path, computational_var_df, outcome_epoched_pupil_df, outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove, uncertainty_param, baseline_correct_window = [], nobc_window = [], covariates = [], exclude_zero=False):
    """
    Prepares a DataFrame for mediation analysis by selecting relevant computational variables and combining with pupil response, conditioned on whether baseline correction is applied. If baseline corrected data is offered as an input, the pupil response will be the pupil amplitude following stimuli presentation (i.e. max - min within the window 0 - 1.4 s). If not, the average pupil response on non-baseline corrected data will be used (the average is taken across the whole epoch). Returns the output of the mediation analysis.

    Parameters:

    - participant_str: string, participant identifier.
    - condition_str: string, identifier of the condition.
    - directory_path: Path, directory to participant level data 
    - computational_var_df: pd.DataFrame, the dataset containing the variables to be analysed.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs.
    - outcome_epoched_pupil_df_nobc: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs.
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within.
    - uncertainty_param: list, names of the predicting variable to be used in the mediation analysis.
    - baseline_correct_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the baseline corrected pupil signal.
    - nobc_window = list (floats), contains the start and end times of the cluster window for the longest significant cluster identified in the peristimulus regression on the nonbaseline corrected pupil signal. 
    - covariates: list, covariates incorporated within the mediation model.  
    - exclude_zero: boolean, if True, will exlcude values where no updates were made in learning rate. 

    Returns:
    - pd.DataFrame: pd.DataFrame, contains the output statistics of the mediation analysis.
    """
    # Checking that the correct integration of uncertainty param 
    if isinstance(uncertainty_param, list):
        raise ValueError("Only one uncertainty_param should be passed as a string.")

    # Preparing the data. Ensuring that the pupil data and "mediators" are matched in terms of the pupil epochs that can be used (do not excede the threshold of too much interpolated data)  
    predictors = computational_var_df.copy()
    outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
    predictors.drop(index=outcomes_to_remove, inplace=True)

    pupil_data_bc = outcome_epoched_pupil_df.copy()
    pupil_data_bc = pupil_data_bc.set_index('time')
    pupil_data_bc = pupil_data_bc.drop(columns=['mean', 'std', 'sem'])

    # Baseline corrected specify pupil data to be examined (the average after an outcome within the window of the peristimulus effect is used as a marker of phasic pupil activity.) 
    filtered_pupil_data_bc = pupil_data_bc.loc[(pupil_data_bc.index >=  baseline_correct_window[0]) & (pupil_data_bc.index <=  baseline_correct_window[1])]
    pupil_variable_bc = filtered_pupil_data_bc.mean()

    # The average of the non-baseline corrected pupil response before the stimulus (used as a marker of tonic activity)
    pupil_data_nobc = outcome_epoched_pupil_df_nobc.copy()
    pupil_data_nobc = pupil_data_nobc.set_index('time')
    pupil_data_nobc = pupil_data_nobc.drop(columns=['mean', 'std', 'sem'])

    filtered_pupil_data_nobc = pupil_data_nobc.loc[(pupil_data_nobc.index >= nobc_window[0]) & (pupil_data_nobc.index <= 0)]
    pupil_variable_nobc = filtered_pupil_data_nobc.mean()

    # Prepare variables for mediation analysis
    mediation_df = pd.DataFrame()
    mediation_df[uncertainty_param] = predictors[uncertainty_param].values
    mediation_df[f'{uncertainty_param}_zscore'] = zscore(mediation_df[uncertainty_param])

    mediation_df['learning_rate'] = predictors['cleaned_learning_rate_estimate']
    mediation_df['learning_rate_zscore'] = zscore(mediation_df['learning_rate'], nan_policy='omit')

    mediation_df['pupil_response_bc'] = pupil_variable_bc.values 
    mediation_df['pupil_response_bc_zscore'] = zscore(mediation_df['pupil_response_bc'], nan_policy='omit')

    mediation_df['pupil_response_nobc'] = pupil_variable_nobc.values 
    mediation_df['pupil_response_nobc_zscore'] = zscore(mediation_df['pupil_response_nobc'], nan_policy='omit')

    # Optional covariates
    if covariates:
        for covar_name in covariates:
            mediation_df[covar_name] = predictors[covar_name].values
            mediation_df[f'{covar_name}_zscore'] = zscore(mediation_df[covar_name], nan_policy='omit')
        covar_cols = [f'{c}_zscore' for c in covariates]
    else:
        covar_cols = None

    # Option to exclude no update trials
    if exclude_zero:
        before_zero_filter = len(mediation_df)
        mediation_df = mediation_df[mediation_df['learning_rate'] != 0]
        removed_zero = before_zero_filter - len(mediation_df)
        print(f"Removed {removed_zero} rows where learning rate was 0.")
        # Re z-scoring the data after zero updates removed
        mediation_df['learning_rate_zscore'] = zscore(mediation_df['learning_rate'])

    print(f'length of mediators before removing NaN values: {len(mediation_df)}')
    mediation_df = mediation_df.dropna(subset=['learning_rate_zscore'])
    print(f'length of mediators after removing Nan: {len(mediation_df)}')

    # Run mediation analysis (currently no option to exclude no update trials)
    mediation_analysis = pingouin.mediation_analysis(data=mediation_df, x=f'{uncertainty_param}_zscore', m=['pupil_response_nobc_zscore'], y='learning_rate_zscore', covar=covar_cols, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

    
    return mediation_analysis