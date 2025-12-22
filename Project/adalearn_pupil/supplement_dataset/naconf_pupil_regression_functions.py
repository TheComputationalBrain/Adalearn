import pandas as pd
import numpy as np
import os
import sys
import pingouin
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


def pupil_peristimulus_regression_analysis(participant_str, directory_path, task_session_df, outcome_epoched_pupil_df, outcome_epochs_to_remove=None, variables=None, covariates=None, position_df = None, baseline_corrected = False, plot_regression_analysis=True):
    """
    Function to analyse the relationship between pupil response and a set of computational predictor variables using multiple linear regression at each peri-stimulus timepoint across defined epochs (defined within the pupil analysis). Can include covariates. Returns the output of the regression analysis.

    Parameters:
    - participant_str: string, identifier of participant number (e.g. one digit for 1-9, two digits for 10-99)
    - directory_path: Path, directory to participant level data 
    - task_session_df: pd.DataFrame, contains the computational variables (e.g. surprise, prior_conf, cpp) to be used as predictors.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs.
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within.
    - variables: list, names of the variables to be used as predictors in the regression analysis.
    - covariates: list, names of the variables to be used as covariates in the regression analysis.
    - position_df: pd.DataFrame, contains the pupil position data to be included as covariates (if any).
    - baseline_corr: boolean, if true, specifies that pupil data used has been baseline corrected (default = true). 
    - plot_regression_analysis: boolean, if True, plots the regression coefficients at each peri-stimulus timepoint.

    Returns:
    - coefficients_df: pd.DataFrame, contains the coefficients of the regression analysis for each predictor variable at each peri-stimulus timepoint.
    - pupil_regress_df: pd.DataFrame, contains model evaluation outputs (e.g. intercept, R^2, mse, rmse) for each peristimulus time point.
    - residuals_df: pd.DataFrame, contains model residuals for each peristimulus timepoint. 
    """
    task_session_df = task_session_df.copy()

    # Ensuring that the task session data is in the correct format
    task_session_df['session'] = task_session_df['session'].apply(lambda x: str(x) if not isinstance(x, str) else x).str.zfill(2)

    # Check that at least one variable has been added as a predictor
    if variables is None:
        raise ValueError("You must provide at least one variable as a predictor.")

    # Ensure variables is always a list of strings
    if isinstance(variables, str):
        variables = [variables]

    if covariates is None:
        covariates = []

    # Z-score predictors
    zscored_variable_names = []

    for var in variables:
        if var not in task_session_df.columns:
            raise ValueError(f"Variable '{var}' not found in task_session_df.")
        
        if task_session_df[var].dtype in ['int64', 'float64']:
            print('processing predictors...')
            print(f'z-scoring {var}')  
            zscored_name = f'{var}_zscore'
            task_session_df[zscored_name] = zscore(task_session_df[var], nan_policy='omit')
            zscored_variable_names.append(zscored_name)
        else:
            print(f'{var} not numeric and not z-scored')
            task_session_df[f'{var}_zscore'] = task_session_df[var]  
            zscored_variable_names.append(f'{var}_zscore')

    # Use z-scored predictors
    predictors = task_session_df[zscored_variable_names].copy()

    # Handling of other covariates to be included (z-scoring etc) 
    if len(covariates) > 0:  
        covariates_to_zscore = list(covariates)
        # Add dummy coded stim variable if 'stim' in covariates (A = 0, G = 1)
        if 'stim' in covariates:
            task_session_df['outcome'] = task_session_df['stim'].map({'A': 0, 'G': 1})
            covariates_to_zscore.extend(['outcome'])
            covariates_to_zscore.remove('stim')

        # Ensure session is integer type
        if 'session' in covariates:
            task_session_df['session'] = task_session_df['session'].astype(float)

        # Add square and sqrt of outcome_num
        if 'trial_ID' in covariates:
            task_session_df['outcome_num_sq'] = pd.to_numeric(task_session_df['trial_ID'] ** 2)
            task_session_df['outcome_num_sqrt'] = pd.to_numeric(task_session_df['trial_ID'] ** 0.5)
            covariates_to_zscore.extend(['outcome_num_sq', 'outcome_num_sqrt'])

        # Compute entropy if 'prior_dist' is in covariates
        if 'prior' in covariates:
            p = np.clip(task_session_df['prior'], 1e-10, 1 - 1e-10)
            task_session_df['entropy'] = pd.to_numeric(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
            covariates_to_zscore.extend(['entropy'])

        # Add the previous confidence value in as a covariate
        if 'previous_conf' in covariates:
            task_session_df['previous_conf'] = pd.to_numeric(task_session_df['prior_conf'].shift(1).fillna(0))
            covariates_to_zscore.extend(['previous_conf'])

        # Add the previous change point probability value in as a covariate
        if 'previous_cpp' in covariates:
            task_session_df['previous_cpp'] = pd.to_numeric(task_session_df['cpp'].shift(1).fillna(0))
            covariates_to_zscore.extend(['previous_cpp'])

        print(f"variables to zscore: {covariates_to_zscore}")    

        for cov in covariates_to_zscore:
            if cov not in task_session_df.columns:
                raise ValueError(f"Covariate '{cov}' not found in task_session_df.")

            if task_session_df[cov].dtype in ['int64', 'float64']:
                print('processing covariates...')
                print(f'z-scoring {cov}')  
                predictors[f'{cov}_zscore'] = zscore(task_session_df[cov], nan_policy='omit')
            else:
                print(f'{cov} not numeric and not z-scored')
                predictors[cov] = task_session_df[cov]


    # Include position data if specified
    if position_df is not None:
        print('Including pupil position data as a covariate in the regression analysis')
        # Z-score the position data
        time_pos_col = position_df['time'] if 'time' in position_df.columns else None

        # Z-score all other columns (trials/epochs) column-wise
        pupil_euclid_z = position_df.drop(columns=['time'], errors='ignore')
        pupil_euclid_z = pupil_euclid_z.apply(lambda col: (col - col.mean()) / col.std(), axis=1)

        # Add time back
        if time_pos_col is not None:
            pupil_euclid_z['time'] = time_pos_col

    # Based on the epochs containing too much interpolated data, remove respective rows                
    if outcome_epochs_to_remove is not None:
        outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
        predictors.drop(index=outcomes_to_remove, inplace=True)
    
    # Ensure there are no epochs containing nan values
    nan_epochs = predictors[predictors.isna().any(axis=1)].index
    predictors = predictors.dropna()

    # Prepare the pupil data
    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time'].astype(float)
    time_points = epoched_pupil_data.index
    epoched_pupil_data.drop(columns=['time'], inplace=True)

    # Prepare the pupil position data if included
    if position_df is not None:
        epoched_pupil_position_data = pupil_euclid_z.copy()
        time = epoched_pupil_position_data['time'].astype(float)
        time_points = epoched_pupil_position_data.index
        epoched_pupil_position_data.drop(columns=['time'], inplace=True)

    # Build validity mask from the ORIGINAL aligned matrices
    valid_time_mask = ~epoched_pupil_data.isna().any(axis=1)

    if position_df is not None:
        valid_time_mask &= ~epoched_pupil_position_data.isna().any(axis=1)

    # Apply mask symmetrically
    epoched_pupil_data = epoched_pupil_data.loc[valid_time_mask]

    if position_df is not None:
        epoched_pupil_position_data = epoched_pupil_position_data.loc[valid_time_mask]

    # Apply to time and time_points
    time = time.loc[valid_time_mask]
    time_points = epoched_pupil_data.index

    assert not epoched_pupil_data.isna().any().any(), "NaNs remain in pupil diameter data"

    if position_df is not None:
        assert not epoched_pupil_position_data.isna().any().any(), "NaNs remain in pupil position data"
        assert epoched_pupil_data.index.equals(epoched_pupil_position_data.index), \
            "Diameter and position indices are misaligned"

    # Transpose the pupil dataframe
    epoched_pupil_diameter_long = epoched_pupil_data.melt(
        var_name='epoch',
        value_name='pupil',
        ignore_index=False  # keeps the row index as the time point
    ).rename_axis('time_points').reset_index()

    epoched_pupil_diameter_long = epoched_pupil_diameter_long[~epoched_pupil_diameter_long['epoch'].isin(nan_epochs)]

    # Transpose the pupil position dataframe if included
    if position_df is not None:
        epoched_pupil_position_long = epoched_pupil_position_data.melt(
            var_name='epoch',
            value_name='pupil_position',
            ignore_index=False
        ).rename_axis('time_points').reset_index()

        epoched_pupil_position_long = epoched_pupil_position_long[~epoched_pupil_position_long['epoch'].isin(nan_epochs)]
        
    # Ensure that the index is of the right data type
    epoched_pupil_diameter_long['epoch'] = epoched_pupil_diameter_long['epoch'].astype(int)

    if position_df is not None:
        epoched_pupil_position_long['epoch'] = epoched_pupil_position_long['epoch'].astype(int)


    # Merge diameter and position data if position is included
    if position_df is not None:
        regression_design_df = epoched_pupil_diameter_long.merge(
            epoched_pupil_position_long,
            how='left',
            on=['time_points','epoch']
        )
    else:
        regression_design_df = epoched_pupil_diameter_long.copy()

    # Map variables to pupil data 
    for var in predictors.columns:
        regression_design_df[var] = regression_design_df['epoch'].map(predictors[var])

    # Set up capturing of outputs from regrprint(epoch_counts[epoch_counts < 5])ession analysis
    coefficients = {var: [] for var in predictors.columns}
    if position_df is not None:
        coefficients['pupil_position'] = []
    residuals_by_time = []
    model_outputs = {'time': [], 'intercept': [], 'r_squared': [], 'mean_squared_error': [], 'root_mean_squared_error': []}

    # Run regression analyses across duration of the epoch
    for t in regression_design_df['time_points'].unique():
        data_at_time = regression_design_df[regression_design_df['time_points'] == t]

        X = data_at_time[predictors.columns].apply(pd.to_numeric, errors='coerce')

        if position_df is not None:
            X['pupil_position'] = data_at_time['pupil_position'].values

        y = pd.to_numeric(data_at_time['pupil'], errors='coerce')

        X = sm.add_constant(X)  # Adds intercept term

        # Drop rows with any NaN or Inf in X or y
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

        X = X.loc[valid]
        y = y.loc[valid]

        # Skip time point if nothing left to fit
        if len(y) < X.shape[1]:
            continue

        model = sm.OLS(y, X).fit()

        for var in predictors.columns:
            coefficients[var].append(model.params[var])

        if position_df is not None:
            coefficients['pupil_position'].append(model.params['pupil_position'])   
        
        residuals = pd.Series(model.resid, index=data_at_time.index)
        residuals = residuals.reset_index(drop=True)
        residuals_by_time.append(residuals)
        
        # Store the outputs for each epoch analysis
        model_outputs['time'].append(t)
        model_outputs['intercept'].append(model.params['const'])
        model_outputs['r_squared'].append(model.rsquared)
        model_outputs['mean_squared_error'].append(mean_squared_error(y, model.fittedvalues))
        model_outputs['root_mean_squared_error'].append(np.sqrt(mean_squared_error(y, model.fittedvalues)))

    # Colate epoch regression outputs 
    residuals_df = pd.DataFrame(residuals_by_time, index=time_points)
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df['time'] = time
    coefficients_df['participant'] = participant_str
    coefficients_df['count'] = range(len(coefficients_df))
    pupil_regress_df = pd.DataFrame(model_outputs)
    
    avg_r_squared = pupil_regress_df['r_squared'].mean()
    avg_rmse = pupil_regress_df['root_mean_squared_error'].mean()

    print(f"Average R^2 across time points within the model: {avg_r_squared:.4f}")
    print(f"Average RMSE across time points within the model: {avg_rmse:.4f}")

    # Additional title information 
    if baseline_corrected:
        title_addon = "baseline-corrected"
    else:
        title_addon = "non-baseline-corrected"

    # Option to plot regression coefficients at each peri-stimulus timepoint
    if plot_regression_analysis:
        fig = go.Figure()
        for var in predictors.columns:
            fig.add_trace(go.Scatter(
                x=coefficients_df['time'], 
                y=coefficients_df[var],
                mode='lines',
                name=f'Coefficient of {var}'
            ))

        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            title=f"Regression coefficients describing pupil response relationships with computational variables over time for participant {participant_str} ",
            xaxis_title="Time (s)",
            yaxis_title="Coefficient value",
            legend_title="Predictors",
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

       # Save the figure to the participant folder
        plot_dir = os.path.join(directory_path, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_{title_addon}-pupil-by-comp-var-peristimulus-regression.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
       

        fig.show()

    print("Pupil regression analysis complete.")
    print(coefficients_df)

    return coefficients_df, pupil_regress_df, residuals_df