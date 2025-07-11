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


def pupil_peristimulus_regression_analysis(participant_str, directory_path, task_session_df, outcome_epoched_pupil_df, outcome_epochs_to_remove=None, variables=None, covariates=None, baseline_corrected = False, plot_regression_analysis=True):
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
    - baseline_corr: boolean, if true, specifies that pupil data used has been baseline corrected (default = true). 
    - plot_regression_analysis: boolean, if True, plots the regression coefficients at each peri-stimulus timepoint.

    Returns:
    - coefficients_df: pd.DataFrame, contains the coefficients of the regression analysis for each predictor variable at each peri-stimulus timepoint.
    - pupil_regress_df: pd.DataFrame, contains model evaluation outputs (e.g. intercept, R^2, mse, rmse) for each peristimulus time point.
    - residuals_df: pd.DataFrame, contains model residuals for each peristimulus timepoint. 
    """
     # Ensuring that the task session data is in the correct format
    task_session_df['session'] = task_session_df['session'].apply(lambda x: str(x) if not isinstance(x, str) else x).str.zfill(2)

    # Check that at least one variable has been added as a predictor
    if variables is None:
        raise ValueError("You must provide at least one variable as a predictor.")

    # Ensure variables is always a list of strings
    if isinstance(variables, str):
        variables = [variables]

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
    if covariates is not None:
        covariates_to_zscore = list(covariates)  
        # Add square and sqrt of outcome_num
        if 'trial_ID' in covariates:
            task_session_df['tiral_ID_sq'] = task_session_df['trial_ID'] ** 2
            task_session_df['trial_ID_sqrt'] = task_session_df['trial_ID'] ** 0.5
            covariates_to_zscore.extend(['trial_ID_sq', 'trial_ID_sqrt'])

        # Compute entropy if 'P1' is in covariates
        if 'prior_dist' in covariates:
            p = np.clip(task_session_df['prior_dist'], 1e-10, 1 - 1e-10)
            task_session_df['entropy'] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            covariates_to_zscore.extend(['entropy'])

        # Add the previous confidence value in as a covariate
        if 'previous_conf' in covariates:
            covariates_to_zscore.remove('previous_conf')
            task_session_df['previous_conf'] = task_session_df['prior_conf'].shift(1).fillna(0)
            covariates_to_zscore.append('previous_conf')

        # Add the previous change point probability value in as a covariate
        if 'previous_cpp' in covariates:
            covariates_to_zscore.remove('previous_cpp')
            task_session_df['previous_cpp'] = task_session_df['cpp'].shift(1).fillna(0)
            covariates_to_zscore.append('previous_cpp')

        for cov in covariates_to_zscore:
            if cov not in task_session_df.columns:
                raise ValueError(f"Covariate '{cov}' not found in task_session_df.")

            if task_session_df[cov].dtype in ['int64', 'float64']:
                print('processing covariates...')
                print(f'z-scoring {cov}')  
                predictors[cov] = zscore(task_session_df[cov], nan_policy='omit')
            else:
                print(f'{cov} not numeric and not z-scored')
                predictors[cov] = task_session_df[cov]

    # Based on the epochs containing too much interpolated data, remove respective rows                
    if outcome_epochs_to_remove is not None:
        outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
        predictors.drop(index=outcomes_to_remove, inplace=True)
    
    # Prepare the pupil data
    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time']
    time_points = epoched_pupil_data.index
    epoched_pupil_data.drop(columns=['time'], inplace=True)

    # Ensure there are no epochs containing nan values
    nan_epochs = predictors[predictors.isna().any(axis=1)].index
    predictors = predictors.dropna()
    
    # Transpose the pupil dataframe
    epoched_pupil_data_long = epoched_pupil_data.melt(var_name='epoch', value_name='pupil', ignore_index=False)
    epoched_pupil_data_long = epoched_pupil_data_long[~epoched_pupil_data_long['epoch'].isin(nan_epochs)]

    # Ensure that the index is of the right data type
    epoched_pupil_data_long['epoch'] = epoched_pupil_data_long['epoch'].astype(int)
    
    # Map variables to pupil data 
    for var in predictors.columns:
        epoched_pupil_data_long[var] = epoched_pupil_data_long['epoch'].map(predictors[var])
    
    # Set up capturing of outputs from regression analysis
    coefficients = {var: [] for var in predictors.columns}
    residuals_by_time = []
    model_outputs = {'time': [], 'intercept': [], 'r_squared': [], 'mean_squared_error': [], 'root_mean_squared_error': []}
    
    # Run regression analyses across duration of the epoch
    for t in time_points:
        data_at_time = epoched_pupil_data_long.loc[t]
        X = data_at_time[predictors.columns].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(data_at_time['pupil'], errors='coerce')
        X = sm.add_constant(X)  # Adds intercept term
        
        model = sm.OLS(y, X).fit()
        
        for var in predictors.columns:
            coefficients[var].append(model.params[var])
        
        residuals = pd.Series(model.resid, index=data_at_time.index)
        residuals = residuals.reset_index(drop=True)
        residuals_by_time.append(residuals)
        
        # Store the outputs for each epoch analysis
        model_outputs['time'].append(t)
        model_outputs['intercept'].append(model.params['const'])
        model_outputs['r_squared'].append(model.rsquared)
        model_outputs['mean_squared_error'].append(mean_squared_error(y, model.fittedvalues))
        model_outputs['root_mean_squared_error'].append(np.sqrt(mean_squared_error(y, model.fittedvalues)))
    
    # Colate epoch rgression outputs 
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

    return coefficients_df, pupil_regress_df, residuals_df