import pandas as pd
import numpy as np
import os
import sys
import pingouin
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


def pupil_peristimulus_regression_analysis(participant_str, condition_str, directory_path, computational_var_df, outcome_epoched_pupil_df, outcome_epochs_to_remove=None, variables=None, covariates=None, baseline_corrected = False, plot_regression_analysis=True):
    """
    Function to analyse the relationship between pupil response and a set of computational predictor variables using multiple linear regression at each peri-stimulus timepoint across defined epochs (defined within the pupil analysis). Can include covariates. Returns the output of the regression analysis.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - directory_path: Path, directory to participant level data 
    - computational_var_df: pd.DataFrame, contains the computational variables (e.g. surprise, prior_conf, cpp) to be used as predictors.
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
    #computational_var_df['task_session'] = computational_var_df['task_session'].apply(lambda x: str(x) if not isinstance(x, str) else x).str.zfill(2)

    # Check that at least one variable has been added as a predictor
    if variables is None:
        raise ValueError("You must provide at least one variable as a predictor.")
        
    
    # This ensures that variable titles are always entered as a list of strings
    if isinstance(variables, str):
        variables = [variables]
      
    # Obtain the columns related to the predictors defined
    predictors = computational_var_df[variables].copy()

    variables_to_zscore = list(variables)
    variables_to_zscore.extend(list(covariates))
        
    # Handling of other covariates to be included (z-scoring etc) 
    if covariates is not None:
        scaler = StandardScaler()  
        # Add square and sqrt of outcome_num
        if 'outcome_num' in covariates:
            computational_var_df['outcome_num_sq'] = computational_var_df['outcome_num'] ** 2
            computational_var_df['outcome_num_sqrt'] = computational_var_df['outcome_num'] ** 0.5
            variables_to_zscore.extend(['outcome_num_sq', 'outcome_num_sqrt'])

        # Compute entropy if 'prior_dist' is in covariates
        if 'prior_dist' in covariates:
            p = np.clip(computational_var_df['prior_dist'], 1e-10, 1 - 1e-10)
            computational_var_df['entropy'] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            variables_to_zscore.extend(['entropy'])

        # Add the previous confidence value in as a covariate
        if 'previous_conf' in covariates:
            variables_to_zscore.remove('previous_conf')
            computational_var_df['previous_conf'] = computational_var_df['prior_conf'].shift(1).fillna(0)
            variables_to_zscore.append('previous_conf')

        # Add the previous change point probability value in as a covariate
        if 'previous_cpp' in covariates:
            variables_to_zscore.remove('previous_cpp')
            computational_var_df['previous_cpp'] = computational_var_df['cpp'].shift(1).fillna(0)
            variables_to_zscore.append('previous_cpp')

        for var in variables_to_zscore:
            if var not in computational_var_df.columns:
                raise ValueError(f"Covariate '{var}' not found in computational_var_df.")

            if computational_var_df[var].dtype in ['int64', 'float64']:
                print('processing variables...')
                print(f'z-scoring {var}')  
                predictors[var] = scaler.fit_transform(computational_var_df[[var]]).flatten()
            else:
                print(f'{var} not numeric and not z-scored')
                predictors[var] = computational_var_df[var]

    # Based on the epochs containing too much interpolated data, remove respective rows                
    if outcome_epochs_to_remove is not None:
        outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
        predictors.drop(index=outcomes_to_remove, inplace=True)
    
    # Prepare the pupil data
    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time'].values.astype(float)
    time_points = epoched_pupil_data.index
    epoched_pupil_data.drop(columns=['mean', 'std', 'sem', 'time'], inplace=True)

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
        model_outputs['time'].append(time)
        model_outputs['intercept'].append(model.params['const'])
        model_outputs['r_squared'].append(model.rsquared)
        model_outputs['mean_squared_error'].append(mean_squared_error(y, model.fittedvalues))
        model_outputs['root_mean_squared_error'].append(np.sqrt(mean_squared_error(y, model.fittedvalues)))
    
    # Colate epoch rgression outputs 
    residuals_df = pd.DataFrame(residuals_by_time, index=time_points)
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df['time'] = time
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
            title=f"Regression coefficients describing pupil response relationships with computational variables over time for participant {participant_str} in condition {condition_str}",
            xaxis_title="Time (s)",
            yaxis_title="Coefficient value",
            legend_title="Predictors",
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

       # Save the figure to the participant folder
        plot_dir = os.path.join(directory_path, f"ses-{condition_str}", "plots")

        os.makedirs(plot_dir, exist_ok=True)

        file_name = f"sub-{participant_str}_ses-{condition_str}_{title_addon}-pupil-by-comp-var-peristimulus-regression.png"

        file_path = os.path.join(plot_dir, file_name)

        fig.write_image(file_path)
       

        fig.show()

    return coefficients_df, pupil_regress_df, residuals_df




def pupil_on_learning_rate_regression_analysis(participant_str, condition_str, directory_path, computational_var_df, outcome_epoched_pupil_df, outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove, parameter_estimate = 'estimate', exclude_zero = False, plot_pupil_lr_regress = True):
    """
    Function to conduct regression analysis for basleine corrected and non-corrected pupil metrics on learning rate. Returns analysis outputs and model parameters. 

    Parameters:
    - participant_str: string, participant identifier (number; set to 2 digits)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, directory path to the participant level of the data storage.
    - computational_var_df: pd.DataFrame, contains parsed data from json adalearn data file for select variables to be used in/or provide context to model analysis.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs for the baseline corrected pupil signal.
    - outcome_epoched_pupil_df_nobc: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs for the non-baseline corrected pupil signal.
    - exclude_zero: boolean, if true, will remove row in which no updates made (learning rate = 0) (default = False)
    - plot_pupil_lr_regress: boolean, if true, will plot the output of the correlations and linear regression weights assessing the association between parameters of uncertainty and learning rate (default = True). 

    Returns:
    - pupil_v_lr_df: pd.DataFRame, contains the regression coefficients 
    """
    # Calculating zscore for computational parameters used in analysis
    scaler = StandardScaler()

    computational_var_df[f'lr_zscore_{parameter_estimate}'] = scaler.fit_transform(computational_var_df[[f'cleaned_learning_rate_{parameter_estimate}']]).flatten()
    
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
    filtered_pupil_data_bc = pupil_data_bc.loc[(pupil_data_bc.index >= 0.6) & (pupil_data_bc.index <= 1.4)]
    pupil_variable_bc = filtered_pupil_data_bc.mean()

    # The average of the non-baseline corrected pupil response before the stimulus (used as a marker of tonic activity)
    pupil_data_nobc = outcome_epoched_pupil_df_nobc.copy()
    pupil_data_nobc = pupil_data_nobc.set_index('time')
    pupil_data_nobc = pupil_data_nobc.drop(columns=['mean', 'std', 'sem'])

    filtered_pupil_data_nobc = pupil_data_nobc.loc[(pupil_data_nobc.index >= -0.25) & (pupil_data_nobc.index <= 0)]
    pupil_variable_nobc = filtered_pupil_data_nobc.mean()

    # Creating dataframe for regression 
    lr_regress_df = pd.DataFrame()
    lr_regress_df['learning_rate_zscore'] = df_clean[f'lr_zscore_{parameter_estimate}']

    lr_regress_df['pupil_response_bc'] = pupil_variable_bc.values 
    lr_regress_df['pupil_response_bc_zscore'] = scaler.fit_transform(lr_regress_df[['pupil_response_bc']]).flatten()

    lr_regress_df['pupil_response_nobc'] = pupil_variable_nobc.values 
    lr_regress_df['pupil_response_nobc_zscore'] = scaler.fit_transform(lr_regress_df[['pupil_response_nobc']]).flatten()

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
        lr_regress_df['learning_rate_zscore'] = scaler.fit_transform(lr_regress_df[['cleaned_learning_rate_estimate']]).flatten()

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



def pupil_uncertainty_lr_mediation_analysis(participant_str, condition_str, directory_path, computational_var_df, outcome_epoched_pupil_df, outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove, uncertainty_param, variables, exclude_zero=False):
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
    - variables: list, names of the variables to be used as predictors in the regression analysis.
    - baseline_corrected: boolean, if true, specifies that the pupil data provided is baseline-corrected and the amplitude of the response will be calculated. If false, non-baseline corrected pupil data provided and the avaerage response will be evaluated (default = True).
    - plot_mediation_analysis: boolean, if true will plot path coefficents of the mediation analysis (default = True)

    Returns:
    - pd.DataFrame: pd.DataFrame, contains the output statistics of the mediation analysis.
    """
    # Preparing the data. Ensuring that the pupil data and "mediators" are matched in terms of the pupil epochs that can be used (do not excede the threshold of too much interpolated data)  
    predictors = computational_var_df.copy()
    outcomes_to_remove = outcome_epochs_to_remove['epochs'].values
    predictors.drop(index=outcomes_to_remove, inplace=True)

    pupil_data_bc = outcome_epoched_pupil_df.copy()
    pupil_data_bc = pupil_data_bc.set_index('time')
    pupil_data_bc = pupil_data_bc.drop(columns=['mean', 'std', 'sem'])

    # Baseline corrected specify pupil data to be examined (the average after an outcome within the window of the peristimulus effect is used as a marker of phasic pupil activity.) 
    filtered_pupil_data_bc = pupil_data_bc.loc[(pupil_data_bc.index >= 0.6) & (pupil_data_bc.index <= 1.4)]
    pupil_variable_bc = filtered_pupil_data_bc.mean()

    # The average of the non-baseline corrected pupil response before the stimulus (used as a marker of tonic activity)
    pupil_data_nobc = outcome_epoched_pupil_df_nobc.copy()
    pupil_data_nobc = pupil_data_nobc.set_index('time')
    pupil_data_nobc = pupil_data_nobc.drop(columns=['mean', 'std', 'sem'])

    filtered_pupil_data_nobc = pupil_data_nobc.loc[(pupil_data_nobc.index >= -0.25) & (pupil_data_nobc.index <= 0)]
    pupil_variable_nobc = filtered_pupil_data_nobc.mean()

    # Prepare variables for mediation analysis
    scaler = StandardScaler() 

    mediation_df = predictors[variables].copy()
    mediation_df['learning_rate_zscore'] = predictors['lr_zscore_estimate']
    mediation_df['pupil_response_bc'] = pupil_variable_bc.values 
    mediation_df['pupil_response_bc_zscore'] = scaler.fit_transform(mediation_df[['pupil_response_bc']]).flatten()

    mediation_df['pupil_response_nobc'] = pupil_variable_nobc.values 
    mediation_df['pupil_response_nobc_zscore'] = scaler.fit_transform(mediation_df[['pupil_response_nobc']]).flatten()

    # Option to exclude no update trials
    if exclude_zero:
        before_zero_filter = len(mediation_df)
        mediation_df['cleaned_learning_rate_estimate'] = predictors['cleaned_learning_rate_estimate']
        mediation_df = mediation_df[mediation_df['cleaned_learning_rate_estimate'] != 0]
        removed_zero = before_zero_filter - len(mediation_df)
        print(f"Removed {removed_zero} rows where learning rate was 0.")
        # Re z-scoring the data after zero updates removed
        mediation_df['learning_rate_zscore'] = scaler.fit_transform(mediation_df[['cleaned_learning_rate_estimate']]).flatten()

    print(f'length of mediators before removing NaN values: {len(mediation_df)}')
    mediation_df = mediation_df.dropna(subset=['learning_rate_zscore'])
    print(f'length of mediators after removing Nan: {len(mediation_df)}')

    # Run mediation analysis (currently no option to exclude no update trials)
    mediation_analysis = pingouin.mediation_analysis(data=mediation_df, x=uncertainty_param, m=['pupil_response_bc', 'pupil_response_nobc'], y='learning_rate_zscore', covar=None, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

    
    return mediation_analysis