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


def pupil_regression_analysis(participant_str, condition_str, computational_var_df, outcome_epoched_pupil_df, outcome_epochs_to_remove=None, variables=None, covariates=None, include_outcome=False, plot_regression_analysis=True):
    """
    Function to analyse the relationship between pupil response and a set of computational predictor variables using multiple linear regression at each peri-stimulus timepoint across defined epochs (defined within the pupil analysis).

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - computational_var_df: pd.DataFrame, contains the computational variables (e.g. surprise, prior_conf, cpp) to be used as predictors.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs.
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within.
    - variables: list, names of the variables to be used as predictors in the regression analysis.
    - covariates: list, names of the variables to be used as covariates in the regression analysis. 
    - include_outcome: boolean, if True, includes the 'outcome' variable as an additional predictor in the regression analysis.
    - plot_regression_analysis: boolean, if True, plots the regression coefficients at each peri-stimulus timepoint.

    Returns:
    - coefficients_df: pd.DataFrame, contains the coefficients of the regression analysis for each predictor variable at each peri-stimulus timepoint.
    - pupil_regress_df: pd.DataFrame, contains model evaluation outputs (e.g. intercept, R^2, mse, rmse) for each peristimulus time point.
    - residuals_df: pd.DataFrame, contains model residuals for each peristimulus timepoint. 
    """
    if variables is None:
        raise ValueError("You must provide at least one variable as a predictor.")
    
    if isinstance(variables, str):
        variables = [variables]
    
    predictors = computational_var_df[variables].copy()
    
    if include_outcome:
        if 'outcome' not in computational_var_df.columns:
            raise ValueError("The 'outcome' variable is not present in computational_var_df.")
        
        scaler = StandardScaler()
        computational_var_df['outcome_zscore'] = scaler.fit_transform(computational_var_df[['outcome']]).flatten()
        predictors['outcome'] = computational_var_df['outcome_zscore']
    
    if covariates is not None:
        scaler = StandardScaler()
    
        for cov in covariates:
            if cov not in computational_var_df.columns:
                raise ValueError(f"Covariate '{cov}' not found in computational_var_df.")

            # Check if the covariate is numeric before z-scoring
            #if computational_var_df[cov].dtype in ['int64', 'float64']:  
                predictors[cov] = scaler.fit_transform(computational_var_df[[cov]]).flatten()
            else:
                predictors[cov] = computational_var_df[cov] 
                   
    if outcome_epochs_to_remove is not None:
        predictors.drop(index=outcome_epochs_to_remove, inplace=True)
    
    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time']
    time_points = epoched_pupil_data.index
    epoched_pupil_data.drop(columns=['mean', 'std', 'sem', 'time'], inplace=True)

    nan_epochs = predictors[predictors.isna().any(axis=1)].index
    predictors = predictors.dropna()
    
    epoched_pupil_data_long = epoched_pupil_data.melt(var_name='epoch', value_name='pupil', ignore_index=False)
    epoched_pupil_data_long = epoched_pupil_data_long[~epoched_pupil_data_long['epoch'].isin(nan_epochs)]
    
    for var in predictors.columns:
        epoched_pupil_data_long[var] = epoched_pupil_data_long['epoch'].map(predictors[var])
    
    coefficients = {var: [] for var in predictors.columns}
    residuals_by_time = []
    model_outputs = {'time': [], 'intercept': [], 'r_squared': [], 'mean_squared_error': [], 'root_mean_squared_error': []}
    
    for t in time_points:
        data_at_time = epoched_pupil_data_long.loc[t]
        X = data_at_time[predictors.columns]
        y = data_at_time['pupil']
        X = sm.add_constant(X)  # Adds intercept term
        
        model = sm.OLS(y, X).fit()
        
        for var in predictors.columns:
            coefficients[var].append(model.params[var])
        
        residuals = pd.Series(model.resid, index=data_at_time.index)
        residuals = residuals.reset_index(drop=True)
        residuals_by_time.append(residuals)
        
        model_outputs['time'].append(t)
        model_outputs['intercept'].append(model.params['const'])
        model_outputs['r_squared'].append(model.rsquared)
        model_outputs['mean_squared_error'].append(mean_squared_error(y, model.fittedvalues))
        model_outputs['root_mean_squared_error'].append(np.sqrt(mean_squared_error(y, model.fittedvalues)))
    
    residuals_df = pd.DataFrame(residuals_by_time, index=time_points)
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df['time'] = time
    pupil_regress_df = pd.DataFrame(model_outputs)
    
    avg_r_squared = pupil_regress_df['r_squared'].mean()
    avg_rmse = pupil_regress_df['root_mean_squared_error'].mean()

    print(f"Average R^2 across time points within the model: {avg_r_squared:.4f}")
    print(f"Average RMSE across time points within the model: {avg_rmse:.4f}")

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
            title=f"Regression coefficients describing pupil response relationships with {variables} over time for participant {participant_str} in condition {condition_str}",
            xaxis_title="Time (s)",
            yaxis_title="Coefficient value",
            legend_title="Predictors",
            height=950,
            width=1800,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-by-{variables}-regression.png"
        fig.write_image(os.path.join(desktop_path, file_name))  

        fig.show()

    return coefficients_df, pupil_regress_df, residuals_df



def pupil_uncertainty_lr_mediation_analysis(participant_str, condition_str, computational_var_df, outcome_epoched_pupil_df, outcome_epochs_to_remove, variables, baseline_corrected = True, plot_mediation_analysis = True):
    """
    Prepares a DataFrame for mediation analysis by selecting relevant computational variables and combining with pupil response, conditioned on whether baseline correction is applied. If baseline corrected data is offered as an input, the pupil response will be the pupil amplitude following stimuli presentation (i.e. max - min within the window 0 - 1.4 s). If not, the average pupil response on non-baseline corrected data will be used (the average is taken across the whole epoch). 

    Parameters:

    - participant_str: string, participant identifier.
    - condition_str: string, identifier of the condition.
    - computational_var_df: pd.DataFrame, the dataset containing the variables to be analysed.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics, and timestamp across epochs.
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within.
    - variables: list, names of the variables to be used as predictors in the regression analysis.
    - baseline_corrected: boolean, if true, specifies that the pupil data provided is baseline-corrected and the amplitude of the response will be calculated. If false, non-baseline corrected pupil data provided and the avaerage response will be evaluated (default = True).
    - plot_mediation_analysis: boolean, if true will plot path coefficents of the mediation analysis (default = True)

    Returns:
    - pd.DataFrame: pd.DataFrame, contains the output statistics of the mediation analysis.
    """
    scaler = StandardScaler()
    
    mediators = computational_var_df.copy()
    mediators.drop(index=outcome_epochs_to_remove, inplace=True)

    pupil_data = outcome_epoched_pupil_df.copy()
    pupil_data = pupil_data.set_index('time')
    pupil_data = pupil_data.drop(columns=['mean', 'std', 'sem'])

    if baseline_corrected:
        filtered_pupil_data_bc = pupil_data.loc[(pupil_data.index >= 0) & (pupil_data.index <= 1.4)]
        pupil_variable = filtered_pupil_data_bc.max() - filtered_pupil_data_bc.min()
        title = f"amplitude"
    else:
        pupil_variable = pupil_data.mean()
        title = f"mean"

    mediation_df = mediators[variables].copy()
    mediation_df['learning_rate'] = mediators['lr_zscore_estimate']
    mediation_df['pupil_response'] = pupil_variable.values 
    mediation_df['pupil_response_zscore'] = scaler.fit_transform(mediation_df[['pupil_response']]).flatten()


    mediation_analysis = pingouin.mediation_analysis(data=mediation_df, x='pupil_response_zscore', m=['conf_zscore', 'cpp_zscore'], y='learning_rate', covar=None, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

    if plot_mediation_analysis:
        fig = go.Figure()
        fig.update_xaxes(range=[0, 10], zeroline=False)
        fig.update_yaxes(range=[0, 10])

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=1, y0=5, x1=5,
            y1=8,
            line=dict(
                color="Black",
                width=3,
            ),
        )

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=5, y0=8, x1=9,
            y1=5,
            line=dict(
                color="Black",
                width=3,
            ),
        )

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=1, y0=5, x1=5,
            y1=2,
            line=dict(
                color="Black",
                width=3,
            ),
        )

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=5, y0=2, x1=9,
            y1=5,
            line=dict(
                color="Black",
                width=3,
            ),
        )

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=1, y0=5.2, x1=9,
            y1=5.2,
            line=dict(
                color="Black",
                width=3,
            ),
        )

        fig.add_shape(type="line",
            xref="x", yref="y",
            x0=1, y0=4.8, x1=9,
            y1=4.8,
            line=dict(
                color="Black",
                width=3,
                dash='dash'
            ),
        )

        fig.add_shape(type="circle",
            xref="x", yref="y",
            fillcolor="PaleTurquoise",
            x0=0.5, y0=4.5, x1=1.5, y1=5.5,
            line_color="LightSeaGreen",
            label=dict(text=f"Pupil: <br>{title}", textangle=0)
        )

        fig.add_shape(type="circle",
            xref="x", yref="y",
            fillcolor="PaleTurquoise",
            x0=8.5, y0=4.5, x1=9.5, y1=5.5,
            line_color="LightSeaGreen",
            label=dict(text="Learning Rate", textangle=0)
        )

        fig.add_shape(type="circle",
            xref="x", yref="y",
            fillcolor="PaleTurquoise",
            x0=4.5, y0=7.5, x1=5.5, y1=8.5,
            line_color="LightSeaGreen",
            label=dict(text="Prior Conf", textangle=0)
        )

        fig.add_shape(type="circle",
            xref="x", yref="y",
            fillcolor="PaleTurquoise",
            x0=4.5, y0=1.5, x1=5.5, y1=2.5,
            line_color="LightSeaGreen",
            label=dict(text="CPP", textangle=0)
        )

        total_path_text = (
                        f"Total Effect (C)"
                        f"<br>Coef = {mediation_analysis['coef'][4]:.2f}"
                        f"<br>95% CI = {mediation_analysis['CI[2.5%]'][4]:.2f} - {mediation_analysis['CI[97.5%]'][4]:.2f}"
                        f"<br>p-value = {mediation_analysis['pval'][4]:.3f}"
                    )


        fig.add_annotation(
            x=5,  
            y=4,
            xref="x",
            yref="y",
            text=total_path_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

        direct_path_text = (
                        f"Direct Effect (C')"
                        f"<br>Coef = {mediation_analysis['coef'][5]:.2f}"
                        f"<br>95% CI = {mediation_analysis['CI[2.5%]'][5]:.2f} - {mediation_analysis['CI[97.5%]'][5]:.2f}"
                        f"<br>p-value = {mediation_analysis['pval'][5]:.3f}"
                    )


        fig.add_annotation(
            x=5,  
            y=6,
            xref="x",
            yref="y",
            text=direct_path_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )


        conf_path_text = (
                        f"Indirect Effect (Conf)"
                        f"<br>Coef = {mediation_analysis['coef'][6]:.2f}"
                        f"<br>95% CI = {mediation_analysis['CI[2.5%]'][6]:.2f} - {mediation_analysis['CI[97.5%]'][6]:.2f}"
                        f"<br>p-value = {mediation_analysis['pval'][6]:.3f}"
                    )


        fig.add_annotation(
            x=5,  
            y=9.2,
            xref="x",
            yref="y",
            text=conf_path_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

        cpp_path_text = (
                        f"Indirect Effect (CPP)"
                        f"<br>Coef = {mediation_analysis['coef'][7]:.2f}"
                        f"<br>95% CI = {mediation_analysis['CI[2.5%]'][7]:.2f} - {mediation_analysis['CI[97.5%]'][7]:.2f}"
                        f"<br>p-value = {mediation_analysis['pval'][7]:.3f}"
                    )


        fig.add_annotation(
            x=5,  
            y=0.8,
            xref="x",
            yref="y",
            text=cpp_path_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

        fig.update_layout(
            title=f"Mediation analysis for particpant {participant_str} in condition {condition_str} examining pupil {title} response on behavioural learning rate <br> mediated by uncertainty parameters",
            width=1000, 
            height=1000,
            xaxis=dict(showticklabels=False), 
            yaxis=dict(showticklabels=False)
        )  

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-{title}-uncertainty-lr-mediation.png"
        fig.write_image(os.path.join(desktop_path, file_name))   

        fig.show()
    
    return mediation_analysis