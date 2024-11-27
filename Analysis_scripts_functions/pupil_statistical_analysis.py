import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from pupil_analysis_test import c1_pupil_trig_df, c1_outcome_epoched_pupil_df, c1_outcome_blink_ratio_df, c1_outcome_epochs_to_remove, c2_pupil_trig_df, c2_outcome_epoched_pupil_df, c2_outcome_blink_ratio_df, c2_outcome_epochs_to_remove

from computational_analysis_test import c1_surprise, c1_cpp_df, c1_prior_conf_df, c2_surprise, c2_cpp_df, c2_prior_conf_df



def pupil_regression_analysis(comp_variable_df, outcome_epoched_pupil_df, pupil_trig_df, outcome_epochs_to_remove = None, outcome_parameter = False, variable = 'surprise'):
    """
    
    """
    comp_variable = comp_variable_df[variable]
    if outcome_epochs_to_remove is not None:
        comp_variable.drop(index=outcome_epochs_to_remove, inplace=True)


    outcome = pupil_trig_df['outcome']
    if outcome_epochs_to_remove is not None:
        outcome.drop(index=outcome_epochs_to_remove, inplace=True)


    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time']
    time_points = epoched_pupil_data.index 
    epoched_pupil_data.drop(columns=['mean', 'std', 'sem', 'time'], inplace=True)

    nan_epochs = comp_variable[comp_variable.isna()].index
    print(f"Removed epochs due to NaN values: {nan_epochs}")


    comp_variable = comp_variable.dropna()
    outcome = outcome.drop(index=nan_epochs)
    epoched_pupil_data_long = epoched_pupil_data.melt(var_name='epoch', value_name='pupil', ignore_index=False)
    epoched_pupil_data_long = epoched_pupil_data_long[~epoched_pupil_data_long['epoch'].isin(nan_epochs)]


    epoched_pupil_data_long[variable] = epoched_pupil_data_long['epoch'].map(comp_variable)
    epoched_pupil_data_long['outcome'] = epoched_pupil_data_long['epoch'].map(outcome)
    epoched_pupil_data_long['interaction'] = epoched_pupil_data_long[variable] * epoched_pupil_data_long['outcome']


    # Prepare lists to store regression results
    coefficients = []

    if outcome_parameter:
        # Run regression for each time point
        for t in time_points:
            # Filter the data for this time point
            data_at_time = epoched_pupil_data_long.loc[t]
    
            # Define predictors and target variable
            X = data_at_time[[variable, 'outcome', 'interaction']]
            y = data_at_time['pupil']
    
            # Initialize and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)
    
            # Store coefficients for each predictor
            coefficients.append(model.coef_)

        coefficients_df = pd.DataFrame(coefficients, columns=[variable, 'outcome', 'interaction'])
        coefficients_df['time'] = time

        # Create Plotly figure
        fig = go.Figure()

        # Add a trace for each predictor
        for predictor in [variable, 'outcome', 'interaction']:
            fig.add_trace(go.Scatter(
                x=coefficients_df['time'], 
                y=coefficients_df[predictor],
                mode='lines',
                name=f'Coefficient of {predictor}'
            ))

        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        # Update layout for better visualization
        fig.update_layout(
            title="Regression coefficients describing relationship between pupil response and computational parameters over time",
            xaxis_title="Time (s)",
            yaxis_title="Coefficient value",
            legend_title="Predictor"
        )

        # Show the plot
        fig.show()
    
    else:
        # Run regression for each time point
        for t in time_points:
            # Filter the data for this time point
            data_at_time = epoched_pupil_data_long.loc[t]
    
            # Define predictors and target variable
            X = data_at_time[variable].values.reshape(-1, 1)
            y = data_at_time['pupil']
    
            # Initialize and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)
    
            # Store coefficients for each predictor
            coefficients.append(model.coef_)

        coefficients_df = pd.DataFrame(coefficients, columns=[variable])
        coefficients_df['time'] = time

        # Create Plotly figure
        fig = go.Figure()

        # Add a trace for each predictor
        for predictor in [variable]:
            fig.add_trace(go.Scatter(
                x=coefficients_df['time'], 
                y=coefficients_df[predictor],
                mode='lines',
                name=f'Coefficient of {predictor}'
            ))

        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black", opacity=0.5)

        # Update layout for better visualization
        fig.update_layout(
            title=f"Regression coefficient describing relationship between pupil response and {variable} over time",
            xaxis_title="Time (s)",
            yaxis_title="Coefficient value",
            legend_title="Predictor"
        )

        # Show the plot
        fig.show()

    return coefficients_df





surprise_coef = pupil_regression_analysis(comp_variable_df=c1_surprise, outcome_epoched_pupil_df=c1_outcome_epoched_pupil_df, pupil_trig_df=c1_pupil_trig_df, outcome_epochs_to_remove = c1_outcome_epochs_to_remove, outcome_parameter=False, variable='surprise')

surprise_coef = pupil_regression_analysis(comp_variable_df=c2_surprise, outcome_epoched_pupil_df=c2_outcome_epoched_pupil_df, pupil_trig_df=c2_pupil_trig_df, outcome_epochs_to_remove = c2_outcome_epochs_to_remove, outcome_parameter=False, variable='surprise')


cpp_coef = pupil_regression_analysis(comp_variable_df=c1_cpp_df, outcome_epoched_pupil_df=c1_outcome_epoched_pupil_df, outcome_epochs_to_remove=c1_outcome_epochs_to_remove, pupil_trig_df=c1_pupil_trig_df, outcome_parameter=False, variable='cpp')

cpp_coef = pupil_regression_analysis(comp_variable_df=c2_cpp_df, outcome_epoched_pupil_df=c2_outcome_epoched_pupil_df, outcome_epochs_to_remove=c2_outcome_epochs_to_remove, pupil_trig_df=c2_pupil_trig_df, outcome_parameter=False, variable='cpp')


prior_conf_coef = pupil_regression_analysis(comp_variable_df=c1_prior_conf_df, outcome_epoched_pupil_df=c1_outcome_epoched_pupil_df, outcome_epochs_to_remove=c1_outcome_epochs_to_remove, pupil_trig_df=c1_pupil_trig_df, outcome_parameter=False, variable='prior_conf')

prior_conf_coef = pupil_regression_analysis(comp_variable_df=c2_prior_conf_df, outcome_epoched_pupil_df=c2_outcome_epoched_pupil_df, outcome_epochs_to_remove=c2_outcome_epochs_to_remove, pupil_trig_df=c2_pupil_trig_df, outcome_parameter=False, variable='prior_conf')