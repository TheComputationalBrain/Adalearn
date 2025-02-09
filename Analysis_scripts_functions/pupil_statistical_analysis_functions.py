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


def pupil_regression_analysis(participant_str, condition_str, comp_variable_df, outcome_epoched_pupil_df, outcome_epochs_to_remove=None, variables=None, plot_regression_analysis=True):
    """
    Function that analyses the relationship between pupil response and a set of computational predictor variables using multiple linear regression for each peri-stimulus timepoint within defined epochs in the pupil analysis.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - comp_variable_df: pd.DataFrame, containis the computational variables (e.g. surprise, prior_conf, cpp) to be used as predictors in regression analysis.
    - outcome_epoched_pupil_df: pd.DataFrame, contains transposed epoched data (each column represents one epoch), the calculated mean, variance metrics and timestamp across epochs.
    - outcome_epochs_to_remove: list, indices of the epochs that exceed the threshold for the amount of interpolation within.
    - variables: list, names of the variables to be used as predictors in the regression analysis.
    - plot_regression_analysis: boolean, if true, plots the regression coefficients at each peri-stimulus timepoint.

    Returns:
    - coefficients_df: pd.DataFrame, contains the coefficients of the regression analysis for each predictor variable at each peri-stimulius timepoint. 
    """
    if variables is None:
        raise ValueError("You must provide at least one variable as a predictor.")
    
    # Ensure variables is a list
    if isinstance(variables, str):
        variables = [variables]

    # Prepare predictors
    predictors = comp_variable_df[variables]
    if outcome_epochs_to_remove is not None:
        predictors.drop(index=outcome_epochs_to_remove, inplace=True)

    epoched_pupil_data = outcome_epoched_pupil_df.copy()
    time = epoched_pupil_data['time']
    time_points = epoched_pupil_data.index
    epoched_pupil_data.drop(columns=['mean', 'std', 'sem', 'time'], inplace=True)

    nan_epochs = predictors[predictors.isna().any(axis=1)].index
    print(f"Removed epochs due to NaN values: {nan_epochs}")

    predictors = predictors.dropna()
    epoched_pupil_data_long = epoched_pupil_data.melt(var_name='epoch', value_name='pupil', ignore_index=False)
    epoched_pupil_data_long = epoched_pupil_data_long[~epoched_pupil_data_long['epoch'].isin(nan_epochs)]

    # Map predictors to the long-format DataFrame
    for var in variables:
        epoched_pupil_data_long[var] = epoched_pupil_data_long['epoch'].map(predictors[var])

    # Prepare storage for regression results
    coefficients = {var: [] for var in variables}

    # Run regression for each time point
    for t in time_points:
        data_at_time = epoched_pupil_data_long.loc[t]

        # Define predictors and target variable
        X = data_at_time[variables].values
        y = data_at_time['pupil']

        # Initialize and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Store coefficients for each predictor
        for idx, var in enumerate(variables):
            coefficients[var].append(model.coef_[idx])

    # Create a DataFrame for coefficients
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df['time'] = time

    # Option to plot regression coefficients at each peri-stimulus timepoint
    if plot_regression_analysis:
        fig = go.Figure()
        for var in variables:
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
            legend_title="Predictors"
        )

        # Save the figure to the desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"sub-{participant_str}_ses-{condition_str}_pupil-by-{variables}-regression.png"
        fig.write_image(os.path.join(desktop_path, file_name))  

        fig.show()

    return coefficients_df