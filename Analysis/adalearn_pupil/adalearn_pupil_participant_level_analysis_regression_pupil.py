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


from adalearn_analysis_participant_level_functions_regression_pupil import pupil_regression_analysis, pupil_uncertainty_lr_mediation_analysis

from adalearn_pupil.adalearn_pupil_participant_level_analysis_pupil import pupil_trig_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_str, from_session, to_session, task_sessions



conf_cpp_coef, pupil_regress_df, residuals_df = pupil_regression_analysis(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals'], covariates = ['outcome_num', 'task_session'], include_outcome=True, plot_regression_analysis=False)


conf_cpp_coef_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_regression_analysis(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals'], covariates = ['outcome_num', 'task_session'], include_outcome=True, plot_regression_analysis=False)



mediation_output = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=True, plot_mediation_analysis=False)


mediation_output_nobc = pupil_uncertainty_lr_mediation_analysis(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore'], baseline_corrected=False, plot_mediation_analysis=False)



participant_str=participant_str
condition_str=condition_str
computational_var_df=computational_var_df
outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc
outcome_epochs_to_remove=outcome_epochs_to_remove_nobc
variables=['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals']
covariates=['cum_outcome_num', 'outcome_num', 'task_session']
include_outcome=True

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


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

predictors = predictors[predictors['outcome_num'] > 15]

nan_epochs = predictors[predictors.isna().any(axis=1)].index
predictors = predictors.dropna()

epoched_pupil_data = outcome_epoched_pupil_df_nobc.copy()
time = epoched_pupil_data['time']
time_points = epoched_pupil_data.index

valid_columns = predictors['cum_outcome_num'].astype(int).tolist()

df_filtered = epoched_pupil_data.iloc[:, epoched_pupil_data.columns.isin(valid_columns)]

epoched_pupil_data_long = df_filtered.melt(var_name='epoch', value_name='pupil', ignore_index=False)

epoched_pupil_data_long = epoched_pupil_data_long[~epoched_pupil_data_long['epoch'].isin(nan_epochs)]

for var in predictors.columns:
    epoched_pupil_data_long[var] = epoched_pupil_data_long['epoch'].map(predictors[var])

epoched_pupil_data_long = epoched_pupil_data_long.drop(columns=['cum_outcome_num', 'outcome_num', 'task_session'])
predictors = predictors.drop(columns=['cum_outcome_num', 'outcome_num', 'task_session'])

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

fig.show()




pupil_baseline_df = outcome_epoched_pupil_df_nobc.copy()
pupil_baseline_df = pupil_baseline_df.drop(columns=['mean', 'std', 'sem'])
pupil_baseline_df = pupil_baseline_df.set_index('time')

filtered_pupil_data = pupil_baseline_df.loc[(pupil_baseline_df.index >= -0.4) & (pupil_baseline_df.index <= 0)]

baseline_means = filtered_pupil_data.mean()

baseline_means_df = pd.DataFrame({
    'baseline_means': baseline_means, 
    'task_session': (baseline_means.index // 75) + 1
    })


unique_sessions = baseline_means_df['task_session'].unique()  
color_map = {session: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, session in enumerate(unique_sessions)}

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=baseline_means_df.index, 
    y=baseline_means_df['baseline_means'], 
    mode='markers', 
    marker=dict(color=[color_map[i] for i in baseline_means_df['task_session']]),  
    name='Baseline'
))

fig.update_layout(
    title=f'Pupil baseline means across trials (identified by task sessions) for particpant {participant_str}',
    xaxis_title='Trial number',
    yaxis_title='Pupil Size (mm)',
    showlegend=False
)

fig.show()


num_sessions = len(unique_sessions)

fig = make_subplots(rows=num_sessions, cols=1, shared_xaxes=False,
                    subplot_titles=[f'Mean baseline (non-corrected) pupil diameter in task session {session}' for session in unique_sessions])

# Add pupil traces to each subplot
for i, session in enumerate(unique_sessions, start=1):
    session_data = baseline_means_df[baseline_means_df['task_session'] == session]

    fig.add_trace(go.Scatter(x=session_data.index, y=session_data['baseline_means'], mode='lines+markers', name='Baseline Pupil Diameter (all data)', showlegend=False, line=dict(color='rgba(0, 0, 255, 0.2)')), row=i, col=1)

    fig.add_trace(go.Scatter(x=session_data.index[10:], y=session_data['baseline_means'].iloc[10:], mode='lines+markers', name='Baseline Pupil Diameter (reduced data)', showlegend=False, line=dict(color='rgba(255, 0, 0, 0.2)')), row=i, col=1)

fig.update_layout(
    height=900*num_sessions,
    width=1200,
    title = f'Pupil baseline means for trials within each task sessions for particpant {participant_str}',
    xaxis_title='Trial number',
    yaxis_title='Pupil Size (mm)',
    showlegend=False)

fig.show()

