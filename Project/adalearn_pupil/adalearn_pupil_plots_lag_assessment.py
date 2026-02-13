"""
This script plots figures for analysis examining the effect of lag on latent parameters effect on learning rate for participant undertaking the Pupil task in the Adalearn project. The following figures are generated:
- Change in correlation between each latent parameter and apparent learning rate from original to lagged
- Fraction of trials shifted for each latent parameter
- Correlation between the fraction of trials shifted for each latent parameter
- Paired differences between the two latent parameters in the change in correlation from original to lagged
- Correlation between the difference in correlation for the two latent parameters and the update frequency across participants
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR


# Loading lag update data 
dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'update_freq_df',
        'uncertainty_params_on_lr_df',
        'shifted_estimate_results',
        'shift_optimization_group_summary',
        'shift_optimization_results',
        'between_variable_shift_optimization',
        'fraction_shift_correlation',
        'update_shift_correlation',
        'lag_corr_difference_paired_ttest'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

################################################## Change in correlation from original to lag

for var in dataframes['shift_optimization_results']['latent_var'].unique():

    fig = go.Figure()

    latent_var_df = dataframes['shift_optimization_results'][dataframes['shift_optimization_results']['latent_var'] == var].copy()

    latent_var_df['participant'] = range(1, len(latent_var_df) + 1)

    if var == 'cpp':
        var_label = 'CPP'
        title_label = 'cpp'
    elif var == 'prior_conf':
        var_label = 'Prior uncertainty'
        title_label = 'prior_uncert'
    else:
        var_label = "variable"
        title_label = "variable"

    fig.add_trace(go.Scatter(
        x=latent_var_df['participant'],
        y=latent_var_df['corr_original'],
        mode='lines+markers',
        name='Original',
        line=dict(
            color='rgba(0, 130, 150, 0.8)',
            width=3
        ),
        marker=dict(size=12)
    ))

    fig.add_trace(go.Scatter(
        x=latent_var_df['participant'],
        y=latent_var_df['corr_after_shift'],
        mode='lines+markers',
        name='Shifted',
                line=dict(
            color='rgba(128, 0, 128, 0.8)',
            width=3
        ),
        marker=dict(size=12)
    ))


    fig.add_trace(go.Scatter(
        x=latent_var_df['participant'],
        y=latent_var_df['null_mean'],
        mode='lines+markers',
        name='Null',
                line=dict(
            color='rgba(220, 120, 0, 0.8)',
            width=3
        ),
        marker=dict(size=12)
    ))


    fig.update_layout(
        xaxis=dict(
            showline=True, 
            linewidth=1.5, 
            linecolor='black', 
            title_text="Participant", 
            tickvals=list(range(1, len(latent_var_df['participant'])+1, 2)),
        ),
        yaxis=dict(
            showline=True, 
            linewidth=1.5, 
            linecolor='black', 
            title_text=f"{var_label} vs. apparent learning rate (correlation)"
        ),
        font=dict(size=24, family="Times New Roman"),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        height=720,
        width=840,
        legend=dict(
            x=0.95,
            y=1.05,
            xanchor='left',
            yanchor='top'
        )
    )

    # Save the figure to the group-level folder
    plot_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Base filename (without extension)
    base_filename = f"adalearn_pupil-group_level_lag_analysis_across_participants_for_{title_label}"
    base_filepath = os.path.join(plot_dir, base_filename)

    # Save as PNG
    fig.write_image(f"{base_filepath}.png", scale=3.125)

    # Save as SVG
    fig.write_image(f"{base_filepath}.svg") 

    fig.show()

################################################## Fraction of shifted trials

fraction_shift_df_list = []

for participant in dataframes['shifted_estimate_results']['participant'].unique():

    participant_df = dataframes['shifted_estimate_results'][dataframes['shifted_estimate_results']['participant'] == participant].copy()

    cpp_shift_mean = participant_df['optimal_shift_cpp'].mean()
    prior_conf_shift_mean = participant_df['optimal_shift_prior_conf'].mean()

    fraction_shift_df = pd.DataFrame({
        'participant': [participant],
        'cpp_shift_mean': [cpp_shift_mean],
        'prior_conf_shift_mean': [prior_conf_shift_mean]
    })

    fraction_shift_df_list.append(fraction_shift_df)    

group_fraction_shift_df = pd.concat(fraction_shift_df_list, ignore_index=True)

group_fraction_shift_df['participant'] = range(1, len(group_fraction_shift_df) + 1) 

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=group_fraction_shift_df['participant'],
    y=group_fraction_shift_df['cpp_shift_mean'],
    mode='lines+markers',
    name='Change point probability',
    line=dict(
        color='rgba(220, 120, 0, 0.8)',
        width=3
    ),
    marker=dict(size=12)
))

fig.add_trace(go.Scatter(
    x=group_fraction_shift_df['participant'],
    y=group_fraction_shift_df['prior_conf_shift_mean'],
    mode='lines+markers',
    name='Prior uncertainty',
    line=dict(  
        color='rgba(0, 130, 150, 0.8)',
        width=3
    ),
    marker=dict(size=12)
))  

fig.update_layout(
    xaxis=dict(
        showline=True,
        linewidth=1.5,
        linecolor='black',
        title_text="Participant",
        tickvals=list(range(1, len(group_fraction_shift_df['participant'])+1, 2)),
    ),
    yaxis=dict(
        showline=True,
        linewidth=1.5,
        linecolor='black',
        title_text="Fraction of shifted observations"
    ),
    font=dict(size=24, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
        x=0.5,
        y=1.05,
        xanchor='left',
        yanchor='top'
    )
)


 # Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_fraction_of_shift_trials_across_participants"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


################################################ Correlation between the two parameters

fig = go.Figure()

x = group_fraction_shift_df['cpp_shift_mean']
y = group_fraction_shift_df['prior_conf_shift_mean']

# Scatter trace (participants)
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    name='participants',
    marker=dict(
        size=25,
        color='rgba(0, 130, 150, 0.5)',
    ),
    showlegend=False
))

between_subj_corr = np.corrcoef(group_fraction_shift_df['cpp_shift_mean'], group_fraction_shift_df['prior_conf_shift_mean'])[0, 1]

# Fit linear regression line
slope, intercept = np.polyfit(x, y, 1)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept

# Add regression line
fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=3,),
    showlegend=False
))


fig.update_layout(
     xaxis=dict(
            showline=True, linewidth=1.5, linecolor='black', title_text="Fraction of shifted observations - CPP"
        ),
        yaxis=dict(
            showline=True, linewidth=1.5, linecolor='black', title_text=f"Fraction of shifted observations - Prior uncertainty "
        ),
        font=dict(size=24, family="Times New Roman"),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        height=720,
        width=840,
    )

 # Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_correlation_between_parameters"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


######################################################## Paired differences between parameters
model_uncertainty_params = dataframes['uncertainty_params_on_lr_df']
shift_corr_diff_ttest = dataframes['lag_corr_difference_paired_ttest']

conf_coef = model_uncertainty_params[model_uncertainty_params['variable'] == 'conf_zscore'].sort_values('participant')
cpp_coef = model_uncertainty_params[model_uncertainty_params['variable'] == 'cpp_zscore'].sort_values('participant')


fig = go.Figure()

# reproducible jitter
np.random.seed(42)
jitter_width = 0.08  

# get latent variables in the order they appear and make numeric positions
latent_vars = list(dataframes['shift_optimization_results']['latent_var'].unique())
x_positions = {v: i for i, v in enumerate(latent_vars)}

# build readable labels in same order
labels = []
for var in latent_vars:
    if var == 'cpp':
        labels.append('Change point probability')
    elif var == 'prior_conf':
        labels.append('Prior uncertainty')
    else:
        labels.append(str(var).title())

# loop and plot
for var in latent_vars:
    var_df = dataframes['shift_optimization_results'][
        dataframes['shift_optimization_results']['latent_var'] == var
    ].copy()

    mean_var_corr_diff = np.mean(var_df['corr_diff'])
    sem_var_corr_diff = stats.sem(var_df['corr_diff'])

    # jitter around the numeric x position
    jitter = np.random.normal(loc=0.0, scale=jitter_width, size=len(var_df))
    x_numeric = x_bar + jitter

    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=var_df['corr_diff'],
        mode='markers',
        name='individuals',
        marker=dict(
            size=16,
            color='rgba(0, 130, 150, 0.3)',
        ),
        showlegend=False
    ))

    # numeric x for the bar
    x_bar = x_positions[var]

    fig.add_trace(go.Bar(
        x=[x_bar],
        y=[mean_var_corr_diff],
        width=0.6,
        name=str(var),
        error_y=dict(
            type='data',
            array=[sem_var_corr_diff],
            visible=True,
            thickness=3,
            width=10
        ),
        marker=dict(
            color='rgba(0, 130, 150, 0.4)',
            line_color='rgba(0, 130, 150, 0.8)',
            line_width=1.0
        ),
        showlegend=False
    ))

# Add a horizontal line at y=0
fig.add_hline(y=0, line_dash="dash", line_color="black")

# define the variable names to compare
var1 = "cpp"
var2 = "prior_conf"

# extract means + SEMs
conf_mean = np.mean(dataframes['shift_optimization_results']
                    .loc[dataframes['shift_optimization_results']['latent_var'] == var1, 'corr_diff'])
conf_sem = stats.sem(dataframes['shift_optimization_results']
                     .loc[dataframes['shift_optimization_results']['latent_var'] == var1, 'corr_diff'])

cpp_mean = np.mean(dataframes['shift_optimization_results']
                   .loc[dataframes['shift_optimization_results']['latent_var'] == var2, 'corr_diff'])
cpp_sem = stats.sem(dataframes['shift_optimization_results']
                    .loc[dataframes['shift_optimization_results']['latent_var'] == var2, 'corr_diff'])

# get x positions
x1 = x_positions[var1]
x2 = x_positions[var2]

# height for comparison line (slightly above both bars)
y_line = max(conf_mean + conf_sem, cpp_mean + cpp_sem) + 0.17

# add the connecting line
fig.add_shape(
    type="line",
    x0=x1, x1=x2,
    y0=y_line, y1=y_line,
    line=dict(color="black", width=2)
)

# significance star (choose * ** *** dynamically if you want)
fig.add_annotation(
    x=(x1 + x2) / 2,
    y=y_line + 0.05,
    text='*',
    showarrow=False,
    font=dict(size=50, color='black')
)

# Adding model predictions insert
# loop and plot
for var in latent_vars:
    var_df = dataframes['shift_optimization_results'][
        dataframes['shift_optimization_results']['latent_var'] == var
    ].copy()

    mean_var_corr = np.mean(var_df['corr_after_shift'])
    sem_var_corr = stats.sem(var_df['corr_after_shift'])

    # jitter around the numeric x position
    jitter = np.random.normal(loc=0.0, scale=jitter_width, size=len(var_df))
    x_numeric = x_bar + jitter

    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=var_df['corr_after_shift'],
        mode='markers',
        name='individuals',
        marker=dict(
            size=16,
            color='rgba(0, 130, 150, 0.3)',
        ),
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    ))

    # numeric x for the bar
    x_bar = x_positions[var]

    fig.add_trace(go.Bar(
        x=[x_bar],
        y=[mean_var_corr],
        width=0.6,
        name=str(var),
        error_y=dict(
            type='data',
            array=[sem_var_corr],
            visible=True,
            thickness=3,
            width=10
        ),
        marker=dict(
            color='rgba(0, 130, 150, 0.4)',
            line_color='rgba(0, 130, 150, 0.8)',
            line_width=1.0
        ),
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    ))

# Add a horizontal line at y=0
fig.add_hline(y=0, line_dash="dash", line_color="black")

# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(x_positions.values()),
        ticktext=labels,
        showline=True,
        linewidth=1.5,
        linecolor='black',
    ),
    yaxis=dict(
        range=[-0.1, 0.80], 
        showline=True, linewidth=1.5, linecolor='black',
        title_text="Shifted correlation difference to null"
    ),
    xaxis2=dict(
        domain=[0.7, 1.0],
        anchor='y2',
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickmode='array',
        tickvals=list(x_positions.values()),
        ticktext=labels,
        tickfont=dict(size=18),
        titlefont=dict(size=20)
    ),
    yaxis2=dict(
        title_text="Shifted correlation",
        domain=[0.7, 1.0],  # inset y position (change to resize/relocate)
        anchor='x2',
        showticklabels=True,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickfont=dict(size=18),
        titlefont=dict(size=20)
    ),
    font=dict(size=24, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_paired_differences_between_parameters"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

################################################ Correlation between fraction of trials shifted and update frequency
fig = go.Figure()

# Ensure participant numbering aligns between dataframes
dataframes['update_freq_df']['participant'] = range(1, len(dataframes['update_freq_df']) + 1)

dataframes['between_variable_shift_optimization']['participant'] = range(1, len(dataframes['between_variable_shift_optimization']) + 1)

merged_df = pd.merge(
    dataframes['between_variable_shift_optimization'],
    dataframes['update_freq_df'][['participant', 'avg_update_percentage']],
    on='participant'
)

variable_labels = ['cpp_shift_mean', 'prior_conf_shift_mean']

for var in variable_labels:
    # Set aesthetics by variable
    if var == 'cpp_shift_mean':
        var_label = 'Change point probability'
        title_label = 'cpp'
        point_color = 'rgba(220, 120, 0, 0.4)'
        line_color = 'rgba(220, 120, 0, 0.8)'
    elif var == 'prior_conf_shift_mean':
        var_label = 'Prior uncertainty'
        title_label = 'prior_uncert'
        point_color = 'rgba(0, 130, 150, 0.4)'
        line_color = 'rgba(0, 130, 150, 0.8)'
    else:
        print(f"Unknown latent variable: {var}")
        var_label = "variable"
        title_label = "variable"
        point_color = 'rgba(128, 128, 128, 0.4)'
        line_color = 'rgba(128, 128, 128, 0.8)'

    # Setting the axis to the same scale
    merged_df[var] = merged_df[var] * 100  # Convert to percentage

    # Compute correlation and p-value
    r, p = stats.pearsonr(merged_df['avg_update_percentage'], merged_df[var])

    # Fit regression line
    slope, intercept, _, _, _ = stats.linregress(
        merged_df['avg_update_percentage'], merged_df[var]
    )
    x_line = np.linspace(merged_df['avg_update_percentage'].min(),
                         merged_df['avg_update_percentage'].max(), 100)

    y_line = intercept + slope * x_line

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=merged_df['avg_update_percentage'],
        y=merged_df[var],
        mode='markers',
        name=f"{var_label}",
        marker=dict(size=16, color=point_color),
        showlegend=True
    ))

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f"{var_label} fit",
        line=dict(color=line_color, width=5, dash='solid'),
        showlegend=False
    ))


# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        ticktext=labels,
        title_text="Average update frequency (%)",
        showline=True,
        linewidth=1.5,
        linecolor='black',
    ),
    yaxis=dict(
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
        title_text=f"Average proportion of trials shifted (%)"
    ),
    font=dict(size=24, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
            x=0.60,
            y=0.15,
            xanchor='left',
            yanchor='top'
    )
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_correlation_between_fraction_shifted_trials_and_update_freq"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

################################################ Correlation between difference in correlation and update frequency

fig = go.Figure()

# Ensure participant numbering aligns between dataframes
dataframes['update_freq_df']['participant'] = range(1, len(dataframes['update_freq_df']) + 1)

for var in dataframes['shift_optimization_results']['latent_var'].unique():

    latent_var_df = dataframes['shift_optimization_results'][
        dataframes['shift_optimization_results']['latent_var'] == var
    ].copy()

    latent_var_df['participant'] = range(1, len(latent_var_df) + 1)

    merged_df = pd.merge(
        latent_var_df[['participant', 'corr_diff']],
        dataframes['update_freq_df'][['participant', 'avg_update_percentage']],
        on='participant'
    )

    # Set aesthetics by variable
    if var == 'cpp':
        var_label = 'Change point probability'
        title_label = 'cpp'
        point_color = 'rgba(220, 120, 0, 0.4)'
        line_color = 'rgba(220, 120, 0, 0.8)'
        annotation_y = 0.99
    elif var == 'prior_conf':
        var_label = 'Prior uncertainty'
        title_label = 'prior_uncert'
        point_color = 'rgba(0, 130, 150, 0.4)'
        line_color = 'rgba(0, 130, 150, 0.8)'
        annotation_y = 0.95
    else:
        print(f"Unknown latent variable: {var}")
        var_label = "variable"
        title_label = "variable"
        point_color = 'rgba(128, 128, 128, 0.4)'
        line_color = 'rgba(128, 128, 128, 0.8)'
        annotation_y = 1.05

    # Compute correlation and p-value
    r, p = stats.pearsonr(merged_df['avg_update_percentage'], merged_df['corr_diff'])

    # Fit regression line
    slope, intercept, _, _, _ = stats.linregress(
        merged_df['avg_update_percentage'], merged_df['corr_diff']
    )
    x_line = np.linspace(merged_df['avg_update_percentage'].min(),
                         merged_df['avg_update_percentage'].max(), 100)
    y_line = intercept + slope * x_line

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=merged_df['avg_update_percentage'],
        y=merged_df['corr_diff'],
        mode='markers',
        name=f"{var_label}",
        marker=dict(size=16, color=point_color),
        showlegend=True
    ))

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f"{var_label} fit",
        line=dict(color=line_color, width=5, dash='solid'),
        showlegend=False
    ))

# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        ticktext=labels,
        title_text="Average update frequency (%)",
        showline=True,
        linewidth=1.5,
        linecolor='black',
    ),
    yaxis=dict(
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
        title_text=f"Shifted correlation difference to null"
    ),
    font=dict(size=24, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
            x=0.60,
            y=0.15,
            xanchor='left',
            yanchor='top'
    )
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_correlation_between_corr_diff_and_update_freq"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

