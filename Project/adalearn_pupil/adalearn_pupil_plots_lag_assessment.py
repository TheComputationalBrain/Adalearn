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
        'lag_corr_difference_paired_ttest',
        'shifted_ideal_observer_results',
        'shift_optimization_IO_group_summary',
        'shift_optimization_IO_results',
        'null_shifted_vectors_summary',
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
        var_label = 'Prior uncert'
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
            width=4
        ),
        marker=dict(size=16)
    ))

    fig.add_trace(go.Scatter(
        x=latent_var_df['participant'],
        y=latent_var_df['corr_after_shift'],
        mode='lines+markers',
        name='Shifted',
                line=dict(
            color='rgba(128, 0, 128, 0.8)',
            width=4
        ),
        marker=dict(size=16)
    ))


    fig.add_trace(go.Scatter(
        x=latent_var_df['participant'],
        y=latent_var_df['null_mean'],
        mode='lines+markers',
        name='Null',
                line=dict(
            color='rgba(220, 120, 0, 0.8)',
            width=4
        ),
        marker=dict(size=16)
    ))


    fig.update_layout(
        xaxis=dict(
            showline=True, 
            linewidth=3, 
            linecolor='black', 
            title_text="Participant", 
            tickvals=list(range(1, len(latent_var_df['participant'])+1, 5)),
            ticks="inside",
            tickwidth=1.5,
            tickcolor='black',
        ),
        yaxis=dict(
            showline=True, 
            linewidth=3, 
            linecolor='black', 
            title_text=f"{var_label} vs. apparent learning rate <br>(correlation)",
            ticks="inside",
            tickwidth=1.5,
            tickcolor='black',
        ),
        font=dict(size=30, family="Times New Roman"),
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
    fig.write_image(f"{base_filepath}.svg", scale=3.125) 

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
        width=4
    ),
    marker=dict(size=16)
))

fig.add_trace(go.Scatter(
    x=group_fraction_shift_df['participant'],
    y=group_fraction_shift_df['prior_conf_shift_mean'],
    mode='lines+markers',
    name='Prior uncertainty',
    line=dict(  
        color='rgba(0, 130, 150, 0.8)',
        width=4
    ),
    marker=dict(size=16)
))  

fig.update_layout(
    xaxis=dict(
        showline=True,
        linewidth=3,
        linecolor='black',
        title_text="Participant",
        tickvals=list(range(1, len(group_fraction_shift_df['participant'])+1, 5)),
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    yaxis=dict(
        showline=True,
        linewidth=3,
        linecolor='black',
        title_text="Fraction of shifted observations",
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    font=dict(size=30, family="Times New Roman"),
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
    line=dict(color='rgba(0, 130, 150, 0.8)', width=4),
    showlegend=False
))


fig.update_layout(
     xaxis=dict(
            showline=True, 
            linewidth=3, 
            linecolor='black', 
            title_text="Fraction of shifted observations - CPP",
            ticks="inside",
            tickwidth=1.5,
            tickcolor='black',
        ),
        yaxis=dict(
            showline=True, 
            linewidth=3, 
            linecolor='black', 
            title_text=f"Fraction of shifted observations - Prior uncert",
            ticks="inside",
            tickwidth=1.5,
            tickcolor='black',
        ),
        font=dict(size=30, family="Times New Roman"),
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
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()


######################################################## Paired differences between parameters
results = dataframes['shift_optimization_results']

fig = go.Figure()

# reproducible jitter
np.random.seed(42)
jitter_width = 0.08

# latent variables in order of appearance -> numeric positions
latent_vars = list(results['latent_var'].unique())
x_positions = {v: i for i, v in enumerate(latent_vars)}

# readable labels
label_map = {'cpp': 'Change point probability', 'prior_conf': 'Prior uncertainty'}
labels = [label_map.get(v, str(v).title()) for v in latent_vars]

# ---------- bars, points, one-sample t-tests vs zero, stars above bars ----------
star_y_positions = {}   # remember how high each star sits, to place the bracket above
for var in latent_vars:
    var_df = results[results['latent_var'] == var].copy()
    vals = var_df['corr_diff'].dropna()

    mean_val = np.mean(vals)
    sem_val = stats.sem(vals)
    x_bar = x_positions[var]

    # one-sample t-test against zero
    t_stat, p_val = stats.ttest_1samp(vals, popmean=0)
    print(f"{var} vs 0: t({len(vals) - 1}) = {t_stat:.3f}, p = {p_val:.4g}")

    # jittered individual points
    jitter = np.random.normal(loc=0.0, scale=jitter_width, size=len(var_df))
    fig.add_trace(go.Scatter(
        x=x_bar + jitter,
        y=var_df['corr_diff'],
        mode='markers',
        name='individuals',
        marker=dict(size=16, color='rgba(0, 130, 150, 0.3)'),
        showlegend=False
    ))

    # bar with SEM
    fig.add_trace(go.Bar(
        x=[x_bar],
        y=[mean_val],
        width=0.6,
        name=str(var),
        error_y=dict(type='data', array=[sem_val], visible=True, thickness=3, width=10),
        marker=dict(
            color='rgba(0, 130, 150, 0.4)',
            line_color='rgba(0, 130, 150, 0.8)',
            line_width=1.0
        ),
        showlegend=False
    ))

    # star just above the highest element of this bar (error bar or top data point)
    top = max(mean_val + sem_val, vals.max())
    y_star = top + 0.02
    star_y_positions[var] = y_star

    fig.add_annotation(
        x=x_bar,
        y=y_star,
        text='*',
        showarrow=False,
        yanchor='bottom',
        font=dict(size=38, color='black')
    )

# horizontal line at y=0
fig.add_hline(y=0, line_dash="dash", line_color="black")

# ---------- between-parameter comparison (paired t-test) ----------
var1, var2 = "cpp", "prior_conf"

wide = results.pivot_table(index='participant', columns='latent_var', values='corr_diff')
wide = wide[[var1, var2]].dropna()
ttest_between = stats.ttest_rel(wide[var1], wide[var2])
p_between = ttest_between.pvalue
print(f"{var1} vs {var2}: t({len(wide) - 1}) = {ttest_between.statistic:.3f}, p = {p_between:.4g}")



x1, x2 = x_positions[var1], x_positions[var2]

# bracket sits clearly above both per-bar stars
y_line = max(star_y_positions[var1], star_y_positions[var2]) + 0.20

fig.add_shape(
    type="line",
    x0=x1, x1=x2, y0=y_line, y1=y_line,
    line=dict(color="black", width=3)
)

fig.add_annotation(
    x=(x1 + x2) / 2,
    y=y_line + 0.02,
    text='*',
    showarrow=False,
    yanchor='bottom',
    font=dict(size=38, color='black')
)

# make sure the y-axis is tall enough to show the bracket and its star
y_max = max(0.8, y_line + 0.12)

# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(x_positions.values()),
        ticktext=labels,
        showline=True,
        linewidth=3,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    yaxis=dict(
        range=[-0.1, y_max],
        showline=True,
        linewidth=3,
        linecolor='black',
        title_text="Shifted correlation difference to null",
        tickvals=[0, 0.2, 0.4, 0.6, 0.8],
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    font=dict(size=30, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840
)

# save
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

base_filename = "adalearn_pupil-group_level_lag_analysis_paired_differences_between_parameters"
base_filepath = os.path.join(plot_dir, base_filename)

fig.write_image(f"{base_filepath}.png", scale=3.125)
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
        line=dict(color=line_color, width=4, dash='solid'),
        showlegend=False
    ))


# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        ticktext=labels,
        title_text="Average update frequency (%)",
        showline=True,
        linewidth=3,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
        
    ),
    yaxis=dict(
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        title_text=f"Average proportion of trials shifted (%)",
        tickvals = list(range(36, 54, 4)),
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    font=dict(size=30, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
            x=0.50,
            y=0.21,
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
        line=dict(color=line_color, width=4, dash='solid'),
        showlegend=False
    ))

# layout
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        ticktext=labels,
        title_text="Average update frequency (%)",
        showline=True,
        linewidth=3,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    yaxis=dict(
        showline=True, 
        linewidth=3, 
        linecolor='black', 
        title_text=f"Shifted correlation difference to null",
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',    
    ),
    font=dict(size=30, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
            x=0.7,
            y=0.21,
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


################################################### Comparison of shifted trials between optimized parameters


participant_data = dataframes['shifted_estimate_results']
io_data = dataframes['shifted_ideal_observer_results']
null_data = dataframes['null_shifted_vectors_summary']


prop_trial_shift_results = {'participant': [], 'subj_cpp_trial_shift': [], 'io_cpp_trial_shift': [], 'subj_conf_trial_shift': [], 'io_conf_trial_shift': [], 'null_cpp_trial_shift': [], 'null_conf_trial_shift': []}

for participant in participant_data['participant'].unique():
    participant_df = participant_data[participant_data['participant'] == participant]
    io_df = io_data[io_data['participant'] == participant]
    null_data_df = null_data[null_data['participant'] == participant]

    # CPP

    subj_cpp_trial_shift = (participant_df['optimal_shift_cpp'].sum() / len(participant_df['optimal_shift_cpp'].dropna())) * 100

    io_cpp_trial_shift = (io_df['optimal_shift_cpp'].sum() / len(io_df['optimal_shift_cpp'].dropna())) * 100

    # Conf

    subj_conf_trial_shift = (participant_df['optimal_shift_prior_conf'].sum() / len(participant_df['optimal_shift_prior_conf'].dropna())) * 100

    io_conf_trial_shift = (io_df['optimal_shift_prior_conf'].sum() / len(io_df['optimal_shift_prior_conf'].dropna())) * 100

    # Append results
    prop_trial_shift_results['participant'].append(participant)
    prop_trial_shift_results['subj_cpp_trial_shift'].append(subj_cpp_trial_shift)
    prop_trial_shift_results['io_cpp_trial_shift'].append(io_cpp_trial_shift)
    prop_trial_shift_results['subj_conf_trial_shift'].append(subj_conf_trial_shift)
    prop_trial_shift_results['io_conf_trial_shift'].append(io_conf_trial_shift)

    # Null
    for condition in null_data_df['condition'].unique():

        null_condition_df = null_data_df[null_data_df['condition'] == condition]

        null_trial_shift = (null_condition_df['prop_shifted'].mean()) * 100

        if condition == 'cpp':
            prop_trial_shift_results['null_cpp_trial_shift'].append(null_trial_shift)
        elif condition == 'prior_conf':
            prop_trial_shift_results['null_conf_trial_shift'].append(null_trial_shift)  


trial_shift_results_df = pd.DataFrame(prop_trial_shift_results)

columns = ['subj_cpp_trial_shift', 'io_cpp_trial_shift',
       'subj_conf_trial_shift', 'io_conf_trial_shift', 'null_cpp_trial_shift',
       'null_conf_trial_shift']

for column in trial_shift_results_df.columns:
    if column in columns:
        print(f'{column}: {trial_shift_results_df[column].mean()} +/- {trial_shift_results_df[column].std()}')


def rgba(rgb, alpha):
    r, g, b = rgb
    return f'rgba({r},{g},{b},{alpha})'

groups = ['Change point probability', 'Prior uncertainty']
series = {
    'Participant':    {'Change point probability': 'subj_cpp_trial_shift', 'Prior uncertainty': 'subj_conf_trial_shift', 'rgb': (0, 130, 150)},
    'Ideal observer': {'Change point probability': 'io_cpp_trial_shift',   'Prior uncertainty': 'io_conf_trial_shift',   'rgb': (179, 1, 1)},
    'Null':           {'Change point probability': 'null_cpp_trial_shift', 'Prior uncertainty': 'null_conf_trial_shift', 'rgb': (255, 127, 14)},
}

bar_alpha = 0.5     # bar fill
line_alpha = 0.8    # bar outline
point_alpha = 0.3  # individual points (lower than bars)

bar_width = 0.25
offsets = {'Participant': -bar_width, 'Ideal observer': 0.0, 'Null': bar_width}
group_pos = {g: i for i, g in enumerate(groups)}

fig = go.Figure()
rng = np.random.default_rng(0)
error_traces = []  # added last so they sit on top

for name, cfg in series.items():
    xs = [group_pos[g] + offsets[name] for g in groups]
    cols = [cfg[g] for g in groups]
    means = [trial_shift_results_df[c].mean() for c in cols]
    sems = [trial_shift_results_df[c].sem() for c in cols]

    # Bars (no error bars here)
    fig.add_trace(go.Bar(
        x=xs, y=means, width=bar_width * 0.9,
        name=name,
        marker=dict(color=rgba(cfg['rgb'], bar_alpha),
                    line_color=rgba(cfg['rgb'], line_alpha),
                    line_width=1.0))
    )
    

    # Individual points, jittered around each bar's center
    px, py, ptxt = [], [], []
    for g, c in zip(groups, cols):
        vals = trial_shift_results_df[c]
        jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(vals))
        px.extend(group_pos[g] + offsets[name] + jitter)
        py.extend(vals.tolist())
        ptxt.extend(trial_shift_results_df['participant'].astype(str).tolist())

    fig.add_trace(go.Scatter(
        x=px, y=py, mode='markers',
        marker=dict(
            color=rgba(cfg['rgb'], point_alpha),
            size=15,
            line=dict(color='rgba(0,0,0,0)', width=1),
        ),
        text=ptxt,
        hovertemplate='ID %{text}<br>%{y:.1f}%<extra></extra>',
        showlegend=False,
    ))

    # SEM error bars stored to draw on top of everything
    error_traces.append(go.Scatter(
        x=xs, y=means, mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=1),
        error_y=dict(
            type='data', array=sems, visible=True,
            color='rgba(0,0,0,1)', thickness=1.5, width=8,
        ),
        hoverinfo='skip',
        showlegend=False,
    ))

for trace in error_traces:
    fig.add_trace(trace)

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(group_pos.values()),
        ticktext=groups,
        title='Optimized parameter',
        showline=True,
        linewidth=3,
        linecolor='black',
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',
    ),
    yaxis=dict(
        showline=True, 
        linewidth=3,
        linecolor='black', 
        title_text=f"Percentage of trials shifted (%)",
        ticks="inside",
        tickwidth=1.5,
        tickcolor='black',    
    ),
    font=dict(size=28, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
            x=0.7,
            y=1.20,
            xanchor='left',
            yanchor='top'
    )
)


# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_lag_analysis_comparison_of_shifted_trials_between_optimized_parameters"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()