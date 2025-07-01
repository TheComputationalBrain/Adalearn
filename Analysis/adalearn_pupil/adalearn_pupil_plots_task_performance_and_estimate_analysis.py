"""
This script plots figures related to performance on the task for participants undertaking the Pupil task in the Adalearn project. The following plots are generated:
- Task session responses for an example participant alongside model estimates and parameter dynamics
- Participants' estimate absolute error and coefficient of variation
- Relationship between participants' and model estimates   
"""
import os
import pandas as pd
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

import adalearn_analysis_participant_level_functions_data_utilities as data_utils 
import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, base_dir, condition_str



dataframes = group_data_utils.load_sample_csvs(directory_path=base_dir, substrings=[
    'behavioural_df', 
    'computational_var_df', 
    'change_point_indices_df', 
    'mean_abs_error_df', 
    'behav_v_model_estimate_corr_df', 
    'behav_v_model_estimate_coef_df'
])

###########################################Plotting example task session parameters

participant = 31

data_behav = dataframes['behavioural_df'][dataframes['behavioural_df']['participant']==participant]
data_comp = dataframes['computational_var_df'][dataframes['computational_var_df']['participant']==participant]
data_change_points = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant']==participant]

data_utils.plot_session_behav_comp(participant_str=str(participant), condition_str=condition_str, directory_path=directory_path, base_dir =base_dir, behavioural_df=data_behav, computational_df=data_comp, change_point_indices=data_change_points, task_session=20, turn_on_plotting=True, plot_group_level=True)


############################################Plotting perforamnce metrics (absolute error and coefficient of variation)

dataframes['mean_abs_error_df']['participant'] = dataframes['mean_abs_error_df']['participant'].astype(int)

task_session_error_mean = dataframes['mean_abs_error_df'].groupby('task_session')['mean_abs_error'].mean()
task_session_error_count = dataframes['mean_abs_error_df'].groupby('task_session')['mean_abs_error'].count()
task_session_error_sd = dataframes['mean_abs_error_df'].groupby('task_session')['std_abs_error'].mean()
task_session_error_cv = dataframes['mean_abs_error_df'].groupby('task_session')['coef_variation'].mean()
task_session_error_score = dataframes['mean_abs_error_df'].groupby('task_session')['score'].mean()

mean_abs_error_avg_df = pd.DataFrame({
    'mean': task_session_error_mean,
    'std': task_session_error_sd,
    'cv': task_session_error_cv,
    'score': task_session_error_score,
    'count': task_session_error_count,
    'task_session': dataframes['mean_abs_error_df']['task_session'].unique()
})


fig = make_subplots(
            rows=1, 
            cols=2, 
            shared_xaxes=False, 
            vertical_spacing=0.1, 
            specs=[   
                [{}, {}]                               
            ]
)

for id in dataframes['mean_abs_error_df']['participant'].unique():
    participant_data = dataframes['mean_abs_error_df'][dataframes['mean_abs_error_df']['participant'] == id]
    
    fig.add_trace(go.Scatter(
        x=participant_data['task_session'], 
        y=participant_data['mean_abs_error'], 
        mode='lines+markers', 
        line=dict(color='rgba(0, 130, 150, 0.12)', width=1.5),
        marker=dict(color='rgba(0, 130, 150, 0.12)', size=12)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=mean_abs_error_avg_df['task_session'], 
        y=mean_abs_error_avg_df['mean'], 
        mode='lines+markers', 
        marker=dict(color='rgba(0, 130, 150, 0.8)', size=19)
    ), row=1, col=1)

    fig.update_yaxes(autorangeoptions_include=0, row=1, col=1)

for id in dataframes['mean_abs_error_df']['participant'].unique():
    participant_data = dataframes['mean_abs_error_df'][dataframes['mean_abs_error_df']['participant'] == id]
            
    fig.add_trace(go.Scatter(
        x=participant_data['task_session'], 
        y=participant_data['coef_variation'], 
        mode='lines+markers', 
        line=dict(color='rgba(0, 130, 150, 0.12)'),
        marker=dict(color='rgba(0, 130, 150, 0.12)', size=8)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=mean_abs_error_avg_df['task_session'], 
        y=mean_abs_error_avg_df['cv'], 
        mode='lines+markers', 
        marker=dict(color='rgba(0, 130, 150, 0.8)', size=16)
    ), row=1, col=2)


fig.update_yaxes(title_text="Mean absolute error", showline=True, linewidth=1.5, linecolor='black', row=1, col=1)
fig.update_yaxes(title_text="Coefficient of variation (%)", showline=True, linewidth=1.5, linecolor='black', row=1, col=2)
fig.update_xaxes(tickvals=list(range(1, 23, 2)), title_text="Task session", showline=True, linewidth=1.5, linecolor='black', row=1, col=1)
fig.update_xaxes(tickvals=list(range(1, 23, 2)), title_text="Task session", showline=True, linewidth=1.5, linecolor='black', row=1, col=2)

fig.update_layout(
    height=2*1200,
    width=1800,
    showlegend=False,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=42, family='Times New Roman'),
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_absolute_error_and_cv"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()

############################################Plotting the relationship between participants' and model estimates

# Bins and plots behavioural estimates against model estimates
# Evaluating relationship between participants' estimates and model estimates
nbins_model_estimate = 6

_, bin_edges_model_estimate = pd.qcut(dataframes['behavioural_df']["ideal_observer"], nbins_model_estimate, retbins=True)

dataframes['behavioural_df']["model_estimate_bin"] = pd.cut(dataframes['behavioural_df']["ideal_observer"], bins=bin_edges_model_estimate, labels=False, include_lowest=True)

avg_model_estimate_per_bin = dataframes['behavioural_df'].groupby("model_estimate_bin")["ideal_observer"].mean()

dataframes['behavioural_df'] = dataframes['behavioural_df'].merge(avg_model_estimate_per_bin, on="model_estimate_bin", suffixes=("", "_bin_avg"))

# Storing the average learning rate at the participant level
binned_data =  {"model_estimate_bin": [], "model_estimate_bin_label": [], "avg_estimate": [], "participant": []}

for participant in dataframes['behavioural_df']["participant"].unique():
    participant_data = dataframes['behavioural_df'][dataframes['behavioural_df']["participant"] == participant]
    
    for model_estimate_bin in sorted(participant_data["model_estimate_bin"].unique()):
            
        bin_data = participant_data[
            (participant_data["model_estimate_bin"] == model_estimate_bin)]
        
        avg_estimate = bin_data["estimate"].mean()
        bin_label = bin_data['ideal_observer_bin_avg'].unique()[0]  

        binned_data['model_estimate_bin_label'].append(bin_label)
        binned_data["model_estimate_bin"].append(model_estimate_bin)
        binned_data["avg_estimate"].append(avg_estimate)
        binned_data["participant"].append(participant)

participant_v_model_df = pd.DataFrame(binned_data)

group_avg_data = {"model_estimate_bin": [], "model_estimate_bin_label": [], "group_avg_estimate": [], "group_sem_estimate": []}

for bin in sorted(participant_v_model_df['model_estimate_bin'].unique()):
    data = participant_v_model_df[participant_v_model_df['model_estimate_bin'] == bin]
    estimate = data['avg_estimate'].mean()
    estimate_sem = stats.sem(data['avg_estimate'])
    bin_label = data['model_estimate_bin_label'].unique()[0] 

    group_avg_data["model_estimate_bin"].append(bin)
    group_avg_data["model_estimate_bin_label"].append(bin_label)
    group_avg_data["group_avg_estimate"].append(estimate)
    group_avg_data["group_sem_estimate"].append(estimate_sem)

group_behav_v_model_estimate_df = pd.DataFrame(group_avg_data)

# Plotting the relationship between participants' estimates and model estimates
fig = go.Figure()

for participant in participant_v_model_df['participant'].unique():
    data = participant_v_model_df[participant_v_model_df['participant']==participant]

    fig.add_trace(go.Scatter(
        x=data['model_estimate_bin_label'], 
        y=data['avg_estimate'], 
        mode='lines+markers',
        line=dict(color='rgba(0, 130, 150, 0.12)', width=1.5),
        marker=dict(color='rgba(0, 130, 150, 0.12)', size=12),
        showlegend=False))

fig.add_trace(go.Scatter(
    x=group_behav_v_model_estimate_df['model_estimate_bin_label'],
    y=group_behav_v_model_estimate_df['group_avg_estimate'], 
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.8)', size=19), 
    showlegend=False, 
    ))

fig.update_xaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Normative estimate", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    title_text="Participant estimate",   
)

# Add histogram to the plot
hist_trace = go.Histogram(
    x=dataframes['behav_v_model_estimate_corr_df'][dataframes['behav_v_model_estimate_corr_df']['variable']=='ideal_observer']['correlation'], 
    marker=dict(color='rgba(0, 130, 150, 0.8)'),
    xaxis='x2',  
    yaxis='y2',
    nbinsx=20,
    opacity=0.75,
    showlegend=False
)

fig.add_trace(hist_trace)

fig.update_layout(
    xaxis2=dict(
        domain=[0.7, 0.95],  
        anchor='y2',
        range=[0, 1.0],      
        title='Correlation coefficient',
        showline=True,
        linecolor='black',
        linewidth=1.5 
    ),
    yaxis2=dict(
        domain=[0.15, 0.4],  
        anchor='x2',
        title='Count',
        showline=True,
        linecolor='black',
        linewidth=1.5, 
    )
)

fig.update_layout(
    xaxis=dict(
        range=[0.1, 0.9],  
        title=dict(text="Model estimate", font=dict(size=42)),
    ),
    yaxis=dict(
        title=dict(text="Participant estimate", font=dict(size=42))
    ),
    xaxis2=dict(
        title=dict(text='Correlation coefficient', font=dict(size=42)),
        tickfont=dict(size=36),
    ),
    yaxis2=dict(
        title=dict(text='Count', font=dict(size=42)),
        tickfont=dict(size=36),
    ),
    font=dict(size=42, family='Times New Roman'), 
    height=1200,
    width=1800,
    template='simple_white',
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_behav_v_model_estimate_relationship"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()