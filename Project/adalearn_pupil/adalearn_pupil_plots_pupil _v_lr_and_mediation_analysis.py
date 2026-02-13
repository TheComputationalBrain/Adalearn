"""
The script plots figure examining the association between pupil responses and learning rate and the mediating influence of pupil responses on the relationship between prior uncertainty and learning rate for participant perfoming the Pupil task in the Adalearn project. The following plots are generated:
- Baseline corrected pupil signal dynamics around change points
- Non-baseline corrected pupil signal dynamics around change points
- Regression analysis on the effect of pupil signals on learning rates
- Mediating influence of non-baseline corrected pupil signal on the relationship between prior uncertainty and learning rate
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as pupil_analysis
from adalearn_analysis_participant_level_functions_comp import variables_by_change_point

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'computational_var_df',
        'change_point_indices_df',
        'pupil_v_lr_regress_coef_df',
        'pupil_v_lr_coef_ttest_df',
        'pupil_bc_around_cp',
        'pupil_nobc_around_cp',
        'pupil_lr_uncert_nobc_mediation',
        'pupil_lr_uncert_bc_mediation',
        'exclude_participants_pupil_analysis_df', 
        'outcome_epochs_time_df', 
        'outcome_epoched_pupil_df',
        'outcome_epochs_columns_df', 
        'outcome_epochs_to_remove_df',
        'outcome_epochs_time_nobc_df', 
        'outcome_epoched_pupil_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc',
        'pupil_v_lr_coef_ttest_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

##########################################################Plotting baseline-corrected and non-baseline corrected pupil effect window signals around change points
pupil_bc_around_cp = dataframes['pupil_bc_around_cp'].copy()
pupil_nobc_around_cp = dataframes['pupil_nobc_around_cp'].copy()
change_point_indices_df = dataframes['change_point_indices_df'].copy()


grouped_pupil_bc_cp_pos_dict = {'rel_pos': [], 'pupil_bc_mean': [], 'pupil_bc_std': [], 'pupil_bc_sem': []} 

for i in pupil_bc_around_cp['stimulus_rel_cp'].unique():
    data = pupil_bc_around_cp[pupil_bc_around_cp['stimulus_rel_cp'] == i]
   
    pupil_bc_mean = data['mean'].mean()
    pupil_bc_std = data['mean'].std()
    pupil_bc_sem = stats.sem(data['mean'])

    grouped_pupil_bc_cp_pos_dict['rel_pos'].append(i)
    grouped_pupil_bc_cp_pos_dict['pupil_bc_mean'].append(pupil_bc_mean)
    grouped_pupil_bc_cp_pos_dict['pupil_bc_std'].append(pupil_bc_std)
    grouped_pupil_bc_cp_pos_dict['pupil_bc_sem'].append(pupil_bc_sem)

grouped_pupil_bc_cp_rel_pos_df = pd.DataFrame(grouped_pupil_bc_cp_pos_dict)


grouped_pupil_nobc_cp_pos_dict = {'rel_pos': [], 'pupil_nobc_mean': [], 'pupil_nobc_std': [], 'pupil_nobc_sem': []} 

for i in pupil_nobc_around_cp['stimulus_rel_cp'].unique():
    data = pupil_nobc_around_cp[pupil_nobc_around_cp['stimulus_rel_cp'] == i]
   
    pupil_nobc_mean = data['mean'].mean()
    pupil_nobc_std = data['mean'].std()
    pupil_nobc_sem = stats.sem(data['mean'])

    grouped_pupil_nobc_cp_pos_dict['rel_pos'].append(i)
    grouped_pupil_nobc_cp_pos_dict['pupil_nobc_mean'].append(pupil_nobc_mean)
    grouped_pupil_nobc_cp_pos_dict['pupil_nobc_std'].append(pupil_nobc_std)
    grouped_pupil_nobc_cp_pos_dict['pupil_nobc_sem'].append(pupil_nobc_sem)

grouped_pupil_nobc_cp_rel_pos_df = pd.DataFrame(grouped_pupil_nobc_cp_pos_dict)

# getting computational variables for inserts
averaged_cp_df = []
averaged_conf_df = []

for participant in dataframes['computational_var_df']['participant'].unique():
    print('Processing participant:', participant)

    variable_data = dataframes['computational_var_df'][dataframes['computational_var_df']['participant'] == participant]
    change_points = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant]

    _, averaged_values_cpp = variables_by_change_point(participant_str=participant,
                                                                        condition_str='01',
                                                                        directory_path=BASE_DIR,
                                                                        variable_data=variable_data,
                                                                        change_point_indices=change_points,
                                                                        parameter='cpp',
                                                                        pre=5,
                                                                        post=5,
                                                                        exclude_zero=False)
    _, averaged_values_conf = variables_by_change_point(participant_str=participant,
                                                                        condition_str='01',
                                                                        directory_path=BASE_DIR,
                                                                        variable_data=variable_data,
                                                                        change_point_indices=change_points,
                                                                        parameter='prior_conf',
                                                                        pre=5,
                                                                        post=5,
                                                                        exclude_zero=False)
    
    averaged_values_cpp['participant'] = participant
    averaged_cp_df.append(averaged_values_cpp)

    averaged_values_conf['participant'] = participant
    averaged_conf_df.append(averaged_values_conf)   

round_cp_averaged_cpp_df = pd.concat(averaged_cp_df, ignore_index=True)
round_cp_averaged_conf_df = pd.concat(averaged_conf_df, ignore_index=True)

round_cp_data = {
    'cpp': round_cp_averaged_cpp_df,
    'prior_conf': round_cp_averaged_conf_df
}

cpp_data_dict = {}
conf_data_dict = {}

for variable, df in round_cp_data.items():
    for pos in df['relative_position'].unique():
        pos_data = df[df['relative_position'] == pos]

        pos_data_mean = pos_data['mean'].mean()
        pos_data_sem = stats.sem(pos_data['mean'])

        if variable == 'cpp':
            cpp_data_dict[pos] = {
                'mean': pos_data_mean,
                'sem': pos_data_sem
            }
        elif variable == 'prior_conf':
            conf_data_dict[pos] = {
                'mean': pos_data_mean,
                'sem': pos_data_sem
            }

cpp_round_cp_df = pd.DataFrame.from_dict(cpp_data_dict, orient='index').reset_index()
cpp_round_cp_df.columns = ['position', 'mean', 'sem']
cpp_round_cp_df = cpp_round_cp_df.sort_values('position').reset_index(drop=True)

conf_round_cp_df = pd.DataFrame.from_dict(conf_data_dict, orient='index').reset_index()
conf_round_cp_df.columns = ['position', 'mean', 'sem']
conf_round_cp_df = conf_round_cp_df.sort_values('position').reset_index(drop=True)


############################################################ Phasic-associated pupil effect window
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.concat([
        grouped_pupil_bc_cp_rel_pos_df['rel_pos'], 
        grouped_pupil_bc_cp_rel_pos_df['rel_pos'][::-1]
    ]),
    y=pd.concat([
        grouped_pupil_bc_cp_rel_pos_df['pupil_bc_mean'] - grouped_pupil_bc_cp_rel_pos_df['pupil_bc_sem'],
        (grouped_pupil_bc_cp_rel_pos_df['pupil_bc_mean'] + grouped_pupil_bc_cp_rel_pos_df['pupil_bc_sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(128, 0, 128, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=grouped_pupil_bc_cp_rel_pos_df['rel_pos'],
    y=grouped_pupil_bc_cp_rel_pos_df['pupil_bc_mean'],
    mode='lines',
    line=dict(color='rgba(128, 0, 128, 0.8)', width=3),
    name='Pupil (Baseline corrected effect window)',
    showlegend=True
))

fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                line=dict(dash="dot", color="black"), 
                xref="x", yref="y domain")
    

fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=1,
    linecolor='black',
    tickmode='array',
    tickvals=list(range(-5, 6)),
    ticktext=["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5", "6"],
    range=[-5.5, 6.5]
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    range=[-0.05, 0.18], 
    title_text='Phasic pupil effect signal (z-score)')

# Add the inset (e.g., CPP variable)
fig.add_trace(go.Scatter(
    x=cpp_round_cp_df['position'],
    y=cpp_round_cp_df['mean'],
    mode='lines',
    line=dict(color='rgba(200, 0, 0, 0.8)', width=3),
    name='CPP',
    xaxis='x2',
    yaxis='y2'
))

# Add SEM shading in the inset
fig.add_trace(go.Scatter(
    x=pd.concat([cpp_round_cp_df['position'], cpp_round_cp_df['position'][::-1]]),
    y=pd.concat([cpp_round_cp_df['mean'] - cpp_round_cp_df['sem'], (cpp_round_cp_df['mean'] + cpp_round_cp_df['sem'])[::-1]]),
    fill='toself',
    fillcolor='rgba(200, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255, 0)'),
    hoverinfo="skip",
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Vertical line at change point (0) in model subplot
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=cpp_round_cp_df['mean'].min() - 0.01,
    y1=cpp_round_cp_df['mean'].max() + 0.01,
    line=dict(color="black", width=2, dash="dot"),
    xref="x2", yref="y2"
)

fig.update_layout(
    xaxis2=dict(
        domain=[0.12, 0.42], 
        anchor='y2',
        range=[-5.5, 6.5],
        showticklabels=False,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickvals=list(range(-5, 6)),
        #title=dict(text="Outcomes relative to change point", font=dict(size=20)),
        #ticktext=["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5", "6"],
        tickfont=dict(size=18),
    ),
    yaxis2=dict(
        domain=[0.7, 1.0],  
        anchor='x2',
        showticklabels=False,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        dtick=0.02,
        title=dict(text="CPP", font=dict(size=20)),
        tickfont=dict(size=20),
    ),
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=24, family="Times New Roman"), 
    showlegend=False
)


# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_bc_pupil_responses_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 


fig.show()

######################################################## Non-baseline corrected pupil effect window
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.concat([
        grouped_pupil_nobc_cp_rel_pos_df['rel_pos'], 
        grouped_pupil_nobc_cp_rel_pos_df['rel_pos'][::-1]
    ]),
    y=pd.concat([
        grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_mean'] - grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_sem'],
        (grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_mean'] + grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_sem'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=grouped_pupil_nobc_cp_rel_pos_df['rel_pos'],
    y=grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_mean'],
    mode='lines',
    line=dict(color='rgba(255, 165, 0, 0.8)', width=3),
    name='Pupil (Non-baseline corrected effect window)',
    showlegend=True
))

fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                line=dict(dash="dot", color="black"), 
                xref="x", yref="y domain")
    

fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=1,
    linecolor='black',
    tickmode='array',
    tickvals=list(range(-5, 6)),
    ticktext=["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5", "6"],
    range=[-5.5, 6.5]
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black',
    range=[-0.18, 0.15], 
    title_text='Tonic pupil effect signal (z-score)')

# Add the inset (e.g., prior uncertainty variable)
fig.add_trace(go.Scatter(
    x=conf_round_cp_df['position'],
    y=conf_round_cp_df['mean'],
    mode='lines',
    line=dict(color='rgba(200, 0, 0, 0.8)', width=3),
    name='Prior uncert',
    xaxis='x2',
    yaxis='y2'
))

# Add SEM shading in the inset
fig.add_trace(go.Scatter(
    x=pd.concat([conf_round_cp_df['position'], conf_round_cp_df['position'][::-1]]),
    y=pd.concat([conf_round_cp_df['mean'] - conf_round_cp_df['sem'], (conf_round_cp_df['mean'] + conf_round_cp_df['sem'])[::-1]]),
    fill='toself',
    fillcolor='rgba(200, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255, 0)'),
    hoverinfo="skip",
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Vertical line at change point (0) in model subplot
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=conf_round_cp_df['mean'].min() - 0.01,
    y1=conf_round_cp_df['mean'].max() + 0.01,
    line=dict(color="black", width=2, dash="dot"),
    xref="x2", yref="y2"
)

fig.update_layout(
    xaxis2=dict(
        domain=[0.12, 0.42], 
        anchor='y2',
        range=[-5.5, 6.5],
        showticklabels=False,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        tickvals=list(range(-5, 6)),
        #title=dict(text="Outcomes relative to change point", font=dict(size=30)),
        #ticktext=["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5", "6"],
        tickfont=dict(size=20),
    ),
    yaxis2=dict(
        domain=[0.70, 1.0],  
        anchor='x2',
        showticklabels=False,
        showline=True,
        linecolor='black',
        linewidth=1.5,
        dtick=0.02,
        title=dict(text="Prior uncert", font=dict(size=20)),
        tickfont=dict(size=20),
    ),
    height=720, 
    width=840, 
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    font=dict(size=24, family="Times New Roman"), 
    showlegend=False
)

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_nobc_pupil_responses_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 


fig.show()

##################################################################################pupil effects on learning rate 
pupil_v_lr_regress_df = dataframes['pupil_v_lr_regress_coef_df']
pupil_v_lr_ttest_df = dataframes['pupil_v_lr_coef_ttest_df']

pupil_bc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_bc_zscore'].sort_values('participant')
pupil_nobc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_nobc_zscore'].sort_values('participant')


pupil_v_lr_bc_mean = pupil_bc_coef['coefficient'].mean()
pupil_v_lr_nobc_mean = pupil_nobc_coef['coefficient'].mean()
pupil_v_lr_bc_sem = stats.sem(pupil_bc_coef['coefficient'])
pupil_v_lr_nobc_sem = stats.sem(pupil_nobc_coef['coefficient'])


comp_df = dataframes['computational_var_df']
epoch_data = dataframes['outcome_epoched_pupil_df']
epoch_data_nobc = dataframes['outcome_epoched_pupil_nobc_df']
epoch_times = dataframes['outcome_epochs_time_df']
analysed_epochs = dataframes['outcome_epochs_columns_df']
epochs_to_remove = dataframes['outcome_epochs_to_remove_df']

bin_df = []

for participant in epoch_data['participant'].unique():
    comp_data = comp_df[comp_df['participant'] == participant].reset_index()

    pupil_data = epoch_data[epoch_data['participant'] == participant].drop(columns=['participant']).T
    column_titles = analysed_epochs[analysed_epochs['participant'] == participant]['columns'].values.astype(int)
    pupil_data.columns = column_titles
    pupil_data['time'] = epoch_times[epoch_times['participant'] == participant]['time'].values.astype(float)

    pupil_data_nobc = epoch_data_nobc[epoch_data_nobc['participant'] == participant].drop(columns=['participant']).T
    pupil_data_nobc.columns = column_titles
    pupil_data_nobc['time'] = epoch_times[epoch_times['participant'] == participant]['time'].values.astype(float)

    baseline_cor_start_time, baseline_cor_end_time = pupil_analysis.extract_cluster_times(dataframes['cluster_permutation_analysis'], pupil_data['time'])
    nobc_start_time, nobc_end_time = pupil_analysis.extract_cluster_times(dataframes['cluster_permutation_analysis_nobc'], pupil_data_nobc['time'])

    pupil_data = pupil_data.set_index('time')
    pupil_data_nobc = pupil_data_nobc.set_index('time')

    epochs_removed = epochs_to_remove[epochs_to_remove['participant'] == participant]['epochs'].values
    comp_data.drop(index=epochs_removed, inplace=True)

    filtered_pupil_data_bc = pupil_data.loc[(pupil_data.index >= baseline_cor_start_time) & (pupil_data.index <= baseline_cor_end_time)]
    filtered_pupil_data_nobc = pupil_data_nobc.loc[(pupil_data_nobc.index >= nobc_start_time) & (pupil_data_nobc.index <= 0)]

    pupil_variable_bc = np.array(filtered_pupil_data_bc.mean())
    pupil_variable_nobc = np.array(filtered_pupil_data_nobc.mean())

    df = pd.DataFrame({
        'lr': comp_data['cleaned_learning_rate_estimate'].values,
        'pupil_bc': pupil_variable_bc,
        'pupil_nobc': pupil_variable_nobc,
        'participant': participant
    })

    bin_df.append(df)

group_bin_df = pd.concat(bin_df, ignore_index=True)

# Binning pupil_bc
nbins = 6
group_bin_df['pupil_bc_bin'], bin_edges_pupil_bc_lr = pd.qcut(group_bin_df["pupil_bc"], nbins, labels=False, retbins=True)

# Calculate mean pupil_bc per bin (labels)
bin_labels = []
for bin in sorted(group_bin_df['pupil_bc_bin'].unique()):
    data = group_bin_df[group_bin_df['pupil_bc_bin'] == bin]
    mean_pupil_bc = data['pupil_bc'].mean()
    bin_labels.append({'pupil_bc_bin': bin, 'pupil_bc_bin_label': mean_pupil_bc})

bin_labels_df = pd.DataFrame(bin_labels)

# Merge bin labels into main dataframe
group_bin_df = group_bin_df.merge(bin_labels_df, on='pupil_bc_bin')

# Storing the average learning rate at the participant level per bin
binned_data = {"pupil_bc_bin": [], "pupil_bc_bin_label": [], "avg_lr": [], "participant": []}

for participant in group_bin_df["participant"].unique():
    participant_data = group_bin_df[group_bin_df["participant"] == participant]

    for bin in sorted(participant_data["pupil_bc_bin"].unique()):
        bin_data = participant_data[participant_data["pupil_bc_bin"] == bin]

        avg_lr = bin_data["lr"].mean()
        bin_label = bin_data['pupil_bc_bin_label'].unique()[0]

        binned_data["pupil_bc_bin"].append(bin)
        binned_data["pupil_bc_bin_label"].append(bin_label)
        binned_data["avg_lr"].append(avg_lr)
        binned_data["participant"].append(participant)

participant_pupil_bc_lr_df = pd.DataFrame(binned_data)


group_avg_data = {"pupil_bc_bin": [], "pupil_bc_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

for bin in sorted(participant_pupil_bc_lr_df['pupil_bc_bin'].unique()):
    data = participant_pupil_bc_lr_df[participant_pupil_bc_lr_df['pupil_bc_bin'] == bin]
    learning_rate = data['avg_lr'].mean()
    lr_sem = stats.sem(data['avg_lr'])
    bin_label = data['pupil_bc_bin_label'].unique()[0] 

    group_avg_data["pupil_bc_bin"].append(bin)
    group_avg_data["pupil_bc_bin_label"].append(bin_label)
    group_avg_data["group_avg_lr"].append(learning_rate)
    group_avg_data["group_sem_lr"].append(lr_sem)

behav_group_pupil_bc_lr_df = pd.DataFrame(group_avg_data)

# Binning pupil_nobc
group_bin_df['pupil_nobc_bin'], bin_edges_pupil_bc_lr = pd.qcut(group_bin_df["pupil_nobc"], nbins, labels=False, retbins=True)

# Calculate mean pupil_bc per bin (labels)
bin_labels_nobc = []
for bin in sorted(group_bin_df['pupil_nobc_bin'].unique()):
    data = group_bin_df[group_bin_df['pupil_nobc_bin'] == bin]
    mean_pupil_nobcbc = data['pupil_nobc'].mean()
    bin_labels_nobc.append({'pupil_nobc_bin': bin, 'pupil_nobc_bin_label': mean_pupil_nobcbc})

bin_labels_nobc_df = pd.DataFrame(bin_labels_nobc)

# Merge bin labels into main dataframe
group_bin_df = group_bin_df.merge(bin_labels_nobc_df, on='pupil_nobc_bin')

# Storing the average learning rate at the participant level per bin
binned_data_nobc = {"pupil_nobc_bin": [], "pupil_nobc_bin_label": [], "avg_lr": [], "participant": []}

for participant in group_bin_df["participant"].unique():
    participant_data = group_bin_df[group_bin_df["participant"] == participant]

    for bin in sorted(participant_data["pupil_nobc_bin"].unique()):
        bin_data = participant_data[participant_data["pupil_nobc_bin"] == bin]

        avg_lr = bin_data["lr"].mean()
        bin_label = bin_data['pupil_nobc_bin_label'].unique()[0]

        binned_data_nobc["pupil_nobc_bin"].append(bin)
        binned_data_nobc["pupil_nobc_bin_label"].append(bin_label)
        binned_data_nobc["avg_lr"].append(avg_lr)
        binned_data_nobc["participant"].append(participant)

participant_pupil_nobc_lr_df = pd.DataFrame(binned_data_nobc)

group_avg_data_nobc = {"pupil_nobc_bin": [], "pupil_nobc_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

for bin in sorted(participant_pupil_nobc_lr_df['pupil_nobc_bin'].unique()):
    data = participant_pupil_nobc_lr_df[participant_pupil_nobc_lr_df['pupil_nobc_bin'] == bin]
    learning_rate = data['avg_lr'].mean()
    lr_sem = stats.sem(data['avg_lr'])
    bin_label = data['pupil_nobc_bin_label'].unique()[0] 

    group_avg_data_nobc["pupil_nobc_bin"].append(bin)
    group_avg_data_nobc["pupil_nobc_bin_label"].append(bin_label)
    group_avg_data_nobc["group_avg_lr"].append(learning_rate)
    group_avg_data_nobc["group_sem_lr"].append(lr_sem)

behav_group_pupil_nobc_lr_df = pd.DataFrame(group_avg_data_nobc)

# PLot figure with inset bar plot and jittered scatter points
fig = make_subplots(
    rows=1, cols=1,
    specs=[[{"type": "xy"}]]
)

# ----- Main Line Plot -----
fig.add_trace(go.Scatter(
    x=behav_group_pupil_bc_lr_df['pupil_bc_bin_label'],
    y=behav_group_pupil_bc_lr_df['group_avg_lr'], 
    mode='markers', 
    name='Phasic', 
    marker=dict(color='rgba(128, 0, 128, 0.8)', size=19),
    error_y=dict(
        type='data',
        array=behav_group_pupil_bc_lr_df['group_sem_lr'],
        visible=True
    )
))

fig.add_trace(go.Scatter(
    x=behav_group_pupil_nobc_lr_df['pupil_nobc_bin_label'],
    y=behav_group_pupil_nobc_lr_df['group_avg_lr'], 
    mode='markers', 
    name='Tonic', 
    marker=dict(color='rgba(255, 165, 0, 0.8)', size=19),
    error_y=dict(
        type='data',
        array=behav_group_pupil_nobc_lr_df['group_sem_lr'],
        visible=True
    )
))

# ----- Add Inset (Bar Plot) -----
# Create a secondary (inset) axis
np.random.seed(42)

# Define jitter width
jitter_width = 0.08

# Generate jittered x-values for scatter points
x_jittered_bc = np.random.normal(loc=0, scale=jitter_width, size=len(pupil_bc_coef))
x_jittered_nobc = np.random.normal(loc=1, scale=jitter_width, size=len(pupil_nobc_coef))

fig.add_trace(go.Bar(
    name='Phasic pupil signal',
    x=[0], y=[pupil_v_lr_bc_mean],
    error_y=dict(type='data', array=[pupil_v_lr_bc_sem]),
    marker=dict(color='rgba(128, 0, 128, 0.5)'),
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Bar 2
fig.add_trace(go.Bar(
    name='Tonic pupil signal',
    x=[1], y=[pupil_v_lr_nobc_mean],
    error_y=dict(type='data', array=[pupil_v_lr_nobc_sem]),
    marker=dict(color='rgba(255, 165, 0, 0.5)'),
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Jittered scatter 1
fig.add_trace(go.Scatter(
    x=x_jittered_bc,
    y=pupil_bc_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(128, 0, 128, 0.3)', size=12),
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Jittered scatter 2
fig.add_trace(go.Scatter(
    x=x_jittered_nobc,
    y=pupil_nobc_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(255, 165, 0, 0.5)', size=12),
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Add a horizontal line at y=0
fig.add_shape(
    type='line',
    x0=-0.5, x1=1.5,  
    y0=0, y1=0,  
    xref='x2',   
    yref='y2',   
    line=dict(
        color='black',
        width=1.5,
        dash='dot'
    )
)

x_bc = 0
x_nobc = 1

y_offset = 0.2  # Adjust this as needed to space the asterisks above the error bars

# Bar tops: mean + error
bc_bar_top = pupil_v_lr_bc_mean + pupil_v_lr_bc_sem
nobc_bar_top = pupil_v_lr_nobc_mean + pupil_v_lr_nobc_sem

# Star positions
pupil_bc_star_y = bc_bar_top + y_offset
pupil_nobc_star_y = nobc_bar_top + y_offset

# For between-group comparison, place the line a little higher
between_bar_top = max(bc_bar_top, nobc_bar_top)
between_star_y = between_bar_top + (y_offset * 2)

pval_bc = dataframes['pupil_v_lr_coef_ttest_df'].loc[
    dataframes['pupil_v_lr_coef_ttest_df']['variable'] == 'pupil_bc', 'p_value'].values[0]
pval_nobc = dataframes['pupil_v_lr_coef_ttest_df'].loc[
    dataframes['pupil_v_lr_coef_ttest_df']['variable'] == 'pupil_nobc', 'p_value'].values[0]
pval_between = dataframes['pupil_v_lr_coef_ttest_df'].loc[
    dataframes['pupil_v_lr_coef_ttest_df']['variable'] == 'between_pupil_coef', 'p_value'].values[0]

# Add asterisk for baseline-corrected if significant
if pval_bc < 0.05:
    fig.add_annotation(
        text='*',
        x=x_bc, y=pupil_bc_star_y,
        showarrow=False,
        font=dict(size=30),
        xref='x2',  # Target inset x-axis
        yref='y2'   # Target inset y-axis
    )

# Add asterisk for non-baseline corrected if significant
if pval_nobc < 0.05:
    fig.add_annotation(
        text='*',
        x=x_nobc, y=pupil_nobc_star_y,
        showarrow=False,
        font=dict(size=30),
        xref='x2',
        yref='y2'
    )

# Add between-group comparison line and asterisk if significant
if pval_between < 0.05:
    fig.add_shape(type='line',
                  x0=0, x1=1,
                  y0=between_star_y, y1=between_star_y,
                  xref='x2',
                  yref='y2',
                  line=dict(color='black', width=1))

    fig.add_annotation(
        text='*',
        x=0.5, y=between_star_y + 0.02,
        showarrow=False,
        font=dict(size=30),
        xref='x2',
        yref='y2'
    )


# ----- Inset Positioning -----
# You can control the inset size and position using xaxis2 and yaxis2 domains
ymin = min(min(pupil_bc_coef['coefficient']), min(pupil_nobc_coef['coefficient'])) - 0.1
ymax = max(bc_bar_top, nobc_bar_top) + 0.2

fig.update_layout(
    xaxis=dict(
        showline=True, linewidth=1.5, linecolor='black', title_text="Pupil signal (z-score)"
    ),
    yaxis=dict(
        showline=True, linewidth=1.5, linecolor='black', title_text="Apparent learning rate (participant)"
    ),
    xaxis2=dict(
        domain=[0.12, 0.42],  # Position inset on the right side
        anchor='y2',
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['Phasic', 'Tonic'],
        showline=True,        
        linewidth=1.5,          
        linecolor='black',    
        tickfont=dict(size=20),  
        titlefont=dict(size=20),
        tickangle=0
    ),
    yaxis2=dict(
    domain=[0.70, 1.0],
    anchor='x2',
    title_text="Regression weight",
    showline=True,        
    linewidth=1.5,          
    linecolor='black',    
    tickfont=dict(size=20),  
    titlefont=dict(size=20),
    range=[ymin, ymax]  # Force the y-axis range
    ),
    font=dict(size=24, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=720,
    width=840,
    legend=dict(
        x=0.70,
        y=0.20,
        xanchor='left',
        yanchor='top'
    )
)

# Link the bar plot to xaxis2 and yaxis2
for i in range(3, len(fig.data)):
    fig.data[i].update(xaxis='x2', yaxis='y2')


# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_pupil_signal_bins_learning_rate"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

############################################################################################### Mediation analysis (nonbaseline-corrected)
mediation_group = dataframes['pupil_lr_uncert_nobc_mediation']


def format_annotation_text(title, coef, sem, ci_lower, ci_upper):
    ci_same_sign = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    text = f"{title}"

    if ci_same_sign:
        return f"<b>{text}</b>"
    else:
        return text

def add_circle(center_x, center_y, radius, label):
    fig.add_shape(
        type="circle",
        x0=center_x - radius,
        y0=center_y - radius,
        x1=center_x + radius,
        y1=center_y + radius,
        fillcolor="PaleTurquoise",
        line_color="LightSeaGreen",
    )
    fig.add_annotation(
        x=center_x,
        y=center_y,
        text=label,
        showarrow=False
    )


fig = go.Figure()
fig.update_xaxes(range=[-0.5, 10.5], showticklabels=False)
fig.update_yaxes(range=[0, 8.5], showticklabels=False, scaleanchor="x", scaleratio=1)

# Adding in the mediation pathways

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2, x1=5,
    y1=6,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=6, x1=9,
    y1=2,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2.10, x1=9,
    y1=2.10,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=1.9, x1=9,
    y1=1.9,
    line=dict(
        color="Black",
        width=3,
    ),
)

add_circle(1, 2, 1.4, "Prior<br>uncertainty")
add_circle(9, 2, 1.4, "Apparent<br>learning<br>rate")
add_circle(5, 6, 1.4, "Pupil:<br>non-corrected<br>(tonic)")

total_path_text = format_annotation_text(
    "Total effect (path c)",
    mediation_group['boot_mean'][2],
    mediation_group['sem'][2],
    mediation_group['ci_lower'][2],
    mediation_group['ci_upper'][2]
)

fig.add_annotation(
    x=5,  
    y=0.80,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = format_annotation_text(
    "Direct effect (path c')",
    mediation_group['boot_mean'][3],
    mediation_group['sem'][3],
    mediation_group['ci_lower'][3],
    mediation_group['ci_upper'][3]
)

fig.add_annotation(
    x=5,  
    y=3.1,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


pupil_corrected_path_text = format_annotation_text(
    "Indirect effect (path a*b)",
    mediation_group['boot_mean'][4],
    mediation_group['sem'][4],
    mediation_group['ci_lower'][4],
    mediation_group['ci_upper'][4]
)


fig.add_annotation(
    x=5,  
    y=8.2,
    xref="x",
    yref="y",
    text=pupil_corrected_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)



pupil_corrected_path_a_text = format_annotation_text(
    "Path a",
    mediation_group['boot_mean'][0],
    mediation_group['sem'][0],
    mediation_group['ci_lower'][0],
    mediation_group['ci_upper'][0]
)


fig.add_annotation(
    x=1.2,  
    y=5.1,
    xref="x",
    yref="y",
    text=pupil_corrected_path_a_text,
    font=dict(color="grey"),
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

pupil_corrected_path_b_text = format_annotation_text(
    "Path b",
    mediation_group['boot_mean'][1],
    mediation_group['sem'][1],
    mediation_group['ci_lower'][1],
    mediation_group['ci_upper'][1]
)


fig.add_annotation(
    x=8.2,  
    y=5.1,
    xref="x",
    yref="y",
    text=pupil_corrected_path_b_text,
    font=dict(color="grey"),
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


fig.update_layout(
    width=720,
    height=840,
    font=dict(size=24, family="Times New Roman"), 
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
) 

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_mediation_analysis_nobc"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()

#####################################################################Mediation analysis (baseline-corrected)
mediation_group = dataframes['pupil_lr_uncert_bc_mediation']

fig = go.Figure()
fig.update_xaxes(range=[-0.5, 10.5], showticklabels=False)
fig.update_yaxes(range=[0, 8.5], showticklabels=False, scaleanchor="x", scaleratio=1)

# Adding in the mediation pathways

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2, x1=5,
    y1=6,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=6, x1=9,
    y1=2,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2.05, x1=9,
    y1=2.05,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=1.95, x1=9,
    y1=1.95,
    line=dict(
        color="Black",
        width=3,
    ),
)

add_circle(1, 2, 1.4, "CPP")
add_circle(9, 2, 1.4, "Apparent<br>learning<br>rate")
add_circle(5, 6, 1.4, "Pupil:<br>non-corrected<br>(phasic)")
)

total_path_text = format_annotation_text(
    "Total effect (path c)",
    mediation_group['boot_mean'][2],
    mediation_group['sem'][2],
    mediation_group['ci_lower'][2],
    mediation_group['ci_upper'][2]
)

fig.add_annotation(
    x=5,  
    y=0.75,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = format_annotation_text(
    "Direct effect (path c')",
    mediation_group['boot_mean'][3],
    mediation_group['sem'][3],
    mediation_group['ci_lower'][3],
    mediation_group['ci_upper'][3]
)

fig.add_annotation(
    x=5,  
    y=3.3,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


pupil_corrected_path_text = format_annotation_text(
    "Indirect effect <br>(Pupil: corrected)",
    mediation_group['boot_mean'][4],
    mediation_group['sem'][4],
    mediation_group['ci_lower'][4],
    mediation_group['ci_upper'][4]
)


fig.add_annotation(
    x=5,  
    y=8.2,
    xref="x",
    yref="y",
    text=pupil_corrected_path_text,
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)



pupil_corrected_path_a_text = format_annotation_text(
    "Path a",
    mediation_group['boot_mean'][0],
    mediation_group['sem'][0],
    mediation_group['ci_lower'][0],
    mediation_group['ci_upper'][0]
)


fig.add_annotation(
    x=1.5,  
    y=5.5,
    xref="x",
    yref="y",
    text=pupil_corrected_path_a_text,
    font=dict(color="grey"),
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

pupil_corrected_path_b_text = format_annotation_text(
    "Path b",
    mediation_group['boot_mean'][1],
    mediation_group['sem'][1],
    mediation_group['ci_lower'][1],
    mediation_group['ci_upper'][1]
)


fig.add_annotation(
    x=8.5,  
    y=5.5,
    xref="x",
    yref="y",
    text=pupil_corrected_path_b_text,
    font=dict(color="grey"),
    showarrow=False,
    bordercolor="white",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


fig.update_layout(
    width=720,
    height=840,
    font=dict(size=24, family="Times New Roman"), 
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
) 

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_mediation_analysis_bc"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png", scale=3.125)

# Save as SVG
fig.write_image(f"{base_filepath}.svg", scale=3.125) 

fig.show()