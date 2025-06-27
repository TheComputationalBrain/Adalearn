"""

"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as pupil_analysis

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, base_dir, condition_str


dataframes = group_data_utils.load_sample_csvs(directory_path=base_dir, substrings=[
'computational_var_df',
'change_point_indices_df',
'pupil_v_lr_regress_coef_df',
'pupil_v_lr_coef_ttest_df',
'pupil_bc_around_cp',
'pupil_nobc_around_cp',
'pupil_lr_uncert_mediation',
'exclude_participants_pupil_analysis_df', 
'outcome_epochs_descriptive_stats_df', 
'outcome_epoched_pupil_df',
'outcome_epochs_columns_df', 
'outcome_epochs_to_remove_df',
'outcome_epochs_descriptive_stats_nobc_df', 
'outcome_epoched_pupil_nobc_df', 
'cluster_permutation_analysis', 
'cluster_permutation_analysis_nobc',
'pupil_v_lr_coef_ttest_df' 
])


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



fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=
                    ['Average baseline corrected pupil signal across effect window',
                     'Average non-baseline corrected pupil signal across effect window'])

# Pupil baseline-corrected effect window
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
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=grouped_pupil_bc_cp_rel_pos_df['rel_pos'],
    y=grouped_pupil_bc_cp_rel_pos_df['pupil_bc_mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=3),
    name='Pupil (Baseline Corrected)'
), row=1, col=1)

# Pupil nonbaseline-corrected effect window
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
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=grouped_pupil_nobc_cp_rel_pos_df['rel_pos'],
    y=grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_mean'],
    mode='lines',
    line=dict(color='rgba(0, 130, 150, 0.8)', width=3),
    name='Pupil (Non-Baseline Corrected)'
), row=2, col=1)


# Add vertical and horizontal lines for all subplots
for i in range(1, 3):  # Rows 1 to 3
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, 
                  line=dict(dash="dot", color="black"), 
                  xref="x", yref="y domain", row=i, col=1)
    

# Update x-axes
for i in range(1, 3):
    fig.update_xaxes(
    title_text="Outcomes relative to change point",
    showline=True,
    linewidth=1,
    linecolor='black',
    tickmode='array',
    tickvals=list(range(-5, 6)),
    ticktext=["", "", "", "", "", "(change point)", "", "", "", "", ""],
    range=[-5.5, 6.5],
    row=i, col=1
)

# Update y-axes
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Pupil diameter signal (z-score)', row=1, col=1)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Pupil diameter signal (z-score)', row=2, col=1)


# Update layout
fig.update_layout(height=1200*2, width=1800, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=42, family="Times New Roman"), showlegend=False)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_pupil_responses_around_change_points"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 


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
epoch_descriptive_stats = dataframes['outcome_epochs_descriptive_stats_df']
analysed_epochs = dataframes['outcome_epochs_columns_df']
epochs_to_remove = dataframes['outcome_epochs_to_remove_df']

bin_df = []

for participant in epoch_data['participant'].unique():
    comp_data = comp_df[comp_df['participant'] == participant].reset_index()

    pupil_data = epoch_data[epoch_data['participant'] == participant].drop(columns=['participant']).T
    column_titles = analysed_epochs[analysed_epochs['participant'] == participant]['columns'].values.astype(int)
    pupil_data.columns = column_titles
    pupil_data['time'] = epoch_descriptive_stats[epoch_descriptive_stats['participant'] == participant]['time'].values.astype(float)

    pupil_data_nobc = epoch_data_nobc[epoch_data_nobc['participant'] == participant].drop(columns=['participant']).T
    pupil_data_nobc.columns = column_titles
    pupil_data_nobc['time'] = epoch_descriptive_stats[epoch_descriptive_stats['participant'] == participant]['time'].values.astype(float)

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
    mode='lines+markers', 
    name='Baseline corrected', 
    line=dict(color='rgba(128, 0, 128, 0.8)', width=2),
    marker=dict(size=19),
    error_y=dict(
        type='data',
        array=behav_group_pupil_bc_lr_df['group_sem_lr'],
        visible=True
    )
))

fig.add_trace(go.Scatter(
    x=behav_group_pupil_nobc_lr_df['pupil_nobc_bin_label'],
    y=behav_group_pupil_nobc_lr_df['group_avg_lr'], 
    mode='lines+markers', 
    name='Non-baseline corrected', 
    line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
    marker=dict(size=19),
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
    name='Baseline corrected pupil signal',
    x=[0], y=[pupil_v_lr_bc_mean],
    error_y=dict(type='data', array=[pupil_v_lr_bc_sem]),
    marker=dict(color='rgba(128, 0, 128, 0.5)'),
    showlegend=False,
    xaxis='x2',
    yaxis='y2'
))

# Bar 2
fig.add_trace(go.Bar(
    name='Non-baseline corrected pupil signal',
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
        font=dict(size=60),
        xref='x2',  # Target inset x-axis
        yref='y2'   # Target inset y-axis
    )

# Add asterisk for non-baseline corrected if significant
if pval_nobc < 0.05:
    fig.add_annotation(
        text='*',
        x=x_nobc, y=pupil_nobc_star_y,
        showarrow=False,
        font=dict(size=60),
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
        font=dict(size=60),
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
        showline=True, linewidth=1.5, linecolor='black', title_text="Learning rate (participant)"
    ),
    xaxis2=dict(
        domain=[0.1, 0.4],  # Position inset on the right side
        anchor='y2',
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['Baseline corrected', 'Non-baseline corrected'],
        showline=True,        
        linewidth=1.5,          
        linecolor='black',    
        tickfont=dict(size=24),  
        titlefont=dict(size=32),
        tickangle=0
    ),
    yaxis2=dict(
    domain=[0.70, 1.0],
    anchor='x2',
    title_text="Regression weight",
    showline=True,        
    linewidth=1.5,          
    linecolor='black',    
    tickfont=dict(size=28),  
    titlefont=dict(size=32),
    range=[ymin, ymax]  # Force the y-axis range
    ),
    font=dict(size=42, family="Times New Roman"),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=1200,
    width=1800,
    legend=dict(
        x=0.70,
        y=0.15,
        xanchor='left',
        yanchor='top'
    )
)

# Link the bar plot to xaxis2 and yaxis2
for i in range(3, len(fig.data)):
    fig.data[i].update(xaxis='x2', yaxis='y2')


# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_pupil_signal_bins_learning_rate"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()

############################################################################################### Mediation analysis
mediation_group = dataframes['pupil_lr_uncert_mediation']


def format_annotation_text(title, coef, sem, ci_lower, ci_upper):
    ci_same_sign = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
    text = (
        f"{title}"
        f"<br>Coef = {coef:.3f}"
        f"<br>SEM = {sem:.3f}"
        f"<br>95% CI = {ci_lower:.3f} - {ci_upper:.3f}"
    )
    return f"<b>{text}</b>" if ci_same_sign else text


fig = go.Figure()
fig.update_xaxes(range=[-5, 12])
fig.update_yaxes(range=[-5, 12])

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

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.0, y0=1.0, x1=2.0, y1=3.0,
    line_color="LightSeaGreen",
    label=dict(text=f"Prior <br>confidence", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.0, y0=1.0, x1=10.0, y1=3.0,
    line_color="LightSeaGreen",
    label=dict(text="Learning <br>rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.0, y0=5.0, x1=6.0, y1=7.0,
    line_color="LightSeaGreen",
    label=dict(text="Pupil:<br>non-corrected<br>(tonic)", textangle=0)
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
    "Indirect effect <br>(Pupil: non-corrected)",
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
    width=2200,
    height=2200,
    font=dict(size=42, family="Times New Roman"), 
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
) 

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_mediation_analysis"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()