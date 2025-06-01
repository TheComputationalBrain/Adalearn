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

from adalearn_analysis_group_level_functions_data_utilities import load_sample_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, base_dir, condition_str


dataframes = load_sample_csvs(directory_path=base_dir, substrings=['change_point_indices_df', 'pupil_v_lr_regress_coef_df', 'pupil_v_lr_coef_ttest_df', 'pupil_bc_around_cp', 'pupil_nobc_around_cp', 'pupil_lr_uncert_mediation'])

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
    ticktext=["", "", "", "", "", "Change point", "", "", "", "", ""],
    row=i, col=1
)

# Update y-axes
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Pupil diameter signal (z-score)', row=1, col=1)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', 
                 title_text='Pupil diameter signal (z-score)', row=2, col=1)


# Update layout
fig.update_layout(height=600*2, width=1200, plot_bgcolor='rgba(255, 255, 255, 1)', font=dict(size=20), showlegend=False)

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




#########################################################Plotting pupil effects on learning rate 

   
pupil_v_lr_regress_df = dataframes['pupil_v_lr_regress_coef_df']
pupil_v_lr_ttest_df = dataframes['pupil_v_lr_coef_ttest_df']

pupil_bc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_bc_zscore'].sort_values('participant')
pupil_nobc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_nobc_zscore'].sort_values('participant')


pupil_v_lr_bc_mean = pupil_bc_coef['coefficient'].mean()
pupil_v_lr_nobc_mean = pupil_nobc_coef['coefficient'].mean()
pupil_v_lr_bc_sem = stats.sem(pupil_bc_coef['coefficient'])
pupil_v_lr_nobc_sem = stats.sem(pupil_nobc_coef['coefficient'])


fig = go.Figure()

x_bc = 0
x_nobc = 1

# Add bars
fig.add_trace(go.Bar(
    name='Baseline corrected pupil signal',
    x=[x_bc], y=[pupil_v_lr_bc_mean],
    error_y=dict(type='data', array=[pupil_v_lr_bc_sem]),
    marker=dict(color='rgba(0, 130, 150, 0.4)'),
    showlegend=False
))

fig.add_trace(go.Bar(
    name='Non-baseline corrected pupil signal',
    x=[x_nobc], y=[pupil_v_lr_nobc_mean],
    error_y=dict(type='data', array=[pupil_v_lr_nobc_sem]),
    marker=dict(color='rgba(0, 130, 150, 0.4)'),
    showlegend=False
))

jitter_strength = 0.05

# Baseline corrected raw data 
x_jittered_bc = np.random.normal(loc=x_bc, scale=jitter_strength, size=len(pupil_bc_coef))
fig.add_trace(go.Scatter(
    x=x_jittered_bc,
    y=pupil_bc_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size = 12),
    showlegend=False
))

# Non-baseline corrected raw data
x_jittered_nobc = np.random.normal(loc=x_nobc, scale=jitter_strength, size=len(pupil_nobc_coef))
fig.add_trace(go.Scatter(
    x=x_jittered_nobc,
    y=pupil_nobc_coef['coefficient'],
    mode='markers',
    marker=dict(color='rgba(0, 130, 150, 0.3)', size = 12),
    showlegend=False
))

fig.update_traces(
    marker_line_color='rgba(0, 0, 0, 0.6)', 
    marker_line_width=1.0
)

fig.add_hline(y=0, line_dash='dot')

# Determine y positions for stars (slightly above error bar max)
y_offset = 0.4 # adjust as needed
pupil_bc_star_y = pupil_v_lr_bc_mean - pupil_v_lr_bc_sem + y_offset
pupil_nobc_star_y = pupil_v_lr_nobc_mean - pupil_v_lr_nobc_sem + y_offset
between_star_y = max(pupil_bc_star_y, pupil_nobc_star_y) + 0.1  # higher for between comparison line

# Add asterisks if significant
if pupil_v_lr_ttest_df.loc[
    pupil_v_lr_ttest_df['variable'] == 'pupil_bc', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=x_bc, y=pupil_bc_star_y,
        showarrow=False,
        font=dict(size=28)
    )

if pupil_v_lr_ttest_df.loc[
    pupil_v_lr_ttest_df['variable'] == 'pupil_nobc', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=x_nobc, y=pupil_nobc_star_y,
        showarrow=False,
        font=dict(size=28)
    )

# Optional: Add a bar and star for between-group comparison
if pupil_v_lr_ttest_df.loc[
    pupil_v_lr_ttest_df['variable'] == 'between_pupil_coef', 'p_value'
].values[0] < 0.05:
    fig.add_shape(type='line',
                  x0=0, x1=1,
                  y0=between_star_y, y1=between_star_y,
                  line=dict(color='black', width=1))
    fig.add_annotation(
        text='*',
        x=0.5, y=between_star_y - 0.02,
        showarrow=False,
        font=dict(size=28)
    )

fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black',  title_text="Regression weight")

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[x_bc, x_nobc],
        ticktext=['Baseline corrected pupil signal (z-score)', 'Non-baseline corrected pupil signal (z-score)']
    ),
    yaxis_title="Regression weight",
    plot_bgcolor='white',
    font=dict(size=20),
    height=950,
    width=1800
)

# Save the figure to the group-level folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_pupil_v_lr_regression"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()


############################################################################################### Mediation anlysis
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
fig.update_xaxes(range=[0, 15], zeroline=False)
fig.update_yaxes(range=[0, 15])

# Adding the total effect separately

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=12, x1=9,
    y1=12,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.0, y0=11.0, x1=2.0, y1=13.0,
    line_color="LightSeaGreen",
    label=dict(text=f"Prior <br>confidence", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.0, y0=11.0, x1=10.0, y1=13.0,
    line_color="LightSeaGreen",
    label=dict(text="Learning <br>rate", textangle=0)
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
    y=13.5,
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

# Adding in the mediation pathways

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2, x1=5,
    y1=6,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=6, x1=9,
    y1=2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=2, x1=9,
    y1=2,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
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
    label=dict(text="Learning <br>Rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.0, y0=5.0, x1=6.0, y1=7.0,
    line_color="LightSeaGreen",
    label=dict(text="Pupil: <br>non-corrected (tonic)", textangle=0)
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
    y=3,
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
    "Path a <br>(Pupil: non-corrected)",
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
    "Path b <br>(Pupil: non-corrected)",
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
    width=1600,
    height=1600,
    font=dict(size=16), 
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