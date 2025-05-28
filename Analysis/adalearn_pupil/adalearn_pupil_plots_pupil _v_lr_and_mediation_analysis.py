"""

"""
import pandas as pd
import numpy as np
from scipy import stats
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_group_level_functions_data_utilities import load_sample_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, base_dir, condition_str


dataframes = load_sample_csvs(directory_path=base_dir, substrings=['change_point_indices_df', 'pupil_v_lr_regress_coef_df', 'pupil_v_lr_coef_ttest_df', 'pupil_bc_around_cp', 'pupil_nobc_around_cp'])

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
    x=grouped_pupil_bc_cp_rel_pos_df['rel_pos'], 
    y=grouped_pupil_bc_cp_rel_pos_df['pupil_bc_mean'], 
    mode='markers', 
    marker=dict(size=16, color='rgba(0, 130, 150, 0.8)'),
    error_y=dict(type='data', array=grouped_pupil_bc_cp_rel_pos_df['pupil_bc_sem'], visible=True)
), row=1, col=1)

# Pupil nonbaseline-corrected effect window
fig.add_trace(go.Scatter(
    x=grouped_pupil_nobc_cp_rel_pos_df['rel_pos'], 
    y=grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_mean'], 
    mode='markers', 
    marker=dict(size=16, color='rgba(0, 130, 150, 0.8)'),
    error_y=dict(type='data', array=grouped_pupil_nobc_cp_rel_pos_df['pupil_nobc_sem'], visible=True)
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


fig.show()




#########################################################Plotting pupil effects on learning rate 

   
pupil_v_lr_regress_df = dataframes['pupil_v_lr_regress_coef_df']
pupil_v_lr_ttest_df = dataframes['pupil_v_lr_coef_ttest_df']

pupil_bc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_response_bc_zscore'].sort_values('participant')
pupil_nobc_coef = pupil_v_lr_regress_df[pupil_v_lr_regress_df['variable'] == 'pupil_response_nobc_zscore'].sort_values('participant')


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
        font=dict(size=24)
    )

if pupil_v_lr_ttest_df.loc[
    pupil_v_lr_ttest_df['variable'] == 'pupil_nobc', 'p_value'
].values[0] < 0.05:
    fig.add_annotation(
        text='*',
        x=x_nobc, y=pupil_nobc_star_y,
        showarrow=False,
        font=dict(size=24)
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
        font=dict(size=24)
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

fig.show()