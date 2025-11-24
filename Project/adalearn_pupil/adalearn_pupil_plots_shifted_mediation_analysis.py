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
    'shifted_lr_uncert_nobc_mediation',
    'shifted_lr_uncert_bc_mediation'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

############################################################################################### Mediation analysis (nonbaseline-corrected)
mediation_group = dataframes['shifted_lr_uncert_nobc_mediation']


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
    label=dict(text=f"Prior <br>uncertainty", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.0, y0=1.0, x1=10.0, y1=3.0,
    line_color="LightSeaGreen",
    label=dict(text="Shifted <br>learning <br>rate", textangle=0)
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
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_shifted_mediation_analysis_nobc"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()

#####################################################################Mediation analysis (baseline-corrected)
mediation_group = dataframes['shifted_lr_uncert_bc_mediation']


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
    label=dict(text=f"CPP", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.0, y0=1.0, x1=10.0, y1=3.0,
    line_color="LightSeaGreen",
    label=dict(text="Shifted <br>learning <br>rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.0, y0=5.0, x1=6.0, y1=7.0,
    line_color="LightSeaGreen",
    label=dict(text="Pupil:<br>corrected<br>(phasic)", textangle=0)
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
    width=2200,
    height=2200,
    font=dict(size=42, family="Times New Roman"), 
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
) 

# Save the figure to the group-level folder
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Base filename (without extension)
base_filename = f"adalearn_pupil-group_level_shifted_mediation_analysis_bc"
base_filepath = os.path.join(plot_dir, base_filename)

# Save as PNG
fig.write_image(f"{base_filepath}.png")

# Save as SVG
fig.write_image(f"{base_filepath}.svg") 

fig.show()