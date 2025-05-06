from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import os
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'


from adalearn_pupil.adalearn_pupil_participant_characteristics import base_dir

excluded_participants = ['17', '33']

condition_str = '01'




#############################################################################################Mediation analysis



pupil_mediation_amp_df = pd.DataFrame()
pupil_mediation_avg_df = pd.DataFrame()
pupil_mediation_amp_zero_excluded_df = pd.DataFrame()
pupil_mediation_avg_zero_excluded_df = pd.DataFrame()



base_path = Path(base_dir)

for sub_folder in base_path.iterdir():
    folder_parts = sub_folder.name.split('-')
    
    if len(folder_parts) > 1:
        participant_num = folder_parts[1]
    else:
        print(f"Skipping folder {sub_folder.name} due to unexpected naming format.")
        continue

    if sub_folder.is_dir() and participant_num not in excluded_participants:
        print(f"Processing folder: {sub_folder.name}")  
        
        # Construct the path to the csv_files folder
        csv_files_path = sub_folder / f'ses-{condition_str}' / 'csv_files'
        
        if csv_files_path.exists():
            print(f"CSV files path exists: {csv_files_path}") 
            
            participant_id = participant_num  # Since we already extracted it safely

            csv_file_name_amp = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_mediation-analysis-pupil-amplitude.csv'
            
            csv_file_path_amp = csv_files_path / csv_file_name_amp
            
            if csv_file_path_amp.exists():
                print(f"CSV file path exists: {csv_file_path_amp}")  
                
                # Read the CSV data
                csv_med_amp_df = pd.read_csv(csv_file_path_amp)
                print(f'The shape of the dataframe for participant {participant_id} is {csv_med_amp_df.shape}')

                csv_med_amp_df['participant_id'] = participant_id  
               
                # Concatenate to the main DataFrame
                pupil_mediation_amp_df = pd.concat([pupil_mediation_amp_df, csv_med_amp_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path_amp}")  

            
            csv_file_name_avg = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_mediation-analysis-pupil-average.csv'
            
            csv_file_path_avg = csv_files_path / csv_file_name_avg
            
            if csv_file_path_avg.exists():
                print(f"CSV file path exists: {csv_file_path_avg}")  
                
                # Read the CSV data
                csv_med_avg_df = pd.read_csv(csv_file_path_avg)
                print(f'The shape of the dataframe for participant {participant_id} is {csv_med_avg_df.shape}')

                csv_med_avg_df['participant_id'] = participant_id  
               
                # Concatenate to the main DataFrame
                pupil_mediation_avg_df = pd.concat([pupil_mediation_avg_df, csv_med_avg_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path_avg}")

            
            
            csv_file_name_amp_zero_exclude = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_mediation-analysis-pupil-amplitude-zero-excluded.csv'
            
            csv_file_path_amp_zero_exclude = csv_files_path / csv_file_name_amp_zero_exclude
            
            if csv_file_path_amp_zero_exclude.exists():
                print(f"CSV file path exists: {csv_file_path_amp_zero_exclude}")  
                
                # Read the CSV data
                csv_med_amp_zero_exclude_df = pd.read_csv(csv_file_path_amp_zero_exclude)
                print(f'The shape of the dataframe for participant {participant_id} is {csv_med_amp_zero_exclude_df.shape}')

                csv_med_amp_zero_exclude_df['participant_id'] = participant_id
               
                # Concatenate to the main DataFrame
                pupil_mediation_amp_zero_excluded_df = pd.concat([pupil_mediation_amp_zero_excluded_df, csv_med_amp_zero_exclude_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path_amp_zero_exclude}")   


            
            csv_file_name_avg_zero_exclude = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_mediation-analysis-pupil-average-zero-excluded.csv'
            
            csv_file_path_avg_zero_exclude = csv_files_path / csv_file_name_avg_zero_exclude
            
            if csv_file_path_avg_zero_exclude.exists():
                print(f"CSV file path exists: {csv_file_path_avg_zero_exclude}")  
                
                # Read the CSV data
                csv_med_avg_zero_exclude_df = pd.read_csv(csv_file_path_avg_zero_exclude)
                print(f'The shape of the dataframe for participant {participant_id} is {csv_med_avg_zero_exclude_df.shape}')

                csv_med_avg_zero_exclude_df['participant_id'] = participant_id  
               
                # Concatenate to the main DataFrame
                pupil_mediation_avg_zero_excluded_df = pd.concat([pupil_mediation_avg_zero_excluded_df, csv_med_avg_zero_exclude_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path_avg_zero_exclude}")   
        else:
            print(f"CSV folder not found: {csv_files_path}")  



mediation_amp_df = pd.DataFrame()

n_boot = 5000  # You can increase for more precision

for path in pupil_mediation_amp_df['path'].unique():
    data = pupil_mediation_amp_df[pupil_mediation_amp_df['path'] == path]
    coefs = data['coef'].values
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    mean = np.mean(boot_means)
    sem = np.std(boot_means, ddof=1)

    df_amp = pd.DataFrame({
        'path': [path_label],
        'mean': [mean],
        'sem': [sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    mediation_amp_df = pd.concat([mediation_amp_df, df_amp], ignore_index=True)


fig = go.Figure()
fig.update_xaxes(range=[0, 10], zeroline=False)
fig.update_yaxes(range=[0, 10])

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=8,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=8, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=2, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5.2, x1=9,
    y1=5.2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=4.8, x1=9,
    y1=4.8,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.5, y0=4.5, x1=1.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text=f"Pupil: <br>amp", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.5, y0=4.5, x1=9.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text="Learning Rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=7.5, x1=5.5, y1=8.5,
    line_color="LightSeaGreen",
    label=dict(text="Prior Conf", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=1.5, x1=5.5, y1=2.5,
    line_color="LightSeaGreen",
    label=dict(text="CPP", textangle=0)
)

total_path_text = (
                f"Total Effect (C)"
                f"<br>Coef = {mediation_amp_df['mean'][4]:.4f}"
                f"<br>SEM = {mediation_amp_df['sem'][4]:.4f}"
                f"<br>95% CI = {mediation_amp_df['ci_lower'][4]:.4f} - {mediation_amp_df['ci_upper'][4]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=4,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = (
                f"Direct Effect (C')"
                f"<br>Coef = {mediation_amp_df['mean'][5]:.4f}"
                f"<br>SEM = {mediation_amp_df['sem'][5]:.4f}"
                f"<br>95% CI = {mediation_amp_df['ci_lower'][5]:.4f} - {mediation_amp_df['ci_upper'][5]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=6,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


conf_path_text = (
                f"Indirect Effect (Conf)"
                f"<br>Coef = {mediation_amp_df['mean'][6]:.4f}"
                f"<br>SEM = {mediation_amp_df['sem'][6]:.4f}"
                f"<br>95% CI = {mediation_amp_df['ci_lower'][6]:.4f} - {mediation_amp_df['ci_upper'][6]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=9.2,
    xref="x",
    yref="y",
    text=conf_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

cpp_path_text = (
                f"Indirect Effect (CPP)"
                f"<br>Coef = {mediation_amp_df['mean'][7]:.4f}"
                f"<br>SEM = {mediation_amp_df['sem'][7]:.4f}"
                f"<br>95% CI = {mediation_amp_df['ci_lower'][7]:.4f} - {mediation_amp_df['ci_upper'][7]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=0.8,
    xref="x",
    yref="y",
    text=cpp_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

fig.update_layout(
    title=f"Mediation analysis examining pupil amplitude response on behavioural learning rate <br> mediated by uncertainty parameters",
    width=1000, 
    height=1000,
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
)  

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"mediation-amplitude-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path) 

fig.show()
    

    
mediation_avg_df = pd.DataFrame()

n_boot = 5000  # You can increase for more precision

for path in pupil_mediation_avg_df['path'].unique():
    data = pupil_mediation_avg_df[pupil_mediation_avg_df['path'] == path]
    coefs = data['coef'].values
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    mean = np.mean(boot_means)
    sem = np.std(boot_means, ddof=1)

    df_avg = pd.DataFrame({
        'path': [path_label],
        'mean': [mean],
        'sem': [sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    mediation_avg_df = pd.concat([mediation_avg_df, df_avg], ignore_index=True)


fig = go.Figure()
fig.update_xaxes(range=[0, 10], zeroline=False)
fig.update_yaxes(range=[0, 10])

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=8,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=8, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=2, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5.2, x1=9,
    y1=5.2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=4.8, x1=9,
    y1=4.8,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.5, y0=4.5, x1=1.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text=f"Pupil: <br>avg", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.5, y0=4.5, x1=9.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text="Learning Rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=7.5, x1=5.5, y1=8.5,
    line_color="LightSeaGreen",
    label=dict(text="Prior Conf", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=1.5, x1=5.5, y1=2.5,
    line_color="LightSeaGreen",
    label=dict(text="CPP", textangle=0)
)

total_path_text = (
                f"Total Effect (C)"
                f"<br>Coef = {mediation_avg_df['mean'][4]:.4f}"
                f"<br>SEM = {mediation_avg_df['sem'][4]:.4f}"
                f"<br>95% CI = {mediation_avg_df['ci_lower'][4]:.4f} - {mediation_avg_df['ci_upper'][4]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=4,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = (
                f"Direct Effect (C')"
                f"<br>Coef = {mediation_avg_df['mean'][5]:.4f}"
                f"<br>SEM = {mediation_avg_df['sem'][5]:.4f}"
                f"<br>95% CI = {mediation_avg_df['ci_lower'][5]:.4f} - {mediation_avg_df['ci_upper'][5]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=6,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


conf_path_text = (
                f"Indirect Effect (Conf)"
                f"<br>Coef = {mediation_avg_df['mean'][6]:.4f}"
                f"<br>SEM = {mediation_avg_df['sem'][6]:.4f}"
                f"<br>95% CI = {mediation_avg_df['ci_lower'][6]:.4f} - {mediation_avg_df['ci_upper'][6]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=9.2,
    xref="x",
    yref="y",
    text=conf_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

cpp_path_text = (
                f"Indirect Effect (CPP)"
                f"<br>Coef = {mediation_avg_df['mean'][7]:.4f}"
                f"<br>SEM = {mediation_avg_df['sem'][7]:.4f}"
                f"<br>95% CI = {mediation_avg_df['ci_lower'][7]:.4f} - {mediation_avg_df['ci_upper'][7]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=0.8,
    xref="x",
    yref="y",
    text=cpp_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

fig.update_layout(
    title=f"Mediation analysis examining pupil average response on behavioural learning rate <br> mediated by uncertainty parameters",
    width=1000, 
    height=1000,
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
)  

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"mediation-average-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path) 

fig.show()
    
#########################################################Something is wrong with this zero_exclude analysis, check original function

mediation_amp_zero_excluded_df = pd.DataFrame()

n_boot = 5000  # You can increase for more precision

for path in pupil_mediation_amp_zero_excluded_df['path'].unique():
    data = pupil_mediation_amp_zero_excluded_df[pupil_mediation_amp_zero_excluded_df['path'] == path]
    coefs = data['coef'].values
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    mean = np.mean(boot_means)
    sem = np.std(boot_means, ddof=1)

    df_amp_zero_exclude = pd.DataFrame({
        'path': [path_label],
        'mean': [mean],
        'sem': [sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    mediation_amp_zero_excluded_df = pd.concat([mediation_amp_zero_excluded_df, df_amp_zero_exclude], ignore_index=True)



fig = go.Figure()
fig.update_xaxes(range=[0, 10], zeroline=False)
fig.update_yaxes(range=[0, 10])

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=8,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=8, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=2, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5.2, x1=9,
    y1=5.2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=4.8, x1=9,
    y1=4.8,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.5, y0=4.5, x1=1.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text=f"Pupil: <br>amp", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.5, y0=4.5, x1=9.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text="Learning Rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=7.5, x1=5.5, y1=8.5,
    line_color="LightSeaGreen",
    label=dict(text="Prior Conf", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=1.5, x1=5.5, y1=2.5,
    line_color="LightSeaGreen",
    label=dict(text="CPP", textangle=0)
)

total_path_text = (
                f"Total Effect (C)"
                f"<br>Coef = {mediation_amp_zero_excluded_df['mean'][4]:.4f}"
                f"<br>SEM = {mediation_amp_zero_excluded_df['sem'][4]:.4f}"
                f"<br>95% CI = {mediation_amp_zero_excluded_df['ci_lower'][4]:.4f} - {mediation_amp_zero_excluded_df['ci_upper'][4]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=4,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = (
                f"Direct Effect (C')"
                f"<br>Coef = {mediation_amp_zero_excluded_df['mean'][5]:.4f}"
                f"<br>SEM = {mediation_amp_zero_excluded_df['sem'][5]:.4f}"
                f"<br>95% CI = {mediation_amp_zero_excluded_df['ci_lower'][5]:.4f} - {mediation_amp_zero_excluded_df['ci_upper'][5]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=6,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


conf_path_text = (
                f"Indirect Effect (Conf)"
                f"<br>Coef = {mediation_amp_zero_excluded_df['mean'][6]:.4f}"
                f"<br>SEM = {mediation_amp_zero_excluded_df['sem'][6]:.4f}"
                f"<br>95% CI = {mediation_amp_zero_excluded_df['ci_lower'][6]:.4f} - {mediation_amp_zero_excluded_df['ci_upper'][6]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=9.2,
    xref="x",
    yref="y",
    text=conf_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

cpp_path_text = (
                f"Indirect Effect (CPP)"
                f"<br>Coef = {mediation_amp_zero_excluded_df['mean'][7]:.4f}"
                f"<br>SEM = {mediation_amp_zero_excluded_df['sem'][7]:.4f}"
                f"<br>95% CI = {mediation_amp_zero_excluded_df['ci_lower'][7]:.4f} - {mediation_amp_zero_excluded_df['ci_upper'][7]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=0.8,
    xref="x",
    yref="y",
    text=cpp_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

fig.update_layout(
    title=f"Mediation analysis examining pupil amplitude response on behavioural learning rate <br> mediated by uncertainty parameters (zero updates excluded)",
    width=1000, 
    height=1000,
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
)  

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"mediation-amplitude-zero-excluded-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path) 

fig.show()
    




mediation_avg_zero_excluded_df = pd.DataFrame()

n_boot = 5000  # You can increase for more precision

for path in pupil_mediation_avg_zero_excluded_df['path'].unique():
    data = pupil_mediation_avg_zero_excluded_df[pupil_mediation_avg_zero_excluded_df['path'] == path]
    coefs = data['coef'].values
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    mean = np.mean(boot_means)
    sem = np.std(boot_means, ddof=1)

    df_avg_zero_exclude = pd.DataFrame({
        'path': [path_label],
        'mean': [mean],
        'sem': [sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    mediation_avg_zero_excluded_df = pd.concat([mediation_avg_zero_excluded_df, df_avg_zero_exclude], ignore_index=True)


fig = go.Figure()
fig.update_xaxes(range=[0, 10], zeroline=False)
fig.update_yaxes(range=[0, 10])

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=8,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=8, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5, x1=5,
    y1=2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=5, y0=2, x1=9,
    y1=5,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=5.2, x1=9,
    y1=5.2,
    line=dict(
        color="Black",
        width=3,
    ),
)

fig.add_shape(type="line",
    xref="x", yref="y",
    x0=1, y0=4.8, x1=9,
    y1=4.8,
    line=dict(
        color="Black",
        width=3,
        dash='dash'
    ),
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=0.5, y0=4.5, x1=1.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text=f"Pupil: <br>avg", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=8.5, y0=4.5, x1=9.5, y1=5.5,
    line_color="LightSeaGreen",
    label=dict(text="Learning Rate", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=7.5, x1=5.5, y1=8.5,
    line_color="LightSeaGreen",
    label=dict(text="Prior Conf", textangle=0)
)

fig.add_shape(type="circle",
    xref="x", yref="y",
    fillcolor="PaleTurquoise",
    x0=4.5, y0=1.5, x1=5.5, y1=2.5,
    line_color="LightSeaGreen",
    label=dict(text="CPP", textangle=0)
)

total_path_text = (
                f"Total Effect (C)"
                f"<br>Coef = {mediation_avg_zero_excluded_df['mean'][4]:.4f}"
                f"<br>SEM = {mediation_avg_zero_excluded_df['sem'][4]:.4f}"
                f"<br>95% CI = {mediation_avg_zero_excluded_df['ci_lower'][4]:.4f} - {mediation_avg_zero_excluded_df['ci_upper'][4]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=4,
    xref="x",
    yref="y",
    text=total_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

direct_path_text = (
                f"Direct Effect (C')"
                f"<br>Coef = {mediation_avg_zero_excluded_df['mean'][5]:.4f}"
                f"<br>SEM = {mediation_avg_zero_excluded_df['sem'][5]:.4f}"
                f"<br>95% CI = {mediation_avg_zero_excluded_df['ci_lower'][5]:.4f} - {mediation_avg_zero_excluded_df['ci_upper'][5]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=6,
    xref="x",
    yref="y",
    text=direct_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)


conf_path_text = (
                f"Indirect Effect (Conf)"
                f"<br>Coef = {mediation_avg_zero_excluded_df['mean'][6]:.4f}"
                f"<br>SEM = {mediation_avg_zero_excluded_df['sem'][6]:.4f}"
                f"<br>95% CI = {mediation_avg_zero_excluded_df['ci_lower'][6]:.4f} - {mediation_avg_zero_excluded_df['ci_upper'][6]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=9.2,
    xref="x",
    yref="y",
    text=conf_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

cpp_path_text = (
                f"Indirect Effect (CPP)"
                f"<br>Coef = {mediation_avg_zero_excluded_df['mean'][7]:.4f}"
                f"<br>SEM = {mediation_avg_zero_excluded_df['sem'][7]:.4f}"
                f"<br>95% CI = {mediation_avg_zero_excluded_df['ci_lower'][7]:.4f} - {mediation_avg_zero_excluded_df['ci_upper'][7]:.4f}"
            )


fig.add_annotation(
    x=5,  
    y=0.8,
    xref="x",
    yref="y",
    text=cpp_path_text,
    showarrow=False,
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    bgcolor="white",
    opacity=0.8
)

fig.update_layout(
    title=f"Mediation analysis examining pupil average response on behavioural learning rate <br> mediated by uncertainty parameters (zero updates excluded)",
    width=1000, 
    height=1000,
    xaxis=dict(showticklabels=False), 
    yaxis=dict(showticklabels=False),
    plot_bgcolor='rgba(255, 255, 255, 1)'
)  

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"mediation-average-zero-excluded-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path) 

fig.show()
    