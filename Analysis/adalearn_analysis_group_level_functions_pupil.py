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




#############################################################################################Assessment of averaged pupil response



pupil_df = pd.DataFrame()
pupil_mean_df = pd.DataFrame()


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
            csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_outcome-epoched-pupil-df.csv'
            
            csv_file_path = csv_files_path / csv_file_name
            
            if csv_file_path.exists():
                print(f"CSV file path exists: {csv_file_path}")  
                
                # Read the CSV data
                csv_df = pd.read_csv(csv_file_path)
                print(f'The shape of the dataframe for participant {participant_id} is {csv_df.shape}')

                mean = pd.DataFrame({
                    'mean': csv_df['mean'],
                    'sem': csv_df['sem'],
                    'time': csv_df['time'],
                    'count': range(len(csv_df)),
                    'participant': participant_id})                

                pupil_mean_df = pd.concat([pupil_mean_df, mean])

                mean = csv_df
                
                # Drop columns you don't need
                csv_df = csv_df.drop(columns=['mean', 'std', 'sem', 'time'])

                # Ensure the columns match before concatenation
                csv_df = csv_df.T
                
                # Add the participant ID to the transposed DataFrame
                csv_df['participant'] = participant_id
                
                # Align the columns across all participants (add missing columns with NaN)
                all_columns = pupil_df.columns.union(csv_df.columns)
                pupil_df = pupil_df.reindex(columns=all_columns)
                csv_df = csv_df.reindex(columns=all_columns)

                # Concatenate to the main DataFrame
                pupil_df = pd.concat([pupil_df, csv_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path}")  
        else:
            print(f"CSV folder not found: {csv_files_path}")  


group_pupil_dict = {'mean_pupil': [], 'sem_pupil': [], 'time': []}

for count in pupil_mean_df['count'].unique():
    data = pupil_mean_df[pupil_mean_df['count'] == count]

    mean_pupil = data['mean'].mean()
    sem_pupil = stats.sem(data['mean'])
    time = data['time'].mean()

    group_pupil_dict['mean_pupil'].append(mean_pupil)
    group_pupil_dict['sem_pupil'].append(sem_pupil)
    group_pupil_dict['time'].append(time)

group_pupil_df = pd.DataFrame(group_pupil_dict)


fig = go.Figure()

fig.add_trace(go.Scatter(x=group_pupil_df['time'], y=group_pupil_df['mean_pupil'], mode = 'lines', line=dict(color='rgba(0, 130, 150, 0.8)'), showlegend=False))

fig.add_trace(go.Scatter(
    x=list(group_pupil_df['time']) + list(group_pupil_df['time'][::-1]),
    y=list(group_pupil_df['mean_pupil'] + group_pupil_df['sem_pupil']) + list((group_pupil_df['mean_pupil'] - group_pupil_df['sem_pupil'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')

fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
    )

fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black',
        title_text = 'Pupil size' 
    )

fig.update_layout(height=950, width=1800, title_text="Average epoched pupil response around outcome presentation across participants", plot_bgcolor='rgba(255, 255, 255, 1)')

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-average-epoched-response-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)   

fig.show()


participants = sorted(pupil_mean_df['participant'].unique())
num_participants = len(participants)

cols = 4
rows = int(np.ceil(num_participants / cols))

fig = go.Figure()

fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, 
                           subplot_titles=[f'Participant {p}' for p in participants])
    
for i, participant in enumerate(participants):
    row = (i // cols) + 1
    col = (i % cols) + 1
    sub_df = pupil_mean_df[pupil_mean_df['participant'] == participant]
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['mean'], mode='lines', line=dict(color='rgba(0, 130, 150, 0.8)'), name=f'Participant {participant}', showlegend=False), row=row, col=col)
    
    fig.add_trace(go.Scatter(x=list(sub_df['time']) + list(sub_df['time'][::-1]),
                             y=list(sub_df['mean'] + sub_df['sem']) + list((sub_df['mean'] - sub_df['sem'])[::-1]),
                            fill='toself',
                            fillcolor='rgba(0, 130, 150, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False), row=row, col=col)
    

    fig.add_vline(x=0, line_dash='dot', row=row, col=col)
    fig.add_hline(y=0, line_dash = 'dot', row=row, col=col)

    fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
        row=row, 
        col=col 
    )

    fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        row=row, 
        col=col  
    )

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-average-epoched-response-individual-participants-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)   

fig.update_layout(height=700*rows, width=1500, title_text="Pupil Response per Participant", plot_bgcolor='rgba(255, 255, 255, 1)')

fig.show()




#############################################################################################Assessment of phasic pupil regression coefficients

pupil_regress_coef_df = pd.DataFrame()

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
        csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
        
        if csv_files_path.exists():
            print(f"CSV files path exists: {csv_files_path}") 
            
            participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
            csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_conf-cpp-coef.csv'
            
            csv_file_path = csv_files_path / csv_file_name
            
            if csv_file_path.exists():
                print(f"CSV file path exists: {csv_file_path}")  
                
                # Read the CSV data
                csv_df = pd.read_csv(csv_file_path)
                print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                
                # Add the participant ID to the DataFrame
                csv_df['participant'] = participant_id

                # Concatenate to the main DataFrame
                pupil_regress_coef_df = pd.concat([pupil_regress_coef_df, csv_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path}")  
        else:
            print(f"CSV folder not found: {csv_files_path}")  


pupil_regress_coef_amp = pupil_regress_coef_df[['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals', 'time', 'participant']] 

pupil_regress_coef_amp['cum_count'] = pupil_regress_coef_amp.groupby('participant').cumcount() + 1

range_df = pupil_regress_coef_amp.groupby('participant')['cum_count'].agg(lambda x: x.max() - x.min())



pupil_coef_data = {"conf_coef": [], "conf_coef_error": [], "cpp_coef": [], "cpp_coef_error": [], "dkl_resid_coef": [], "dkl_resid_coef_error": [], "time_mean": [], "time_mean_error": []}

for number in pupil_regress_coef_amp['cum_count'].unique():
    data = pupil_regress_coef_amp[pupil_regress_coef_amp['cum_count'] == number]

    conf_zscore_mean = data['conf_zscore'].mean()
    conf_zscore_sem = stats.sem(data['conf_zscore'])
    cpp_zscore_mean = data['cpp_zscore'].mean()
    cpp_zscore_sem = stats.sem(data['cpp_zscore'])
    dkl_resid_zscore_mean = data['dkl_regress_residuals'].mean()
    dkl_resid_zscore_sem = stats.sem(data['dkl_regress_residuals'])
    time_mean = data['time'].mean()
    time_error = stats.sem(data['time'])

    pupil_coef_data['conf_coef'].append(conf_zscore_mean)
    pupil_coef_data['conf_coef_error'].append(conf_zscore_sem)
    pupil_coef_data['cpp_coef'].append(cpp_zscore_mean)
    pupil_coef_data['cpp_coef_error'].append(cpp_zscore_sem)
    pupil_coef_data['dkl_resid_coef'].append(dkl_resid_zscore_mean)
    pupil_coef_data['dkl_resid_coef_error'].append(dkl_resid_zscore_sem)
    pupil_coef_data['time_mean'].append(time_mean)
    pupil_coef_data['time_mean_error'].append(time_error)

pupil_coef_df = pd.DataFrame(pupil_coef_data)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pupil_coef_df['time_mean'], 
    y=pupil_coef_df['dkl_resid_coef'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
    name='Information gain (regression residuals)'))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_df['time_mean']) + list(pupil_coef_df['time_mean'][::-1]),
    y=list(pupil_coef_df['dkl_resid_coef'] + pupil_coef_df['dkl_resid_coef_error']) + list((pupil_coef_df['dkl_resid_coef'] - pupil_coef_df['dkl_resid_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=pupil_coef_df['time_mean'], 
    y=pupil_coef_df['cpp_coef'], 
    mode= 'lines',
    line=dict(color='rgba(160, 70, 180, 0.8)'), 
    name='Change point probability '))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_df['time_mean']) + list(pupil_coef_df['time_mean'][::-1]),
    y=list(pupil_coef_df['cpp_coef'] + pupil_coef_df['cpp_coef_error']) + list((pupil_coef_df['cpp_coef'] - pupil_coef_df['cpp_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(160, 70, 180, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=pupil_coef_df['time_mean'],
    y=pupil_coef_df['conf_coef'], 
    mode= 'lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'),
    name='Prior confidence'))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_df['time_mean']) + list(pupil_coef_df['time_mean'][::-1]),
    y=list(pupil_coef_df['conf_coef'] + pupil_coef_df['conf_coef_error']) + list((pupil_coef_df['conf_coef'] - pupil_coef_df['conf_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')

fig.update_xaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Regression coefficients",  
)

fig.update_layout(
    title=dict(
            text="Peristimulus regression coefficients for uncertainty parameters on baseline corrected pupil response"
        ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800)

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-regress-coef-baseline-corrected-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)



fig.show()


# Plotting individual coefficient regressions
participants = sorted(pupil_regress_coef_amp['participant'].unique())
num_participants = len(participants)

cols = 4
rows = int(np.ceil(num_participants / cols))

fig = go.Figure()

fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, 
                           subplot_titles=[f'Participant {p}' for p in participants])
    
for i, participant in enumerate(participants):
    row = (i // cols) + 1
    col = (i % cols) + 1
    sub_df = pupil_regress_coef_amp[pupil_regress_coef_amp['participant'] == participant]
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['conf_zscore'], mode='lines', line=dict(color='rgba(0, 130, 150, 0.8)'), name=f'Prior confidence', showlegend=False), row=row, col=col)
    
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['cpp_zscore'], mode='lines', line=dict(color='rgba(160, 70, 180, 0.8)'), name=f'Change point probability', showlegend=False), row=row, col=col)
    
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['dkl_regress_residuals'], mode='lines', line=dict(color='rgba(220, 120, 0, 0.8)'), name=f'Information gain (regression residuals)', showlegend=False), row=row, col=col)
    

    fig.add_vline(x=0, line_dash='dot', row=row, col=col)
    fig.add_hline(y=0, line_dash = 'dot', row=row, col=col)

    fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
        row=row, 
        col=col 
    )

    fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        row=row, 
        col=col  
    )

    
fig.update_layout(height=700*rows, width=1500, title_text="Pupil versus parameters of uncertainty peristimulus regression coefficients per participant", plot_bgcolor='rgba(255, 255, 255, 1)')

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-regress-coef-baseline-corrected-individual-subplots-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)

fig.show()




#############################################################################################Assessment of tonic pupil regression coefficients


pupil_regress_coef_nobc_df = pd.DataFrame()

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
        csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
        
        if csv_files_path.exists():
            print(f"CSV files path exists: {csv_files_path}") 
            
            participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
            csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_conf-cpp-coef-nobc.csv'
            
            csv_file_path = csv_files_path / csv_file_name
            
            if csv_file_path.exists():
                print(f"CSV file path exists: {csv_file_path}")  
                
                # Read the CSV data
                csv_df = pd.read_csv(csv_file_path)
                print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                
                # Add the participant ID to the DataFrame
                csv_df['participant'] = participant_id

                # Concatenate to the main DataFrame
                pupil_regress_coef_nobc_df = pd.concat([pupil_regress_coef_nobc_df, csv_df], ignore_index=True)
                
            else:
                print(f"CSV file not found: {csv_file_path}")  
        else:
            print(f"CSV folder not found: {csv_files_path}")  


pupil_regress_coef_nobc_amp = pupil_regress_coef_nobc_df[['conf_zscore', 'cpp_zscore', 'dkl_regress_residuals', 'time', 'participant']] 

pupil_regress_coef_nobc_amp['cum_count'] = pupil_regress_coef_nobc_amp.groupby('participant').cumcount() + 1

range_df = pupil_regress_coef_nobc_amp.groupby('participant')['cum_count'].agg(lambda x: x.max() - x.min())


pupil_coef_nobc_data = {"conf_coef": [], "conf_coef_error": [], "cpp_coef": [], "cpp_coef_error": [], "dkl_resid_coef": [], "dkl_resid_coef_error": [], "time_mean": [], "time_mean_error": []}

for number in pupil_regress_coef_nobc_amp['cum_count'].unique():
    data = pupil_regress_coef_nobc_amp[pupil_regress_coef_nobc_amp['cum_count'] == number]

    conf_zscore_mean = data['conf_zscore'].mean()
    conf_zscore_sem = stats.sem(data['conf_zscore'])
    cpp_zscore_mean = data['cpp_zscore'].mean()
    cpp_zscore_sem = stats.sem(data['cpp_zscore'])
    dkl_resid_zscore_mean = data['dkl_regress_residuals'].mean()
    dkl_resid_zscore_sem = stats.sem(data['dkl_regress_residuals'])
    time_mean = data['time'].mean()
    time_error = stats.sem(data['time'])

    pupil_coef_nobc_data['conf_coef'].append(conf_zscore_mean)
    pupil_coef_nobc_data['conf_coef_error'].append(conf_zscore_sem)
    pupil_coef_nobc_data['cpp_coef'].append(cpp_zscore_mean)
    pupil_coef_nobc_data['cpp_coef_error'].append(cpp_zscore_sem)
    pupil_coef_nobc_data['dkl_resid_coef'].append(dkl_resid_zscore_mean)
    pupil_coef_nobc_data['dkl_resid_coef_error'].append(dkl_resid_zscore_sem)
    pupil_coef_nobc_data['time_mean'].append(time_mean)
    pupil_coef_nobc_data['time_mean_error'].append(time_error)

pupil_coef_nobc_df = pd.DataFrame(pupil_coef_nobc_data)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pupil_coef_nobc_df['time_mean'], 
    y=pupil_coef_nobc_df['dkl_resid_coef'], 
    mode= 'lines',
    line=dict(color='rgba(220, 120, 0, 0.8)'),  
    name='Information gain (regression residuals)'))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_nobc_df['time_mean']) + list(pupil_coef_nobc_df['time_mean'][::-1]),
    y=list(pupil_coef_nobc_df['dkl_resid_coef'] + pupil_coef_nobc_df['dkl_resid_coef_error']) + list((pupil_coef_nobc_df['dkl_resid_coef'] - pupil_coef_nobc_df['dkl_resid_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(220, 120, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=pupil_coef_nobc_df['time_mean'], 
    y=pupil_coef_nobc_df['cpp_coef'], 
    mode= 'lines',
    line=dict(color='rgba(160, 70, 180, 0.8)'), 
    name='Change point probability '))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_nobc_df['time_mean']) + list(pupil_coef_nobc_df['time_mean'][::-1]),
    y=list(pupil_coef_nobc_df['cpp_coef'] + pupil_coef_nobc_df['cpp_coef_error']) + list((pupil_coef_nobc_df['cpp_coef'] - pupil_coef_nobc_df['cpp_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(160, 70, 180, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=pupil_coef_nobc_df['time_mean'],
    y=pupil_coef_nobc_df['conf_coef'], 
    mode= 'lines',
    line=dict(color='rgba(0, 130, 150, 0.8)'),
    name='Prior confidence'))

fig.add_trace(go.Scatter(
    x=list(pupil_coef_nobc_df['time_mean']) + list(pupil_coef_nobc_df['time_mean'][::-1]),
    y=list(pupil_coef_nobc_df['conf_coef'] + pupil_coef_nobc_df['conf_coef_error']) + list((pupil_coef_nobc_df['conf_coef'] - pupil_coef_nobc_df['conf_coef_error'])[::-1]),
    fill='toself',
    fillcolor='rgba(0, 130, 150, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.add_vline(x=0, line_dash='dot')
fig.add_hline(y=0, line_dash = 'dot')

fig.update_xaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Time relative to outcome presentation (s)", 
)

fig.update_yaxes(
    showline=True, 
    linewidth=1, 
    linecolor='black', 
    title_text="Regression coefficients",  
)

fig.update_layout(
    title=dict(
            text="Peristimulus regression coefficients for uncertainty parameters on non-baseline corrected pupil response"
        ),
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=950,
    width=1800)

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-regress-coef-nonbaseline-corrected-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)

fig.show()

# Plotting individual coefficient regressions
participants = sorted(pupil_regress_coef_nobc_amp['participant'].unique())
num_participants = len(participants)

cols = 4
rows = int(np.ceil(num_participants / cols))

fig = go.Figure()

fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, 
                           subplot_titles=[f'Participant {p}' for p in participants])
    
for i, participant in enumerate(participants):
    row = (i // cols) + 1
    col = (i % cols) + 1
    sub_df = pupil_regress_coef_nobc_amp[pupil_regress_coef_nobc_amp['participant'] == participant]
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['conf_zscore'], mode='lines', line=dict(color='rgba(0, 130, 150, 0.8)'), name=f'Prior confidence', showlegend=False), row=row, col=col)
    
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['cpp_zscore'], mode='lines', line=dict(color='rgba(160, 70, 180, 0.8)'), name=f'Change point probability', showlegend=False), row=row, col=col)
    
    fig.add_trace(go.Scatter(x=sub_df['time'], y=sub_df['dkl_regress_residuals'], mode='lines', line=dict(color='rgba(220, 120, 0, 0.8)'), name=f'Information gain (regression residuals)', showlegend=False), row=row, col=col)
    

    fig.add_vline(x=0, line_dash='dot', row=row, col=col)
    fig.add_hline(y=0, line_dash = 'dot', row=row, col=col)

    fig.update_xaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        title_text="Time relative to outcome presentation (s)",
        row=row, 
        col=col 
    )

    fig.update_yaxes(
        showline=True, 
        linewidth=1, 
        linecolor='black', 
        row=row, 
        col=col  
    )

    
fig.update_layout(height=700*rows, width=1500, title_text="Pupil versus parameters of uncertainty peristimulus regression coefficients per participant", plot_bgcolor='rgba(255, 255, 255, 1)')

# Save the figure to the group level plots folder
plot_dir = os.path.join(base_dir, "group_level_plots")
os.makedirs(plot_dir, exist_ok=True)
file_name = f"pupil-regress-coef-nonbaseline-corrected-individual-subplots-(Adalearn-Pupil).png"
file_path = os.path.join(plot_dir, file_name)
fig.write_image(file_path)

fig.show()


