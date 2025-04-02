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

from adalearn_pupil.adalearn_pupil_participant_characteristics import base_dir, condition_str
from adalearn_pupil.adalearn_pupil_group_level_participant_demographics import excluded_participants

def sample_absolute_error_and_cv(condition_str, base_dir, excluded_participants, save_plot = True):
    """
    Function to plot the absolute error and coefficient of variation in absolute error across task sessions.
    
    Parameters:
    - condition_strs: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)
    
    Returns:
    - mean_abs_error_avg_df: pd.DataFrame, contains the grouped absolute error and coefficient of variation across task sessions.
    """
    mean_abs_error_group_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}")  
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}") 
                
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_mean-abs-error-df.csv'
                
                csv_file_path = csv_files_path / csv_file_name
                
                if csv_file_path.exists():
                    print(f"CSV file path exists: {csv_file_path}")  
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path)
                    print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    mean_abs_error_group_df = pd.concat([mean_abs_error_group_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path}")  
            else:
                print(f"CSV folder not found: {csv_files_path}")  


    mean_abs_error_group_df['participant'] = mean_abs_error_group_df['participant'].astype(int)
    mean_abs_error_group_df['coef_variation'] = mean_abs_error_group_df['std_abs_error']/ mean_abs_error_group_df['mean_abs_error']

    task_session_mean = mean_abs_error_group_df.groupby('task_session')['mean_abs_error'].mean()
    task_session_count = mean_abs_error_group_df.groupby('task_session')['mean_abs_error'].count()
    task_session_sd = mean_abs_error_group_df.groupby('task_session')['std_abs_error'].mean()
    task_session_cv = mean_abs_error_group_df.groupby('task_session')['coef_variation'].mean()

    mean_abs_error_avg_df = pd.DataFrame({
        'mean': task_session_mean,
        'std': task_session_sd,
        'cv': task_session_cv,
        'count': task_session_count,
        'task_session': mean_abs_error_group_df['task_session'].unique()
    })


    # Plot the mean absolute error across task sessions
    if save_plot:
        fig = go.Figure()

        for id in mean_abs_error_group_df['participant'].unique():
            participant_data = mean_abs_error_group_df[mean_abs_error_group_df['participant'] == id]
            
            fig.add_trace(go.Scatter(
                x=participant_data['task_session'], 
                y=participant_data['mean_abs_error'], 
                mode='lines+markers', 
                line=dict(color='rgba(0, 0, 255, 0.12)'),
                marker=dict(color='rgba(0, 0, 255, 0.12)', size=6)
            ))

        fig.add_trace(go.Scatter(
            x=mean_abs_error_avg_df['task_session'], 
            y=mean_abs_error_avg_df['mean'], 
            mode='lines+markers', 
            marker=dict(color='rgba(0, 0, 255, 1.0)', size=12)
        ))

        fig.update_layout(
            title=dict(
                text="Mean absolute error across task sessions"
            ),
            xaxis=dict(
                tickvals=list(range(1, 23)),
                title=dict(
                    text="Task session"
                ),
            ),
            yaxis=dict(
                title=dict(
                    text="Mean absolute error"
                ),
            ),
            height=950,
            width=1800,
            template = 'simple_white',
            showlegend=False
        )

        fig.update_yaxes(autorangeoptions_include=0)

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"mean-absolute-error-across-task-sessions-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)


        fig.show()

        # Plot the coefficient of variation of absolute error across task sessions
        fig = go.Figure()

        for id in mean_abs_error_group_df['participant'].unique():
            participant_data = mean_abs_error_group_df[mean_abs_error_group_df['participant'] == id]
            
            fig.add_trace(go.Scatter(
                x=participant_data['task_session'], 
                y=participant_data['coef_variation'], 
                mode='lines+markers', 
                line=dict(color='rgba(0, 0, 255, 0.12)'),
                marker=dict(color='rgba(0, 0, 255, 0.12)', size=6)
            ))

        fig.add_trace(go.Scatter(
            x=mean_abs_error_avg_df['task_session'], 
            y=mean_abs_error_avg_df['cv'], 
            mode='lines+markers', 
            marker=dict(color='rgba(0, 0, 255, 1.0)', size=12)
        ))

        fig.update_layout(
            title=dict(
                text="Coefficient of variation for absolute error across task sessions"
            ),
            xaxis=dict(
                tickvals=list(range(1, 23)),
                title=dict(
                    text="Task session"
                ),
            ),
            yaxis=dict(
                title=dict(
                    text="Coefficient of variation (absolute error)"
                ),
            ),
            height=950,
            width=1800,
            template = 'simple_white',
            showlegend=False
        )

        fig.update_yaxes(autorangeoptions_include=0)

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"cv-across-task-sessions-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)

        fig.show()


    return mean_abs_error_avg_df



def sample_behav_v_model_estimate(condition_str, base_dir, excluded_participants, save_plot = True, plot_correlation_histogram=True):
    """
    Function to plot the behavioural estimates versus model predictions across participants. Runs analysis on correlation coefficients and regression analysis, grouped across participants. Option to plot a histogram of the recorded correlation coefficients between behavioural and model-derived learning rates across participants. Threshold defining a consideration of minimum perfromance based on the bottom 10% of correlation coefficients recorded in Cedric anfd Florent's previous analysis (https://doi.org/10.1162/opmi_a_00139)

    Parameters:
    - condition_str: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)
    - plot_correlation_histogram: boolean, if true, will plot a histogram of the correlation coefficients between behavioural and model estimates (default = True).


    Returns:
    - behav_model_estimate_regress_df: pd.DataFrame, contains the group-level outcomes of the regression analysis.
    - behav_model_estimate_corr_df: pd.DataFrame, contains the group-level outcomes of the correlation analysis
    - bin_average_data: pd.DataFrame, contains the binned average estimates.   
    """
    # Assesses the correlation and regression analysis performed within each participant at the group level 
    behav_model_linear_analysis_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}") 
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}") 
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_behav-v-model-estimate.csv'
                
                csv_file_path = csv_files_path / csv_file_name
                
                if csv_file_path.exists():
                    print(f"CSV file path exists: {csv_file_path}") 
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path)
                    print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_model_linear_analysis_df = pd.concat([behav_model_linear_analysis_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path}")  
            else:
                print(f"CSV folder not found: {csv_files_path}") 

    behav_model_intercept_df = behav_model_linear_analysis_df[behav_model_linear_analysis_df['variable'] == 'const']
    behav_model_estimate_intercept_mean = behav_model_intercept_df['slope'].mean()
    behav_model_estimate_intercept_std = behav_model_intercept_df['slope'].std()
    behav_model_estimate_intercept_sem = behav_model_intercept_df['slope'].sem()

    behav_model_slope_df = behav_model_linear_analysis_df[behav_model_linear_analysis_df['variable'] == 'ideal_observer']
    behav_model_estimate_slope_mean = behav_model_slope_df['slope'].mean()
    behav_model_estimate_slope_std = behav_model_intercept_df['slope'].std()
    behav_model_estimate_slope_sem = behav_model_intercept_df['slope'].sem()
    behav_model_slope_ttest_zero = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=0)
    behav_model_slope_ttest_one = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=1)

    behav_model_estimate_regress_df = pd.DataFrame({
        'mean_intercept': [behav_model_estimate_intercept_mean],
        'std_intercept': [behav_model_estimate_intercept_std],
        'sem_intercept': [behav_model_estimate_intercept_sem],
        'mean_slope': [behav_model_estimate_slope_mean],
        'std_slope': [behav_model_estimate_slope_std],
        'sem_slope': [behav_model_estimate_slope_sem],
        'tvalue_against_zero': [behav_model_slope_ttest_zero[0]],
        'pvalue_against_zero': [behav_model_slope_ttest_zero[1]],
        'tvalue_against_one': [behav_model_slope_ttest_one[0]],
        'pvalue_against_one': [behav_model_slope_ttest_one[1]],
        })

    behav_model_corr_mean = behav_model_slope_df['correlation'].mean()
    behav_model_corr_std = behav_model_slope_df['correlation'].std()
    behav_model_corr_sem = behav_model_slope_df['correlation'].sem()
    behav_model_corr_ttest = stats.ttest_1samp(behav_model_slope_df['correlation'], popmean=0)

    behav_model_estimate_corr_df = pd.DataFrame({
        'mean_corr': [behav_model_corr_mean],
        'std_corr': [behav_model_corr_std],
        'sem_corr': [behav_model_corr_sem],
        'tvalue_against_zero': [behav_model_corr_ttest[0]],
        'pvalue_against_zero': [behav_model_corr_ttest[1]],
        })


    # Bins and plots behavioural estimates against model estimates
    behav_model_estimate_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}") 
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}") 
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_behavioural-df.csv'
                
                csv_file_path = csv_files_path / csv_file_name
                
                if csv_file_path.exists():
                    print(f"CSV file path exists: {csv_file_path}") 
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path)
                    print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_model_estimate_df = pd.concat([behav_model_estimate_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path}")  
            else:
                print(f"CSV folder not found: {csv_files_path}")  


    # Evaluating relationship between participants' estimates and model estimates
    nbins_model_estimate = 6

    _, bin_edges_model_estimate = pd.qcut(behav_model_estimate_df["ideal_observer"], nbins_model_estimate, retbins=True)

    bin_midpoints = [(bin_edges_model_estimate[i] + bin_edges_model_estimate[i+1]) / 2 for i in range(len(bin_edges_model_estimate) - 1)]

    behav_model_estimate_df["model_estimate_bin"] = pd.cut(behav_model_estimate_df["ideal_observer"], bins=bin_edges_model_estimate, labels=False, include_lowest=True)

    behav_model_estimate_df["model_estimate_bin_label"] = behav_model_estimate_df["model_estimate_bin"].apply(lambda x: bin_midpoints[x])

    # Storing the average learning rate at the participant level
    binned_data =  {"model_estimate_bin": [], "model_estimate_bin_label": [], "avg_estimate": [], "participant": []}

    for participant in behav_model_estimate_df["participant"].unique():
        participant_data = behav_model_estimate_df[behav_model_estimate_df["participant"] == participant]
        
        for model_estimate_bin in sorted(participant_data["model_estimate_bin"].unique()):
                
            bin_data = participant_data[
                (participant_data["model_estimate_bin"] == model_estimate_bin)]
            
            avg_estimate = bin_data["estimate"].mean()
            bin_label = bin_data['model_estimate_bin_label'].unique()[0]  

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

    if save_plot:
        fig = go.Figure()

        for participant in participant_v_model_df['participant'].unique():
            data = participant_v_model_df[participant_v_model_df['participant']==participant]

            fig.add_trace(go.Scatter(
                x=data['model_estimate_bin_label'], 
                y=data['avg_estimate'], 
                mode='lines+markers',
                line=dict(color='rgba(0, 0, 255, 0.2)'),
                showlegend=False))

        fig.add_trace(go.Scatter(
            x=group_behav_v_model_estimate_df['model_estimate_bin_label'],
            y=group_behav_v_model_estimate_df['group_avg_estimate'], 
            mode='markers',
            marker=dict(color='rgba(0, 0, 255, 0.8)', size=14), 
            showlegend=False, 
            ))
        
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Normative estimate", 
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Participant estimate",   
        )

        fig.update_layout(
            title=dict(text="Relationship between participant and normative estimates"),
            xaxis=dict(
                range=[0.1, 0.9],  
                title=dict(text="Normative estimate"),
            ),
            yaxis=dict(
                title=dict(text="Participant estimate")
            ),
            height=950,
            width=1800,
            template='simple_white',
            showlegend=False
        )

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"behav-v-model-estimates-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)

        fig.show()

        if plot_correlation_histogram:
            fig = go.Figure(data=[go.Histogram(x = behav_model_slope_df['correlation'])])

            fig.add_vline(x=0.6275770499186455, line_dash='dash')

            fig.update_layout(
            title=dict(text="Histogram of correlation coefficients defining the relationship between participant and normative estimates"),
            xaxis=dict(
                range=[0, 1.0], 
                title=dict(text="Correlation coefficient"),
            ),
            yaxis=dict(
                title=dict(text="Number of participants")
            ),
            paper_bgcolor='white',
            plot_bgcolor='white', 
            height=950,
            width=1800,
            showlegend=False
            )

            fig.update_xaxes(
                linecolor='black',              
                tickmode='auto'                                  
            )

            fig.update_yaxes(    
                linecolor='black',              
                tickmode='linear'                     
            )

            # Save the figure to the group level plots folder
            plot_dir = os.path.join(base_dir, "group_level_plots")
            os.makedirs(plot_dir, exist_ok=True)
            file_name = f"behav-v-model-estimates-correlation-histogram-(Adalearn-Pupil).png"
            file_path = os.path.join(plot_dir, file_name)
            fig.write_image(file_path)


            fig.show()


    return behav_model_estimate_regress_df, behav_model_estimate_corr_df, bin_average_data



def sample_behav_v_model_learning_rate(condition_str, base_dir, excluded_participants, exclude_zero=False, save_plot = True):
    """
    Function to plot the behaviourally- versus model-derived learning rates. Runs analysis on correlation coefficients and regression analysis, grouped across participants.

    Parameters:
    - condition_str: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - exclude_zero: boolean, if true, will plot and analyse data excluding zero learning rates (default = False).
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)

    Returns:
    - behav_model_lr_regress_df: pd.DataFrame, contains the group-level outcomes of the behaviourally- v model-derived learning rate regression analysis. 
    - behav_model_estimate_corr_df: pd.DataFrame, contains the group-level outcomes of the behaviourally- v model-derived learning rate correlation analysis
    - bin_average_data: pd.DataFrame, contains the binned average estimates.  
    """
    # Assesses the correlation and regression analysis performed within each participant at the group level 
    behav_model_lr_linear_analysis_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}") 
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}") 
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                if exclude_zero:
                     csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_behav-v-model-lr-zero-excluded.csv'
                else:
                    csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_behav-v-model-lr.csv'
                
                csv_file_path = csv_files_path / csv_file_name
                
                if csv_file_path.exists():
                    print(f"CSV file path exists: {csv_file_path}") 
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path)
                    print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_model_lr_linear_analysis_df = pd.concat([behav_model_lr_linear_analysis_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path}")  
            else:
                print(f"CSV folder not found: {csv_files_path}") 

    behav_model_lr_intercept_df = behav_model_lr_linear_analysis_df[behav_model_lr_linear_analysis_df['variable'] == 'const']
    behav_model_lr_intercept_mean = behav_model_lr_intercept_df['slope'].mean()
    behav_model_lr_intercept_std = behav_model_lr_intercept_df['slope'].std()
    behav_model_lr_intercept_sem = behav_model_lr_intercept_df['slope'].sem()

    behav_model_lr_slope_df = behav_model_lr_linear_analysis_df[behav_model_lr_linear_analysis_df['variable'] == 'cleaned_learning_rate_ideal_observer']
    behav_model_lr_slope_mean = behav_model_lr_slope_df['slope'].mean()
    behav_model_lr_slope_std = behav_model_lr_slope_df['slope'].std()
    behav_model_lr_slope_sem = behav_model_lr_slope_df['slope'].sem()
    behav_model_lr_slope_ttest_zero = stats.ttest_1samp(behav_model_lr_slope_df['slope'], popmean=0)
    behav_model_lr_slope_ttest_one = stats.ttest_1samp(behav_model_lr_slope_df['slope'], popmean=1)

    behav_model_lr_regress_df = pd.DataFrame({
        'mean_intercept': [behav_model_lr_intercept_mean],
        'std_intercept': [behav_model_lr_intercept_std],
        'sem_intercept': [behav_model_lr_intercept_sem],
        'mean_slope': [behav_model_lr_slope_mean],
        'std_slope': [behav_model_lr_slope_std],
        'sem_slope': [behav_model_lr_slope_sem],
        'tvalue_against_zero': [behav_model_lr_slope_ttest_zero[0]],
        'pvalue_against_zero': [behav_model_lr_slope_ttest_zero[1]],
        'tvalue_against_one': [behav_model_lr_slope_ttest_one[0]],
        'pvalue_against_one': [behav_model_lr_slope_ttest_one[1]],
        })

    behav_model_lr_corr_mean = behav_model_lr_slope_df['correlation'].mean()
    behav_model_lr_corr_std = behav_model_lr_slope_df['correlation'].std()
    behav_model_lr_corr_sem = behav_model_lr_slope_df['correlation'].sem()
    behav_model_lr_corr_ttest = stats.ttest_1samp(behav_model_lr_slope_df['correlation'], popmean=0)

    behav_model_lr_corr_df = pd.DataFrame({
        'mean_corr': [behav_model_lr_corr_mean],
        'std_corr': [behav_model_lr_corr_std],
        'sem_corr': [behav_model_lr_corr_sem],
        'tvalue_against_zero': [behav_model_lr_corr_ttest[0]],
        'pvalue_against_zero': [behav_model_lr_corr_ttest[1]],
        })

    # Plotting relationship behaviourally- and model-derived learning rates (option to remove zero updates)
    behav_model_lr_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}") 
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}") 
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                csv_file_name = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_computational-var-df.csv'
                
                csv_file_path = csv_files_path / csv_file_name
                
                if csv_file_path.exists():
                    print(f"CSV file path exists: {csv_file_path}") 
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path)
                    print(f"Data from {csv_file_name}:\n{csv_df.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_model_lr_df = pd.concat([behav_model_lr_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path}")  
            else:
                print(f"CSV folder not found: {csv_files_path}")  

    if exclude_zero:
        behav_model_lr_df = behav_model_lr_df[behav_model_lr_df['cleaned_learning_rate_estimate'] != 0]
        title_text = "(zero learning rates excluded)"
        figure_text= "-(zero-excluded)"
    else:
        title_text = ""
        figure_text = ""


    behav_model_lr_df['model_bin'] = pd.cut(behav_model_lr_df['cleaned_learning_rate_ideal_observer'], bins=6)
    behav_model_lr_df['bin_mid'] = behav_model_lr_df['model_bin'].apply(lambda x: x.mid)
    participant_averaged_data = behav_model_lr_df.groupby(['participant', 'bin_mid'])['estimate'].mean().reset_index()
    bin_average_data =  behav_model_lr_df.groupby(['bin_mid'])['estimate'].mean().reset_index()

    if save_plot:
        fig = go.Figure()

        for participant in participant_averaged_data['participant'].unique():
            participant_data = participant_averaged_data[participant_averaged_data['participant'] == participant]
        
            fig.add_trace(go.Scatter(x=participant_data['bin_mid'], y=participant_data['estimate'], mode='lines+markers', marker=dict(color='rgba(0,0,255,0.12)')))

        fig.add_trace(go.Scatter(x=bin_average_data['bin_mid'], y=bin_average_data['estimate'], mode='markers', marker=dict(color='rgba(0,0,255,1.0)',size=10)))


        fig.update_layout(
            title=dict(text=f"Relationship between participant and normative learning rates {title_text}"),
            xaxis=dict(  
                title=dict(text="Normative learning rate"),
            ),
            yaxis=dict(
                title=dict(text="Participant learning rate")
            ),
            height=950,
            width=1800,
            template='simple_white',
            showlegend=False
        )

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"behav-v-model-learning-rates{figure_text}-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)

        fig.show()


    return behav_model_lr_regress_df, behav_model_lr_corr_df, bin_average_data



def sample_learning_rates_by_cp(condition_str, base_dir, excluded_participants, exclude_zero=False, save_plot = True):
    """
    Function to plot the learning rates around change points across task sessions. Also analyses the correlations performed at the participant level for behavioural and model-derived learning rates around change points.
    
    Parameters:
    - condition_str: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - exclude_zero: boolean, if true, will plot and analyse data excluding zero learning rates (default = False).
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)
    
    Returns:
    - lr_by_cp_avg_df_behav: pd.DataFrame, contains the grouped learning rates around change points across task sessions derived from behavioural estimates.
    - lr_by_cp_avg_df_model: pd.DataFrame, contains the grouped learning rates around change points across task sessions derived from model estimates.
    """
    # Assessment of learning rates around change points
    behav_lr_by_cp_group_df = pd.DataFrame()
    model_lr_by_cp_group_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}")  

            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}")  
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                if exclude_zero:
                    csv_file_name_behav = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_averaged-behav-lr-by-cp-zero-excluded.csv'

                    title = "-zero-excluded"
                else:
                    csv_file_name_behav = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_averaged-behav-lr-by-cp.csv' 

                    title = ""

                csv_file_name_model = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_averaged-model-lr-by-cp.csv'
                
                csv_file_path_behav = csv_files_path / csv_file_name_behav
                
                if csv_file_path_behav.exists():
                    print(f"CSV file path exists: {csv_file_path_behav}") 
                    
                    # Read the CSV data
                    csv_df_behav = pd.read_csv(csv_file_path_behav)
                    print(f"Data from {csv_file_name_behav}:\n{csv_df_behav.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df_behav['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_lr_by_cp_group_df = pd.concat([behav_lr_by_cp_group_df, csv_df_behav], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_behav}")
                    
                    
                csv_file_path_model = csv_files_path / csv_file_name_model
                
                if csv_file_path_model.exists():
                    print(f"CSV file path exists: {csv_file_path_model}")  
                    
                    # Read the CSV data
                    csv_df_model = pd.read_csv(csv_file_path_model)
                    print(f"Data from {csv_file_name_model}:\n{csv_df_model.head()}") 
                    
                    # Add the participant ID to the DataFrame
                    csv_df_model['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    model_lr_by_cp_group_df = pd.concat([model_lr_by_cp_group_df, csv_df_model], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_model}")  
            else:
                print(f"CSV folder not found: {csv_files_path}")  


    behav_lr_by_cp_group_df['participant'] = behav_lr_by_cp_group_df['participant'].astype(int)
    model_lr_by_cp_group_df['participant'] = model_lr_by_cp_group_df['participant'].astype(int)


    lr_by_cp_mean_behav = behav_lr_by_cp_group_df.groupby('relative_position')['mean'].mean()
    lr_by_cp_count_behav = behav_lr_by_cp_group_df.groupby('relative_position')['mean'].count()
    lr_by_cp_std_behav = behav_lr_by_cp_group_df.groupby('relative_position')['mean'].std()
    lr_by_cp_sem_behav = lr_by_cp_std_behav / np.sqrt(lr_by_cp_count_behav)

    lr_by_cp_avg_df_behav = pd.DataFrame({
        'mean': lr_by_cp_mean_behav,
        'std': lr_by_cp_std_behav,
        'sem': lr_by_cp_sem_behav,
        'count': lr_by_cp_count_behav
    })


    lr_by_cp_mean_model = model_lr_by_cp_group_df.groupby('relative_position')['mean'].mean()
    lr_by_cp_count_model = model_lr_by_cp_group_df.groupby('relative_position')['mean'].count()
    lr_by_cp_std_model = model_lr_by_cp_group_df.groupby('relative_position')['mean'].std()
    lr_by_cp_sem_model = lr_by_cp_std_model / np.sqrt(lr_by_cp_count_model)

    lr_by_cp_avg_df_model = pd.DataFrame({
        'mean': lr_by_cp_mean_model,
        'std': lr_by_cp_std_model,
        'sem': lr_by_cp_sem_model,
        'count': lr_by_cp_count_model
    })

    # Assessment of correlation between behviourally- and model-derived learning rates around change points
    behav_lr_by_cp_points_df = pd.DataFrame()
    behav_lr_by_cp_corr_df = pd.DataFrame()
    
    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}")  

            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}")  
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                if exclude_zero:
                    csv_file_name_point_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_lr-around-cp-zero-excluded.csv'

                    title = "-zero-excluded"
                else:
                    csv_file_name_point_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_lr-around-cp.csv' 

                    title = ""

                if exclude_zero:
                    csv_file_name_corr_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_lr-around-cp-corr-zero-excluded.csv'

                else:
                    csv_file_name_corr_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_lr-around-cp-corr.csv' 


                csv_file_path_point = csv_files_path / csv_file_name_point_df
                csv_file_path_corr = csv_files_path / csv_file_name_corr_df
                
                if csv_file_path_point.exists():
                    print(f"CSV file path exists: {csv_file_path_point}") 
                    
                    # Read the CSV data
                    csv_df_point = pd.read_csv(csv_file_path_point)
                    print(f"Data from {csv_file_name_point_df}:\n{csv_df_point.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df_point['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_lr_by_cp_points_df = pd.concat([behav_lr_by_cp_points_df, csv_df_point], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_point}")

                if csv_file_path_corr.exists():
                    print(f"CSV file path exists: {csv_file_path_corr}") 
                    
                    # Read the CSV data
                    csv_df_corr = pd.read_csv(csv_file_path_corr)
                    print(f"Data from {csv_file_name_corr_df}:\n{csv_df_corr.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df_corr['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_lr_by_cp_corr_df = pd.concat([behav_lr_by_cp_corr_df, csv_df_corr], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_corr}")
            else:
                print(f"CSV folder not found: {csv_files_path}")  

    # Saving correlation assessment in a dataframe
    lr_around_cp_corr_mean = behav_lr_by_cp_corr_df['correlation'].mean()
    lr_around_cp_corr_std = behav_lr_by_cp_corr_df['correlation'].std()
    lr_around_cp_corr_sem = behav_lr_by_cp_corr_df['correlation'].sem()
    lr_around_cp_corr_ttest = stats.ttest_1samp(behav_lr_by_cp_corr_df['correlation'], popmean=0)

    behav_model_lr_round_cp_corr_df = pd.DataFrame({
    'mean_corr': [lr_around_cp_corr_mean],
    'std_corr': [lr_around_cp_corr_std],
    'sem_corr': [lr_around_cp_corr_sem],
    'tvalue_against_zero': [lr_around_cp_corr_ttest[0]],
    'pvalue_against_zero': [lr_around_cp_corr_ttest[1]],
    })


    # Plot the learning rates around change points
    if save_plot:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_behav.index, 
            y=lr_by_cp_avg_df_behav['mean'], 
            mode='lines', 
            line=dict(color='blue'),
            name="Behavior Mean"
        ))


        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_behav.index, 
            y=lr_by_cp_avg_df_behav['mean'] + lr_by_cp_avg_df_behav['sem'], 
            mode='lines', 
            line=dict(color='rgba(0, 0, 255, 0)'),  
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_behav.index, 
            y=lr_by_cp_avg_df_behav['mean'] - lr_by_cp_avg_df_behav['sem'], 
            mode='lines', 
            fill='tonexty', 
            fillcolor='rgba(0, 0, 255, 0.2)',  
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False
        ))

        fig.update_layout(
            xaxis2=dict(domain=[0.7, 1.0], anchor='y2'),
            yaxis2=dict(domain=[0.7, 1.0], anchor='x2')
        )

        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_model.index, 
            y=lr_by_cp_avg_df_model['mean'], 
            xaxis="x2", 
            yaxis="y2",
            mode='lines',
            line=dict(color='red'),
            name="Model Mean"
        ))

        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_model.index, 
            y=lr_by_cp_avg_df_model['mean'] + lr_by_cp_avg_df_model['sem'],
            xaxis="x2", 
            yaxis="y2", 
            mode='lines', 
            line=dict(color='rgba(255, 0, 0, 0)'),  
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=lr_by_cp_avg_df_model.index, 
            y=lr_by_cp_avg_df_model['mean'] - lr_by_cp_avg_df_model['sem'],
            xaxis="x2", 
            yaxis="y2", 
            mode='lines', 
            fill='tonexty', 
            fillcolor='rgba(255, 0, 0, 0.2)', 
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False
        ))

        if exclude_zero:
            title_text = "Behaviourally- and model-derived learning rates relative to change point (zero learning rates excluded)"
        else:
            title_text = "Behaviourally- and model-derived learning rates relative to change point"

        fig.update_layout(
            title=dict(text=title_text),
            
            xaxis=dict(
                range=[-3, 11], 
                title=dict(text="Outcome relative to change point"),
            ),
            yaxis=dict(
                title=dict(text="Learning rate (behavioural)")
            ),

            xaxis2=dict(
                domain=[0.7, 1.0],  
                anchor="y2",
                range=[-3, 11],  
                title=dict(text="Outcome relative to change point")  
            ),
            yaxis2=dict(
                domain=[0.7, 1.0], 
                anchor="x2",
                title=dict(text="Learning rate (model)")  
            ),
            height=950,
            width=1800,
            template='simple_white',
            showlegend=False
        )

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"behav-learning-rates-around-change-points{title}-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)


        fig.show()


    return lr_by_cp_avg_df_behav, lr_by_cp_avg_df_model, behav_model_lr_round_cp_corr_df, 



def model_uncertainty_on_learning_rate(condition_str, base_dir, excluded_participants, save_plot = True):
    """
    Function to plot the relationships between model-derived learning rare and parameters of uncertainty (i.e. 'change point probability' and 'prior confidence'). Serves as normative comparison for behavioural assessments.

    Parameters:
    - condition_str: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)

    Returns:
    - model_uncert_on_lr_df: pd.DataFrame, contains the regression coefficients for uncertainty parameters on learning rate for all partcipants included in the final sample. 
    - model_group_avg_df: pd.DataFrame, contains the binned learning rate data across the uncertainty parameters. 
    """
    computational_var_dfs = pd.DataFrame()
    model_uncert_on_lr_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}")  
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}")  
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')
                csv_file_model_uncert = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_uncertainty-v-lr-model-df.csv'
                
                csv_file_path_model_uncert = csv_files_path / csv_file_model_uncert
                
                if csv_file_path_model_uncert.exists():
                    print(f"CSV file path exists: {csv_file_path_model_uncert}")  
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path_model_uncert)
                    print(f"Data from {csv_file_model_uncert}:\n{csv_df.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    model_uncert_on_lr_df = pd.concat([model_uncert_on_lr_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_model_uncert}") 


                csv_file_com_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_computational-var-df.csv'
                
                csv_file_path_com_df = csv_files_path / csv_file_com_df
                
                if csv_file_path_com_df.exists():
                    print(f"CSV file path exists: {csv_file_com_df}")  
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path_com_df)
                    print(f"Data from {csv_file_com_df}:\n{csv_df.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    computational_var_dfs = pd.concat([computational_var_dfs, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_com_df}")  
            else:
                print(f"CSV folder not found: {csv_files_path}") 


    model_conf_coef = model_uncert_on_lr_df[model_uncert_on_lr_df['variable'] == 'conf_zscore'].sort_values('participant')
    model_cpp_coef = model_uncert_on_lr_df[model_uncert_on_lr_df['variable'] == 'cpp_zscore'].sort_values('participant')

    model_conf_coef_mean = model_conf_coef['coefficient'].mean()
    model_cpp_coef_mean = model_cpp_coef['coefficient'].mean()
    model_conf_coef_sem = stats.sem(model_conf_coef['coefficient'])
    model_cpp_coef_sem = stats.sem(model_cpp_coef['coefficient'])

    # Splitting data into bins to perform same analysis as in Cedric's paper
    model_uncertainty_df = computational_var_dfs[['cleaned_learning_rate_ideal_observer', 'prior_conf', 'cpp', 'task_session', 'participant']]

    # Splitting by prior confidence
    nbins_uncert = 15 

    _, bin_edges_conf = pd.qcut(model_uncertainty_df["prior_conf"], nbins_uncert, retbins=True)

    bin_midpoints = [(bin_edges_conf[i] + bin_edges_conf[i+1]) / 2 for i in range(len(bin_edges_conf) - 1)]
    model_uncertainty_df["prior_conf_bin"] = pd.cut(model_uncertainty_df["prior_conf"], bins=bin_edges_conf, labels=False, include_lowest=True)
    model_uncertainty_df["prior_conf_bin_label"] = model_uncertainty_df["prior_conf_bin"].apply(lambda x: bin_midpoints[x])

    # Splitting by cpp
    _, bin_edges_cpp = pd.qcut(model_uncertainty_df["cpp"], 2, retbins=True)

    model_uncertainty_df["cpp_bin"] = pd.cut(model_uncertainty_df["cpp"], bins=bin_edges_cpp, labels=False, include_lowest=True)

    # Storing the average learning rate at the participant level
    binned_data =  {"prior_conf_bin": [], "prior_conf_bin_label": [], "cpp_bin": [], "avg_lr": [], "participant": []}

    for participant in model_uncertainty_df["participant"].unique():
        participant_data = model_uncertainty_df[model_uncertainty_df["participant"] == participant]
        
        for prior_conf_bin in sorted(participant_data["prior_conf_bin"].unique()):
            for cpp_bin in sorted(participant_data["cpp_bin"].unique()):
                
                bin_data = participant_data[
                    (participant_data["prior_conf_bin"] == prior_conf_bin) & 
                    (participant_data["cpp_bin"] == cpp_bin)
                ]
                
                avg_lr = bin_data["cleaned_learning_rate_ideal_observer"].mean()
                bin_label = bin_data['prior_conf_bin_label'].unique()[0]  

                binned_data['prior_conf_bin_label'].append(bin_label)
                binned_data["prior_conf_bin"].append(prior_conf_bin)
                binned_data["cpp_bin"].append(cpp_bin)
                binned_data["avg_lr"].append(avg_lr)
                binned_data["participant"].append(participant)

    participant_level_df = pd.DataFrame(binned_data)

    # Splitting the data to compute and store the averaged data across the sample
    cp_0_df = participant_level_df[participant_level_df['cpp_bin'] == 0]
    cp_1_df = participant_level_df[participant_level_df['cpp_bin'] == 1]

    group_avg_data = {"prior_conf_bin": [], "prior_conf_bin_label": [], "cpp_bin": [], "group_avg_lr": []}

    # Process cpp_bin == 0
    for bin in sorted(cp_0_df['prior_conf_bin'].unique()):
        data = cp_0_df[cp_0_df['prior_conf_bin'] == bin]
        learning_rate = data['avg_lr'].mean()
        bin_label = data['prior_conf_bin_label'].unique()[0]  

        group_avg_data["prior_conf_bin"].append(bin)
        group_avg_data["prior_conf_bin_label"].append(bin_label)
        group_avg_data["cpp_bin"].append(0)
        group_avg_data["group_avg_lr"].append(learning_rate)

    # Process cpp_bin == 1
    for bin in sorted(cp_1_df['prior_conf_bin'].unique()):
        data = cp_1_df[cp_1_df['prior_conf_bin'] == bin]
        learning_rate = data['avg_lr'].mean()
        bin_label = data['prior_conf_bin_label'].unique()[0] 

        group_avg_data["prior_conf_bin"].append(bin)
        group_avg_data["prior_conf_bin_label"].append(bin_label)
        group_avg_data["cpp_bin"].append(1)
        group_avg_data["group_avg_lr"].append(learning_rate)

    model_group_avg_df = pd.DataFrame(group_avg_data)

    # Plotting normative model outputs across subplots
    if save_plot:
        fig = make_subplots(
        rows=2, cols=2,  
        specs=[
            [{"rowspan": 2, "colspan": 1}, {"colspan": 1}],  
            [None, {}],  
        ],
        subplot_titles=(
            "Regression weights for prior confidence and change point probability <br> on participants' learning rate derived from the ideal observer model", 
            "Individual regression weights for prior confidence and change point probability <br> on participants' learning rate derived from the ideal observer model", 
            "Predicted relationship between learning rate, prior confidence and change point probability <br> derived from the ideal observer model"),
            )
        
        # Add bars for Prior conf (z-score) and Change point prob (z-score) in the first subplot
        fig.add_trace(go.Bar(
            name='Prior conf (z-score)',
            x=['Prior conf (z-score)'], y=[model_conf_coef_mean],
            error_y=dict(type='data', array=[model_conf_coef_sem]),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name='Change point prob (z-score)',
            x=['Change point prob (z-score)'], y=[model_cpp_coef_mean],
            error_y=dict(type='data', array=[model_cpp_coef_sem]),
            showlegend=False
        ), row=1, col=1)

        # Add a horizontal line at y=0 in the first subplot
        fig.add_hline(y=0, line_dash='dot', row=1, col=1)

        # Update bar appearance for the first subplot
        fig.update_traces(
            marker_color='rgba(0, 0, 225, 0.4)', 
            marker_line_color='rgba(0, 0 , 255, 0.8)', 
            marker_line_width=1.0,
            row=1, col=1
        )

        # Update axes for the first subplot (left)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black',  title_text="Regression Weight", row=1, col=1)

        # Add scatter plot for individual regression weights in the second subplot
        fig.add_trace(go.Scatter(
            x=model_conf_coef['coefficient'], 
            y=model_cpp_coef['coefficient'], 
            mode='markers', 
            marker=dict(color='rgba(0, 0, 255, 0.8)', size=10),
            showlegend=False
        ), row=1, col=2)

        # Add horizontal and vertical lines for scatter plot
        fig.add_hline(y=0, line_dash='dot', row=1, col=2)
        fig.add_vline(x=0, line_dash='dot', row=1, col=2)

        # Update x and y axes for the second subplot (right)
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Prior Confidence (Learning Rate)", 
            range=[-0.9, -0.85],  
            row=1, col=2
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Change Point Probability (Learning Rate)",  
            range=[0.39, 0.5], 
            row=1, col=2
        )

        # Add predicted relationship between learning rate, prior confidence and change point probability in the final subplot
        fig.add_trace(go.Scatter(x=model_group_avg_df[model_group_avg_df['cpp_bin']==0]['prior_conf_bin_label'], y=model_group_avg_df[model_group_avg_df['cpp_bin']==0]['group_avg_lr'], mode='lines+markers', name=f"{bin_edges_cpp[0]:.2f} - {bin_edges_cpp[1]:.2f}"), row=2, col=2)

        fig.add_trace(go.Scatter(x=model_group_avg_df[model_group_avg_df['cpp_bin']==1]['prior_conf_bin_label'], y=model_group_avg_df[model_group_avg_df['cpp_bin']==1]['group_avg_lr'], mode='lines+markers', name=f"{bin_edges_cpp[1]:.2f} - {bin_edges_cpp[2]:.2f}"), row=2, col=2)

        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Prior Confidence (-log(std))", 
            row=2, col=2
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Model-derived learning rate",  
            row=2, col=2
        )

        fig.update_layout(
        legend=dict(
            title="Change Point Probability (bins)", 
            x=0.95,  
            y=0.35,  
            xanchor='center',  
            yanchor='top'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        height=950,
        width=1800)

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"uncertainty-params-on-learning-rate-model-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)


        fig.show()


    return model_uncert_on_lr_df, model_group_avg_df



def sample_behav_uncertainty_on_learning_rate(condition_str, base_dir, excluded_participants, exclude_zero=False, save_plot = True):
    """
     Function to plot the relationships between participants' behaviour-derived learning rare and parameters of uncertainty (i.e. 'change point probability' and 'prior confidence'). 

    Parameters:
    - condition_str: string, denoting the condition strings for filtering the data files.
    - base_dir: Path, base directory for data files.
    - excluded_participants: list, participants to exclude from plots/analysis.
    - exclude_zero: boolean, if true, will remove all rows associated with no update in learning rate (i.e learning rate = 0) (default = True).
    - save_plot: boolean, if True, will save the produce plot(s) in a group level file (default = True)

    Returns:
    - behav_uncert_on_lr_df: pd.DataFrame, contains the regression coefficients for uncertainty parameters on learning rate for all partcipants included in the final sample. 
    - behav_group_conf_lr_df: pd.DataFrame, contains the binned learning rate data across  prior confidence for each participant in the sample.
    - behav_group_conf_cpp_df: pd.DataFrame, contains the binned learning rate data across  prior confidence and change point probability periods for each participant in the sample.
    """
    computational_var_dfs = pd.DataFrame()
    behav_uncert_on_lr_df = pd.DataFrame()

    base_path = Path(base_dir)

    for sub_folder in base_path.iterdir():
        if sub_folder.is_dir() and sub_folder.name not in excluded_participants:
            print(f"Processing folder: {sub_folder.name}")  
            
            # Construct the path to the csv_files folder
            csv_files_path = Path(sub_folder / f'ses-{condition_str}' / 'csv_files')
            
            if csv_files_path.exists():
                print(f"CSV files path exists: {csv_files_path}")  
                
                # Correct the csv file name by using the participant ID correctly
                participant_id = sub_folder.name.split('-')[1]  # Extract participant ID (xx from 'sub-xx')

                if exclude_zero:
                    csv_file_model_uncert = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_uncertainty-v-lr-df-zero-excluded.csv'
                    fig_title = "-zero-excluded"
                    plot_title = "(zero updates excluded)"
                else:
                    csv_file_model_uncert = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_uncertainty-v-lr-df.csv'
                    fig_title = ""
                    plot_title = ""
                
                csv_file_path_model_uncert = csv_files_path / csv_file_model_uncert
                
                if csv_file_path_model_uncert.exists():
                    print(f"CSV file path exists: {csv_file_path_model_uncert}")  
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path_model_uncert)
                    print(f"Data from {csv_file_model_uncert}:\n{csv_df.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    behav_uncert_on_lr_df = pd.concat([behav_uncert_on_lr_df, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_model_uncert}") 


                csv_file_com_df = f'sub-{participant_id}_ses-{condition_str}_task-Adalearn-Pupil_computational-var-df.csv'
                
                csv_file_path_com_df = csv_files_path / csv_file_com_df
                
                if csv_file_path_com_df.exists():
                    print(f"CSV file path exists: {csv_file_com_df}")  
                    
                    # Read the CSV data
                    csv_df = pd.read_csv(csv_file_path_com_df)
                    print(f"Data from {csv_file_com_df}:\n{csv_df.head()}")  
                    
                    # Add the participant ID to the DataFrame
                    csv_df['participant'] = participant_id

                    # Concatenate to the main DataFrame
                    computational_var_dfs = pd.concat([computational_var_dfs, csv_df], ignore_index=True)
                    
                else:
                    print(f"CSV file not found: {csv_file_path_com_df}")  
            else:
                print(f"CSV folder not found: {csv_files_path}") 


    behav_conf_coef = behav_uncert_on_lr_df[behav_uncert_on_lr_df['variable'] == 'conf_zscore'].sort_values('participant')
    behav_cpp_coef = behav_uncert_on_lr_df[behav_uncert_on_lr_df['variable'] == 'cpp_zscore'].sort_values('participant')

    behav_conf_coef_mean = behav_conf_coef['coefficient'].mean()
    behav_cpp_coef_mean = behav_cpp_coef['coefficient'].mean()
    behav_conf_coef_sem = stats.sem(behav_conf_coef['coefficient'])
    behav_cpp_coef_sem = stats.sem(behav_cpp_coef['coefficient'])

    # Splitting data into bins to perform the same analysis as in Cedric's paper
    behav_uncertainty_df = computational_var_dfs[['cleaned_learning_rate_estimate', 'prior_conf', 'cpp', 'task_session', 'participant']]

    if exclude_zero:
        beginning_length = len(behav_uncertainty_df)
        behav_uncertainty_df = behav_uncertainty_df[behav_uncertainty_df['cleaned_learning_rate_estimate'] != 0]
        end_length = len(behav_uncertainty_df)
        print(f'Number of rows removed due to zero learning rate = {beginning_length - end_length}')

    # Splitting by prior confidence (defining the number of bins)
    nbins_uncert_lr = 6 
    
    # Evaluating the relationship between learning rate and prior confidence based on behavioural responses
    _, bin_edges_conf_lr = pd.qcut(behav_uncertainty_df["prior_conf"], nbins_uncert_lr, retbins=True)

    bin_midpoints = [(bin_edges_conf_lr[i] + bin_edges_conf_lr[i+1]) / 2 for i in range(len(bin_edges_conf_lr) - 1)]
    behav_uncertainty_df["prior_conf_bin"] = pd.cut(behav_uncertainty_df["prior_conf"], bins=bin_edges_conf_lr, labels=False, include_lowest=True)
    behav_uncertainty_df["prior_conf_bin_label"] = behav_uncertainty_df["prior_conf_bin"].apply(lambda x: bin_midpoints[x])

    # Storing the average learning rate at the participant level
    binned_data =  {"prior_conf_bin": [], "prior_conf_bin_label": [], "avg_lr": [], "participant": []}

    for participant in behav_uncertainty_df["participant"].unique():
        participant_data = behav_uncertainty_df[behav_uncertainty_df["participant"] == participant]
        
        for prior_conf_bin in sorted(participant_data["prior_conf_bin"].unique()):
                
            bin_data = participant_data[
                (participant_data["prior_conf_bin"] == prior_conf_bin)]
            
            avg_lr = bin_data["cleaned_learning_rate_estimate"].mean()
            bin_label = bin_data['prior_conf_bin_label'].unique()[0]  

            binned_data['prior_conf_bin_label'].append(bin_label)
            binned_data["prior_conf_bin"].append(prior_conf_bin)
            binned_data["avg_lr"].append(avg_lr)
            binned_data["participant"].append(participant)

    participant_conf_lr_df = pd.DataFrame(binned_data)

    group_avg_data = {"prior_conf_bin": [], "prior_conf_bin_label": [], "group_avg_lr": [], "group_sem_lr": []}

    for bin in sorted(participant_conf_lr_df['prior_conf_bin'].unique()):
        data = participant_conf_lr_df[participant_conf_lr_df['prior_conf_bin'] == bin]
        learning_rate = data['avg_lr'].mean()
        lr_sem = stats.sem(data['avg_lr'])
        bin_label = data['prior_conf_bin_label'].unique()[0] 

        group_avg_data["prior_conf_bin"].append(bin)
        group_avg_data["prior_conf_bin_label"].append(bin_label)
        group_avg_data["group_avg_lr"].append(learning_rate)
        group_avg_data["group_sem_lr"].append(lr_sem)

    behav_group_conf_lr_df = pd.DataFrame(group_avg_data)

    # Evaluating the relationship between learning rate, prior confidence and change point probability based on behavioural responses
    nbins_uncert_cpp_lr = nbins_uncert_lr // 2

    _, bin_edges_conf_cpp = pd.qcut(behav_uncertainty_df["prior_conf"], nbins_uncert_cpp_lr, retbins=True)

    bin_midpoints = [(bin_edges_conf_cpp[i] + bin_edges_conf_cpp[i+1]) / 2 for i in range(len(bin_edges_conf_cpp) - 1)]
    behav_uncertainty_df["prior_conf_bin"] = pd.cut(behav_uncertainty_df["prior_conf"], bins=bin_edges_conf_cpp, labels=False, include_lowest=True)
    behav_uncertainty_df["prior_conf_bin_label"] = behav_uncertainty_df["prior_conf_bin"].apply(lambda x: bin_midpoints[x])

    # Splitting by cpp
    _, bin_edges_cpp = pd.qcut(behav_uncertainty_df["cpp"], 2, retbins=True)

    behav_uncertainty_df["cpp_bin"] = pd.cut(behav_uncertainty_df["cpp"], bins=bin_edges_cpp, labels=False, include_lowest=True)

    # Storing the average learning rate at the participant level
    binned_data =  {"prior_conf_bin": [], "prior_conf_bin_label": [], "cpp_bin": [], "avg_lr": [], "participant": []}

    for participant in behav_uncertainty_df["participant"].unique():
        participant_data = behav_uncertainty_df[behav_uncertainty_df["participant"] == participant]
        
        for prior_conf_bin in sorted(participant_data["prior_conf_bin"].unique()):
            for cpp_bin in sorted(participant_data["cpp_bin"].unique()):
                
                bin_data = participant_data[
                    (participant_data["prior_conf_bin"] == prior_conf_bin) & 
                    (participant_data["cpp_bin"] == cpp_bin)
                ]
                
                avg_lr = bin_data["cleaned_learning_rate_estimate"].mean()
                bin_label = bin_data['prior_conf_bin_label'].unique()[0]  

                binned_data['prior_conf_bin_label'].append(bin_label)
                binned_data["prior_conf_bin"].append(prior_conf_bin)
                binned_data["cpp_bin"].append(cpp_bin)
                binned_data["avg_lr"].append(avg_lr)
                binned_data["participant"].append(participant)

    participant_level_conf_cpp_df = pd.DataFrame(binned_data)

    # Splitting the data to compute and store the averaged data across the sample
    cp_0_df = participant_level_conf_cpp_df[participant_level_conf_cpp_df['cpp_bin'] == 0]
    cp_1_df = participant_level_conf_cpp_df[participant_level_conf_cpp_df['cpp_bin'] == 1]

    group_avg_data = {"prior_conf_bin": [], "prior_conf_bin_label": [], "cpp_bin": [], "group_avg_lr": [], "group_sem_lr": []}

    # Process cpp_bin == 0
    for bin in sorted(cp_0_df['prior_conf_bin'].unique()):
        data = cp_0_df[cp_0_df['prior_conf_bin'] == bin]
        learning_rate = data['avg_lr'].mean()
        lr_sem = stats.sem(data['avg_lr'])
        bin_label = data['prior_conf_bin_label'].unique()[0]  

        group_avg_data["prior_conf_bin"].append(bin)
        group_avg_data["prior_conf_bin_label"].append(bin_label)
        group_avg_data["cpp_bin"].append(0)
        group_avg_data["group_avg_lr"].append(learning_rate)
        group_avg_data["group_sem_lr"].append(lr_sem)

    # Process cpp_bin == 1
    for bin in sorted(cp_1_df['prior_conf_bin'].unique()):
        data = cp_1_df[cp_1_df['prior_conf_bin'] == bin]
        learning_rate = data['avg_lr'].mean()
        lr_sem = stats.sem(data['avg_lr'])
        bin_label = data['prior_conf_bin_label'].unique()[0] 

        group_avg_data["prior_conf_bin"].append(bin)
        group_avg_data["prior_conf_bin_label"].append(bin_label)
        group_avg_data["cpp_bin"].append(1)
        group_avg_data["group_avg_lr"].append(learning_rate)
        group_avg_data["group_sem_lr"].append(lr_sem)

    behav_group_conf_cpp_df = pd.DataFrame(group_avg_data)

    # Plotting normative model outputs across subplots
    if save_plot:
        fig = make_subplots(
        rows=2, cols=2,  
        subplot_titles=(
            f"Regression weights for prior confidence and change point probability <br> on participants' learning rate derived from participants' estimates {plot_title}", 
            f"Relationship between learning rate and prior confidence <br> derived from participants' estimates {plot_title}",
            f"Individual regression weights for prior confidence and change point probability <br> on participants' learning rate derived from participants' estimates {plot_title}",
            f"Relationship between learning rate, prior confidence and change point probability <br> derived from participants' estimates {plot_title}"),
            )
        
        # Add bars for Prior conf (z-score) and Change point prob (z-score) in the first subplot
        fig.add_trace(go.Bar(
            name='Prior conf (z-score)',
            x=['Prior conf (z-score)'], y=[behav_conf_coef_mean],
            error_y=dict(type='data', array=[behav_conf_coef_sem]),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name='Change point prob (z-score)',
            x=['Change point prob (z-score)'], y=[behav_cpp_coef_mean],
            error_y=dict(type='data', array=[behav_cpp_coef_sem]),
            showlegend=False
        ), row=1, col=1)

        # Add a horizontal line at y=0 in the first subplot
        fig.add_hline(y=0, line_dash='dot', row=1, col=1)

        # Update bar appearance for the first subplot
        fig.update_traces(
            marker_color='rgba(0, 0, 225, 0.4)', 
            marker_line_color='rgba(0, 0 , 255, 0.8)', 
            marker_line_width=1.0,
            row=1, col=1
        )

        # Update axes for the first subplot (left)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black',  title_text="Regression Weight", row=1, col=1)

        # Add scatter plot for individual regression weights in the second subplot
        fig.add_trace(go.Scatter(
            x=behav_conf_coef['coefficient'], 
            y=behav_cpp_coef['coefficient'], 
            mode='markers', 
            marker=dict(color='rgba(0, 0, 255, 0.8)', size=10),
            showlegend=False
        ), row=2, col=1)

        # Add horizontal and vertical lines for scatter plot
        fig.add_hline(y=0, line_dash='dot', row=2, col=1)
        fig.add_vline(x=0, line_dash='dot', row=2, col=1)

        # Update x and y axes for the second subplot (right)
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Prior Confidence (Learning Rate)", 
            range=[-0.6, 0.6],  
            row=2, col=1
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Change Point Probability (Learning Rate)",  
            range=[-0.6, 0.6], 
            row=2, col=1
        )

        # Add relationship between participants learning rate and prior confidence (binned data) to third subplot
        fig.add_trace(go.Scatter(
            x=behav_group_conf_lr_df['prior_conf_bin_label'],
            y=behav_group_conf_lr_df['group_avg_lr'], 
            mode='lines+markers', 
            showlegend=False, 
            line = dict(color = 'teal'),
            error_y=dict(
                type='data',
                array=behav_group_conf_lr_df['group_sem_lr'],
                visible=True ),
            ), row=1, col=2)
        
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Prior Confidence (-log(sd))", 
            row=1, col=2
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Learning rate",   
            row=1, col=2
        )


        # Add relationship between learning rate, prior confidence and change point probability in the final subplot
        fig.add_trace(go.Scatter(
            x=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==0]['prior_conf_bin_label'],
            y=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==0]['group_avg_lr'],
            mode='lines+markers',
            name=f"{bin_edges_cpp[0]:.2f} - {bin_edges_cpp[1]:.2f}",
            line = dict(color='purple'),
            error_y=dict(
                type='data',
                array=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==0]['group_sem_lr'],
                visible=True ),
            ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==1]['prior_conf_bin_label'],
            y=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==1]['group_avg_lr'],
            mode='lines+markers',
            name=f"{bin_edges_cpp[1]:.2f} - {bin_edges_cpp[2]:.2f}",
            line = dict(color='orange'),
            error_y=dict(
                type='data',
                array=behav_group_conf_cpp_df[behav_group_conf_cpp_df['cpp_bin']==1]['group_sem_lr'],
                visible=True ),
            ), row=2, col=2)

        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Prior Confidence (-log(std))", 
            row=2, col=2
        )

        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            title_text="Learning rate",  
            row=2, col=2
        )

        fig.update_layout(
        legend=dict(
            title="Change Point Probability (bins)", 
            x=0.95,  
            y=0.35,  
            xanchor='center',  
            yanchor='top'
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        height=950,
        width=1800)

        # Save the figure to the group level plots folder
        plot_dir = os.path.join(base_dir, "group_level_plots")
        os.makedirs(plot_dir, exist_ok=True)
        file_name = f"uncertainty-params-on-learning-rate{fig_title}-(Adalearn-Pupil).png"
        file_path = os.path.join(plot_dir, file_name)
        fig.write_image(file_path)


        
        fig.show()


    return  behav_uncert_on_lr_df, behav_group_conf_lr_df, behav_group_conf_cpp_df