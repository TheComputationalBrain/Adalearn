"""

"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, save_sample_dataframes_to_csv 

from adalearn_analysis_participant_level_functions_behav import load_behavioural_data, behavioural_change_point_analysis, behavioural_performance_analysis

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, task_sessions



participant_list = get_participant_numbers(base_dir)

group_behavioural_df_list = []
group_mean_error_df_list = []
group_change_point_indices_list = []

# For loop, extracting data from behavioural files for each participant within data list
for participant in participant_list:
    participant_str = str(participant).zfill(2)
    print(f"\n\n\nProcessing participant {participant_str}...")
    directory_path = Path(f'{base_dir}/sub-{participant_str}')

    print(f"\nLoading task data and computing task performance...")

    behavioural_df, _ = load_behavioural_data( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

    behavioural_df['participant'] = participant_str

    group_behavioural_df_list.append(behavioural_df)

    mean_abs_error_df = behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_performance=False)

    mean_abs_error_df['coef_variation'] = mean_abs_error_df['std_abs_error']/ mean_abs_error_df['mean_abs_error']

    mean_abs_error_df['participant'] = participant_str

    group_mean_error_df_list.append(mean_abs_error_df)

    print(f"\nExtracting change point indices...")

    num_change_points, change_points_per_session, change_point_indices, behavioural_df = behavioural_change_point_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_histogram_zscores=False)

    change_point_indices['participant'] = participant_str

    group_change_point_indices_list.append(change_point_indices)


# Concatenate the dataframes in the lists into a single group-level dataframe
group_level_behavioural_df = pd.concat(group_behavioural_df_list, ignore_index=True)
group_level_mean_error_df = pd.concat(group_mean_error_df_list, ignore_index=True)
group_level_change_point_indices_df = pd.concat(group_change_point_indices_list, ignore_index=True)

# Running one-way ANOVA on performance metrics
anova_mean_error = AnovaRM(data=group_level_mean_error_df, depvar='mean_abs_error', subject='participant', within=['task_session']).fit()

print('Outcome of one-way repeated measures ANOVA on mean absolute error...')
print(f'\n{anova_mean_error}')

anova_mean_cv = AnovaRM(data=group_level_mean_error_df, depvar='coef_variation', subject='participant', within=['task_session']).fit()

print('Outcome of one-way repeated measures ANOVA on mean coefficient of variation...')
print(f'\n{anova_mean_cv}')

# Saving group level data to .csv files
save_sample_dataframes_to_csv(dataframes={
    "sample_level_analysis_behavioural_df": group_level_behavioural_df,
    "sample_level_analysis_mean_abs_error_df": group_level_mean_error_df,
    "sample_level_analysis_change_point_indices_df": group_level_change_point_indices_df
    }, 
directory_path=base_dir,
condition_str=condition_str,
task='Pupil')