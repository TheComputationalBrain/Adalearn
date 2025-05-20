"""

"""
import pandas as pd
import os
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_regression_pupil import pupil_peristimulus_regression_analysis, pupil_on_learning_rate_regression_analysis, pupil_uncertainty_lr_mediation_analysis

from adalearn_analysis_participant_level_functions_pupil import pupil_response_around_change_point

from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, process_epoched_df, load_sample_csvs, save_sample_dataframes_to_csv 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str



dataframes = load_sample_csvs(directory_path=base_dir, substrings=['computational_var_df', 'change_point_indices_df', 'exclude_participants_pupil_analysis_df', 'group_level_outcome_epochs_descriptive_stats_df',
    'group_level_outcome_epoched_pupil_df','group_level_outcome_epochs_columns_df',
    'group_level_outcome_epochs_to_remove_df',
    'group_level_outcome_epochs_descriptive_stats_nobc_df', 
    'group_level_outcome_epoched_pupil_nobc_df'
])

participant_list = get_participant_numbers(directory=base_dir)

excluded_participants = list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])

pupil_regress_coef_df_list = []
pupil_regress_coef_nobc_df_list = []

# Running for all participants then excluding those identified due to excessive interpolation 
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{base_dir}/sub-{participant_str}')
    else:
        continue

    # Getting individual participant data
    comp_df = dataframes['computational_var_df'][dataframes['computational_var_df']['participant']==participant].copy()
    comp_df = comp_df.reset_index()

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant']==participant].copy()

    epoch_data = dataframes['group_level_outcome_epoched_pupil_df'][dataframes['group_level_outcome_epoched_pupil_df']['participant'] == participant]

    epoch_nobc_data = dataframes['group_level_outcome_epoched_pupil_nobc_df'][dataframes['group_level_outcome_epoched_pupil_nobc_df']['participant'] == participant]

    epoch_descriptive_stats = dataframes['group_level_outcome_epochs_descriptive_stats_df'][dataframes['group_level_outcome_epochs_descriptive_stats_df']['participant']==participant]

    epoch_nobc_descriptive_stats = dataframes['group_level_outcome_epochs_descriptive_stats_nobc_df'][dataframes['group_level_outcome_epochs_descriptive_stats_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['group_level_outcome_epochs_columns_df'][dataframes['group_level_outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['group_level_outcome_epochs_to_remove_df'][dataframes['group_level_outcome_epochs_to_remove_df']['participant']==participant]
    
    # Processing epoched dataframes
    column_titles = list(analysed_epochs['columns'].values.astype(int))

    if len(column_titles)+len(epochs_to_remove) == len(comp_df):
        print(f'correct number of epochs imported for participant {participant_str}')
    else:
        print(f'incorrect number of epochs imported for participant {participant_str}')

    epoch_data = epoch_data.drop(columns=['participant'])
    epoch_data = epoch_data.T
    epoch_data.columns = column_titles
    epoch_data['mean'] = epoch_descriptive_stats['mean'].values.astype(float)
    epoch_data['std'] = epoch_descriptive_stats['std'].values.astype(float)
    epoch_data['sem'] = epoch_descriptive_stats['sem'].values.astype(float)
    epoch_data['time'] = epoch_descriptive_stats['time'].values.astype(float)


    epoch_nobc_data = epoch_nobc_data.drop(columns=['participant'])
    epoch_nobc_data = epoch_nobc_data.T
    epoch_nobc_data.columns = column_titles
    epoch_nobc_data['mean'] = epoch_nobc_descriptive_stats['mean'].values.astype(float)
    epoch_nobc_data['std'] = epoch_nobc_descriptive_stats['std'].values.astype(float)
    epoch_nobc_data['sem'] = epoch_nobc_descriptive_stats['sem'].values.astype(float)
    epoch_nobc_data['time'] = epoch_nobc_descriptive_stats['time'].values.astype(float)

    # Running peristimulus regression analysis
    print(f"\nRunning regression on baseline corrected pupil signal...")   
    conf_cpp_coef, pupil_regress_df, residuals_df = pupil_peristimulus_regression_analysis(participant_str=participant, condition_str=condition_str, directory_path=directory_path, computational_var_df=comp_df, outcome_epoched_pupil_df=epoch_data, outcome_epochs_to_remove=epochs_to_remove, variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=True,  plot_regression_analysis=False)

    conf_cpp_coef['participant'] = participant
    conf_cpp_coef['count'] = range(len(conf_cpp_coef))

    pupil_regress_coef_df_list.append(conf_cpp_coef)
   
    print(f"\nRunning regression on non-baseline corrected pupil signal...")
    conf_cpp_coef_nobc, pupil_regress_nobc_df, residuals_nobc_df = pupil_peristimulus_regression_analysis(participant_str=participant, condition_str=condition_str, directory_path=directory_path, computational_var_df=comp_df, outcome_epoched_pupil_df=epoch_nobc_data, outcome_epochs_to_remove=epochs_to_remove, variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior_dist'], baseline_corrected=True,  plot_regression_analysis=False)

    conf_cpp_coef_nobc['participant'] = participant
    conf_cpp_coef_nobc['count'] = range(len(conf_cpp_coef_nobc))

    pupil_regress_coef_nobc_df_list.append(conf_cpp_coef_nobc) 

# Concatenating dataframes to group_level
pupil_regress_coef_df = pd.concat(pupil_regress_coef_df_list, ignore_index=True)
pupil_regress_coef_nobc_df = pd.concat(pupil_regress_coef_nobc_df_list, ignore_index=True)

# Creating a dataframe contain the averaged response across time points within the epoch
print(f'\n Creating dataframe containing the averaged coefficients for the baseline corrected pupil signal')
coefs_dict = {'cpp': [], 'cpp_error': [], 'conf': [], 'conf_error': [], 'time': []}

for count in pupil_regress_coef_df['count'].unique():
    data = pupil_regress_coef_df[pupil_regress_coef_df['count'] == count].sort_values(by='participant')
    mean_cpp = data['cpp'].mean()
    sem_cpp = stats.sem(data['cpp'])
    mean_conf = data['prior_conf'].mean()
    sem_conf = stats.sem(data['prior_conf'])
    mean_time = data['time'].mean()

    coefs_dict['cpp'].append(mean_cpp)
    coefs_dict['cpp_error'].append(sem_cpp)
    coefs_dict['conf'].append(mean_conf)
    coefs_dict['conf_error'].append(sem_conf)
    coefs_dict['time'].append(mean_time)

coefs_df = pd.DataFrame(coefs_dict)

print(f'\n Creating dataframe containing the averaged coefficients for the baseline corrected pupil signal')
coefs_nobc_dict = {'cpp': [], 'cpp_error': [], 'conf': [], 'conf_error': [], 'time': []}

for count in pupil_regress_coef_nobc_df['count'].unique():
    data = pupil_regress_coef_nobc_df[pupil_regress_coef_nobc_df['count'] == count].sort_values(by='participant')
    mean_cpp = data['cpp'].mean()
    sem_cpp = stats.sem(data['cpp'])
    mean_conf = data['prior_conf'].mean()
    sem_conf = stats.sem(data['prior_conf'])
    mean_time = data['time'].mean()

    coefs_nobc_dict['cpp'].append(mean_cpp)
    coefs_nobc_dict['cpp_error'].append(sem_cpp)
    coefs_nobc_dict['conf'].append(mean_conf)
    coefs_nobc_dict['conf_error'].append(sem_conf)
    coefs_nobc_dict['time'].append(mean_time)

coefs_nobc_df = pd.DataFrame(coefs_nobc_dict)

# Saving dataframes
save_sample_dataframes_to_csv(dataframes={
    'pupil_regress_coef_df': pupil_regress_coef_df,
    'pupil_regress_coef_nobc_df': pupil_regress_coef_nobc_df,
    'coefs_df': coefs_df,
    'coefs_nobc_df': coefs_nobc_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil') 