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


from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, load_sample_csvs, save_sample_dataframes_to_csv 

from adalearn_analysis_group_level_functions_pupil_analysis import extract_cluster_times, pupil_response_around_change_point, pupil_on_learning_rate_regression_analysis, pupil_uncertainty_lr_mediation_analysis

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str



dataframes = load_sample_csvs(directory_path=base_dir, substrings=['computational_var_df', 'change_point_indices_df', 'exclude_participants_pupil_analysis_df', 'outcome_epochs_descriptive_stats_df',
    'outcome_epoched_pupil_df','outcome_epochs_columns_df',
    'outcome_epochs_to_remove_df',
    'outcome_epochs_descriptive_stats_nobc_df', 
    'outcome_epoched_pupil_nobc_df', 'cluster_permutation_analysis', 'cluster_permutation_analysis_nobc'
])

participant_list = get_participant_numbers(directory=base_dir)

excluded_participants = list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])

pupil_v_lr_regress_coef_df_list = []
pupil_bc_around_cp_list = []
pupil_nobc_around_cp_list = []

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

    epoch_data = dataframes['outcome_epoched_pupil_df'][dataframes['outcome_epoched_pupil_df']['participant'] == participant]

    epoch_nobc_data = dataframes['outcome_epoched_pupil_nobc_df'][dataframes['outcome_epoched_pupil_nobc_df']['participant'] == participant]

    epoch_descriptive_stats = dataframes['outcome_epochs_descriptive_stats_df'][dataframes['outcome_epochs_descriptive_stats_df']['participant']==participant]

    epoch_nobc_descriptive_stats = dataframes['outcome_epochs_descriptive_stats_nobc_df'][dataframes['outcome_epochs_descriptive_stats_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['outcome_epochs_columns_df'][dataframes['outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['outcome_epochs_to_remove_df'][dataframes['outcome_epochs_to_remove_df']['participant']==participant]
    
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

    # Extracting cluster times to set window for regression and mediation analyses
    baseline_cor_start_time, baseline_cor_end_time = extract_cluster_times(
        dataframes['cluster_permutation_analysis'],
        epoch_data['time']
    )

    nobc_start_time, nobc_end_time = extract_cluster_times(
        dataframes['cluster_permutation_analysis_nobc'],
        epoch_nobc_data['time']
    )

    # Examining pupil responses around change points 
    pupil_bc_effect_around_cp_df = pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=epoch_data, change_point_indices=change_point_indices, baseline_correct_window=[baseline_cor_start_time, baseline_cor_end_time], nobc_window=[nobc_start_time, nobc_end_time], before=5, after=5, estimate_type='baseline_corrected_effect', plot_pupil_around_change_point=False)

    pupil_bc_effect_around_cp_df['participant'] = participant_str

    pupil_nobc_effect_around_cp_df = pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=epoch_nobc_data, change_point_indices=change_point_indices, baseline_correct_window=[baseline_cor_start_time, baseline_cor_end_time], nobc_window=[nobc_start_time, nobc_end_time], before=5, after=5, estimate_type='nonbaseline_corrected_effect', plot_pupil_around_change_point=False)

    pupil_nobc_effect_around_cp_df['participant'] = participant_str

    pupil_bc_around_cp_list.append(pupil_bc_effect_around_cp_df)
    pupil_nobc_around_cp_list.append(pupil_nobc_effect_around_cp_df)

    # Running regression analysis for pupil metrics on learning rate
    pupil_v_lr_regress_df = pupil_on_learning_rate_regression_analysis(directory_path=directory_path,participant_str=participant_str, condition_str=condition_str, computational_var_df= comp_df, outcome_epoched_pupil_df=epoch_data, outcome_epoched_pupil_df_nobc=epoch_nobc_data, outcome_epochs_to_remove=epochs_to_remove, baseline_correct_window=[baseline_cor_start_time, baseline_cor_end_time], nobc_window=[nobc_start_time, nobc_end_time],parameter_estimate='estimate', exclude_zero=False, plot_pupil_lr_regress=False)

    pupil_v_lr_regress_df['participant'] = participant_str

    pupil_v_lr_regress_coef_df_list.append(pupil_v_lr_regress_df)


# Collating dataframes
group_pupil_v_lr_regress_df = pd.concat(pupil_v_lr_regress_coef_df_list, ignore_index=True)
group_pupil_bc_around_cp_df = pd.concat(pupil_bc_around_cp_list, ignore_index=True)
group_pupil_nobc_around_cp_df = pd.concat(pupil_nobc_around_cp_list, ignore_index=True)

# Running t-tests on pupil on learning rate regression 
pupil_v_lr_bc_coef = group_pupil_v_lr_regress_df[group_pupil_v_lr_regress_df['variable'] == 'pupil_response_bc_zscore'].sort_values('participant').copy()
pupil_v_lr_nobc_coef = group_pupil_v_lr_regress_df[group_pupil_v_lr_regress_df['variable'] == 'pupil_response_nobc_zscore'].sort_values('participant').copy()

pupil_v_lr_bc_coef_mean = pupil_v_lr_bc_coef['coefficient'].mean()
pupil_v_lr_nobc_coef_mean = pupil_v_lr_nobc_coef['coefficient'].mean()
pupil_v_lr_bc_coef_sem = stats.sem(pupil_v_lr_bc_coef['coefficient'])
behav_v_lr_nobc_coef_sem = stats.sem(pupil_v_lr_nobc_coef['coefficient'])

# t-tests against zero and between coefficients
t_bc_pupil = stats.ttest_1samp(pupil_v_lr_bc_coef['coefficient'], 0)
t_nobc_pupil = stats.ttest_1samp(pupil_v_lr_nobc_coef['coefficient'], 0)
t_between_coef = stats.ttest_rel(pupil_v_lr_bc_coef['coefficient'], pupil_v_lr_nobc_coef['coefficient'], alternative='two-sided')

pupil_v_lr_ttest_df = pd.DataFrame({
    'variable': ['pupil_bc', 'pupil_nobc', 'between_pupil_coef'],
    'regression_coef': [pupil_v_lr_bc_coef_mean, pupil_v_lr_nobc_coef_mean, None],
    'regress_coef_sem': [pupil_v_lr_bc_coef_sem, behav_v_lr_nobc_coef_sem, None],
    't_statistic': [t_bc_pupil.statistic, t_nobc_pupil.statistic, t_between_coef.statistic],
    'p_value': [t_bc_pupil.pvalue, t_nobc_pupil.pvalue, t_between_coef.pvalue]
})

# Saving dataframes
save_sample_dataframes_to_csv(dataframes={
    'pupil_v_lr_regress_coef_df': group_pupil_v_lr_regress_df,
    'pupil_v_lr_coef_ttest_df': pupil_v_lr_ttest_df,
    'pupil_bc_around_cp': group_pupil_bc_around_cp_df,
    'pupil_nobc_around_cp': group_pupil_nobc_around_cp_df,
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil')