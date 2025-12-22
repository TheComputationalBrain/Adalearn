"""
This script performs group-level analysis of the peristimulus epoched pupil signals in the Pupil task of the Adalearn project. Using multiple linear rgeression, it computes the coefficients for the prior confidence and change point probability (cpp) variables, both with and without baseline correction (for phasic- and tonic-associated pupil responses, respectively). The script also performs cluster-based permutation tests to identify significant clusters in the regression coefficients across participants. The results are saved in .csv files for further analysis. The following script need to be run before this script to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_model_parameters.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_assessment.py
"""
import pandas as pd
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_group_level_functions_statistical_analysis as group_stats_analysis
import adalearn_analysis_participant_level_functions_pupil as pupil
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'exclude_participants_pupil_analysis_df', 
        'computational_var_df', 
        'change_point_indices_df', 
        'outcome_epochs_time_df',
        'outcome_epoched_pupil_df',
        'outcome_epochs_columns_df',
        'outcome_epochs_to_remove_df',
        'outcome_epochs_time_nobc_df', 
        'outcome_epoched_pupil_nobc_df',
        'outcome_epochs_euclidean_distance_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(directory=Path(f'{BASE_DIR}/data'))

excluded_participants = list(set(
    list(dataframes['exclude_participants_behav_cor_df']['participant']) +
    list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])
))

pupil_regress_coef_df_list = []
pupil_regress_coef_nobc_df_list = []

# Running for all participants then excluding those identified due to excessive interpolation 
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{BASE_DIR}/data/sub-{participant_str}')
    else:
        continue

    # Getting individual participant data
    comp_df = dataframes['computational_var_df'][dataframes['computational_var_df']['participant']==participant].copy()
    comp_df = comp_df.reset_index()

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant']==participant].copy()

    epoch_data = dataframes['outcome_epoched_pupil_df'][dataframes['outcome_epoched_pupil_df']['participant'] == participant]

    epoch_nobc_data = dataframes['outcome_epoched_pupil_nobc_df'][dataframes['outcome_epoched_pupil_nobc_df']['participant'] == participant]

    epoch_times = dataframes['outcome_epochs_time_df'][dataframes['outcome_epochs_time_df']['participant']==participant]

    epoch_nobc_times = dataframes['outcome_epochs_time_nobc_df'][dataframes['outcome_epochs_time_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['outcome_epochs_columns_df'][dataframes['outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['outcome_epochs_to_remove_df'][dataframes['outcome_epochs_to_remove_df']['participant']==participant]

    epoch_positions = dataframes['outcome_epochs_euclidean_distance_df'][dataframes['outcome_epochs_euclidean_distance_df']['participant']==participant]
    
    # Processing epoched dataframes
    column_titles = list(analysed_epochs['columns'].values.astype(int))

    if len(column_titles)+len(epochs_to_remove) == len(comp_df):
        print(f'correct number of epochs imported for participant {participant_str}')
    else:
        print(f'incorrect number of epochs imported for participant {participant_str}')

    epoch_data = epoch_data.drop(columns=['participant'])
    epoch_data = epoch_data.T
    epoch_data.columns = column_titles
    epoch_data['time'] = epoch_times['time'].values.astype(float)


    epoch_nobc_data = epoch_nobc_data.drop(columns=['participant'])
    epoch_nobc_data = epoch_nobc_data.T
    epoch_nobc_data.columns = column_titles
    epoch_nobc_data['time'] = epoch_nobc_times['time'].values.astype(float)

    epoch_positions = epoch_positions.drop(columns=['participant'])
    epoch_positions = epoch_positions.T
    epoch_positions.columns = column_titles
    epoch_positions['time'] = epoch_times['time'].values.astype(float)

    # Running peristimulus regression analysis
    print(f"\nRunning regression on baseline corrected pupil signal...")   
    conf_cpp_coef, pupil_regress_df, residuals_df = pupil.pupil_peristimulus_regression_analysis(participant_str=participant, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=comp_df, outcome_epoched_pupil_df=epoch_data, outcome_epochs_to_remove=epochs_to_remove, variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior'], position_df=epoch_positions, baseline_corrected=True,  plot_regression_analysis=False)

    conf_cpp_coef['participant'] = participant
    conf_cpp_coef['count'] = range(len(conf_cpp_coef))

    pupil_regress_coef_df_list.append(conf_cpp_coef)
   
    print(f"\nRunning regression on non-baseline corrected pupil signal...")
    conf_cpp_coef_nobc, pupil_regress_nobc_df, residuals_nobc_df = pupil.pupil_peristimulus_regression_analysis(participant_str=participant, condition_str=CONDITION_STR, directory_path=directory_path, computational_var_df=comp_df, outcome_epoched_pupil_df=epoch_nobc_data, outcome_epochs_to_remove=epochs_to_remove, variables=['prior_conf', 'cpp'], covariates = ['outcome', 'outcome_num', 'task_session', 'prior'], position_df=epoch_positions, baseline_corrected=False,  plot_regression_analysis=False)

    conf_cpp_coef_nobc['participant'] = participant
    conf_cpp_coef_nobc['count'] = range(len(conf_cpp_coef_nobc))

    pupil_regress_coef_nobc_df_list.append(conf_cpp_coef_nobc) 

# Concatenating dataframes to group_level
pupil_regress_coef_df = pd.concat(pupil_regress_coef_df_list, ignore_index=True)
pupil_regress_coef_nobc_df = pd.concat(pupil_regress_coef_nobc_df_list, ignore_index=True)

# Running cluster permutation analysis on signals
cluster_based_permutation_results, significant_clusters_df = group_stats_analysis.compute_group_level_permutation_cluster_test(group_level_df=pupil_regress_coef_df, vars = ['prior_conf', 'cpp'], alpha=0.05)

cluster_based_permutation_results_nobc, significant_clusters_nobc_df = group_stats_analysis.compute_group_level_permutation_cluster_test(group_level_df=pupil_regress_coef_nobc_df, vars=['prior_conf', 'cpp'], alpha=0.05)

# Running cluster permutation analysis between coefficients for baseline and non-baseline corrected analyses
pupil_regress_coef_df['conf_cpp_diff'] = pupil_regress_coef_df['prior_conf'] - pupil_regress_coef_nobc_df['cpp']

pupil_regress_coef_nobc_df['conf_cpp_diff'] = pupil_regress_coef_nobc_df['prior_conf'] - pupil_regress_coef_nobc_df['cpp']

between_coef_permutation_results, between_coef_significant_clusters_df = group_stats_analysis.compute_group_level_permutation_cluster_test(group_level_df=pupil_regress_coef_df, vars = ['conf_cpp_diff'], alpha=0.05)

between_coef_permutation_results_nobc, between_coef_significant_clusters_nobc_df = group_stats_analysis.compute_group_level_permutation_cluster_test(group_level_df=pupil_regress_coef_nobc_df, vars=['conf_cpp_diff'], alpha=0.05)

# Creating a dataframe to contain the averaged response across time points within the epoch
print(f'\n Creating dataframe containing the averaged coefficients for the baseline corrected pupil signal')

coefs_df = group_pupil.compute_peristim_mean_sem_by_count(pupil_regress_coef_df, value_cols=['cpp', 'prior_conf', 'conf_cpp_diff'])

coefs_nobc_df = group_pupil.compute_peristim_mean_sem_by_count(pupil_regress_coef_nobc_df, value_cols=['cpp', 'prior_conf', 'conf_cpp_diff'])

# Saving dataframes
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'pupil_regress_coef_df': pupil_regress_coef_df,
    'pupil_regress_coef_nobc_df': pupil_regress_coef_nobc_df,
    'coefs_df': coefs_df,
    'coefs_nobc_df': coefs_nobc_df,
    'cluster_permutation_analysis': significant_clusters_df,
    'cluster_permutation_analysis_nobc': significant_clusters_nobc_df, 
    'between_coef_permutation_analysis': between_coef_significant_clusters_df,
    'between_coef_permutation_analysis_nobc': between_coef_significant_clusters_nobc_df
},
directory_path=BASE_DIR,
condition_str=CONDITION_STR,
task='Pupil') 