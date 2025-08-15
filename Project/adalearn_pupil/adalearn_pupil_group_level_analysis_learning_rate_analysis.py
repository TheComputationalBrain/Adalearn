"""
This script performs group-level analysis of learning rate across participants undertaking the Pupil task in the Adalearn project. It evaluates the association between participants' learning rates and those predicted by a normative model, computes learning rates around change points, and assesses the impact of uncertainty parameters on learning rates. The results are saved to .csv files. The following script are required to be run before this script to load the required .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_participant_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_participant_level_analysis_model_parameters.py
"""
import pandas as pd
import sys
from scipy import stats
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_statistical_analysis as group_stat_analysis 
import adalearn_analysis_participant_level_functions_comp as comp
import adalearn_analysis_participant_level_functions_data_utilities as data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, exclude_zero


dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'behavioural_df', 
        'change_point_indices_df', 
        'computational_var_df'
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(Path(f'{base_dir}/data'))

excluded_participants = list(dataframes['exclude_participants_behav_cor_df']['participant'])

group_update_task_session_list = []
group_averaged_lr_session_list = []
group_behav_v_model_lr_corr_list = []
group_averaged_lr_around_cp_list = []
group_filtered_lr_around_cp_list = []
group_lr_corr_around_cp_list = []
group_uncertainty_params_on_lr_list = []

# For loop, loading the behavioural data and computed model parameters and then evaluating the effect on learning rate
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{base_dir}/data/sub-{participant_str}')
    else:
        continue

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant].copy()
    
    computational_var_df = dataframes['computational_var_df'][dataframes['computational_var_df']['participant'] == participant].copy()

    behavioural_df = dataframes['behavioural_df'][dataframes['behavioural_df']['participant'] == participant].copy()

    # Evaluating the learning rate to extract baseline learning rate
    lr_evaluation_dict = data_utils.evaluation_of_learning_rate(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, computational_var_df=computational_var_df, change_point_indices=change_point_indices, quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=False)

    lr_evaluation_dict['avg_lr_session_df']['participant'] = participant_str

    group_averaged_lr_session_list.append(lr_evaluation_dict['avg_lr_session_df'])

    # Extracting the update proportion for each task session
    update_dict = {'participant': [], 'task_session': [], 'update_count': [], 'update_percentage': []}
   
    for session in computational_var_df['task_session'].unique():
        update_data = computational_var_df[computational_var_df['task_session'] == session].copy()
        sum_update = update_data['was_updated_estimate'].sum()

        update_dict['participant'].append(participant_str)
        update_dict['task_session'].append(session)
        update_dict['update_count'].append(sum_update)
        update_dict['update_percentage'].append(sum_update / len(update_data) * 100)

    update_df = pd.DataFrame(update_dict)
    group_update_task_session_list.append(update_df)

    print(f"\nAnalysing association between participants' and model learning rate...")
    # Evaluating the association between participants' and model learning rates
    behav_v_model_lr_df = comp.behav_v_model_linear_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df
    , outcome_var='cleaned_learning_rate_estimate', pred_var='cleaned_learning_rate_ideal_observer', exclude_zero=exclude_zero, plot_behav_v_model_scatter=False, add_regression=True)

    behav_v_model_lr_df['participant'] = participant_str

    group_behav_v_model_lr_corr_list.append(behav_v_model_lr_df)

    print(f"\nAnalysing association between participants' and model learning rates around change points...")
    # Computing the learning rate around change point (participants' estimates)
    filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp = comp.learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=computational_var_df, change_point_indices=change_point_indices, learning_rate_parameter='estimate', pre=2, post=15, exclude_zero=exclude_zero, plot_learning_rate_by_cp=False)

    # Computing the learning rate around change points (model estimates)
    filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp = comp.learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=computational_var_df, change_point_indices=change_point_indices, learning_rate_parameter='ideal_observer', pre=2, post=15, exclude_zero = exclude_zero, plot_learning_rate_by_cp=False)

    averaged_learning_rate_by_cp = pd.merge(left=averaged_behav_learning_rate_by_cp, right=averaged_model_learning_rate_by_cp, on=['relative_position'], how='outer', suffixes=('', '_model'))

    filtered_behav_learning_rate_by_cp['participant'] = participant_str
    averaged_learning_rate_by_cp['participant'] = participant_str

    group_averaged_lr_around_cp_list.append(averaged_learning_rate_by_cp)
    group_filtered_lr_around_cp_list.append(filtered_behav_learning_rate_by_cp)

    # Evaluating the correlation between participants' and model estimates around change points
    change_point_avg_lrs_df, lr_around_cp_corr_df = comp.correlation_around_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,filtered_learning_rate_by_cp=filtered_behav_learning_rate_by_cp, average_over='position', plot_correlation_round_cp=False)

    change_point_avg_lrs_df['participant'] = participant_str

    group_lr_corr_around_cp_list.append(lr_around_cp_corr_df)

    (f"\nComputing the effect of uncertainty parameters on learning rate...")
    # Linear regression for uncertainty parameters on participants' learning rate
    learning_rate_analysis_dict = comp.uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='estimate', exclude_zero=exclude_zero, include_interaction=True, plot_corr_subplots=False)

    # Linear regression for uncertainty parameters on model learning rate (normative account)
    learning_rate_model_analysis_dict = comp.uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='ideal_observer', exclude_zero=exclude_zero, include_interaction=True, plot_corr_subplots=False)

    group_uncertainty_params_on_lr = pd.merge(left=learning_rate_analysis_dict['uncertainty_v_lr_df'], right=learning_rate_model_analysis_dict['uncertainty_v_lr_df'], on=['variable'], how='outer', suffixes=('', '_model'))

    group_uncertainty_params_on_lr['participant'] = participant_str

    group_uncertainty_params_on_lr_list.append(group_uncertainty_params_on_lr)

# Collate lists to dataframes
group_update_task_session_df = pd.concat(group_update_task_session_list, ignore_index=True)
group_behav_v_model_lr_corr_df = pd.concat(group_behav_v_model_lr_corr_list, ignore_index=True)
group_averaged_lr_around_cp_df = pd.concat(group_averaged_lr_around_cp_list, ignore_index=True)
group_filtered_lr_around_cp_df = pd.concat(group_filtered_lr_around_cp_list, ignore_index=True)
group_lr_corr_around_cp_df = pd.concat(group_lr_corr_around_cp_list, ignore_index=True)
group_uncertainty_params_on_lr_df = pd.concat(group_uncertainty_params_on_lr_list)
group_averaged_lr_session_df = pd.concat(group_averaged_lr_session_list, ignore_index=True)

# Getting the average update percentage across task sessions for each participant
average_update_percent_dict = {
    'participant': [],
    'avg_update_percentage': []
}

for participant in group_update_task_session_df['participant'].unique():
    participant_df = group_update_task_session_df[group_update_task_session_df['participant'] == participant].copy()
    avg_update_percentage = participant_df['update_percentage'].mean()

    average_update_percent_dict['participant'].append(participant)
    average_update_percent_dict['avg_update_percentage'].append(avg_update_percentage)
    
group_average_update_percent_df = pd.DataFrame(average_update_percent_dict)

# Perform group-level correlation and regression parameter assessment on learning rate across all task sessions
# Collating regression parameters at participant level
behav_model_intercept_df = group_behav_v_model_lr_corr_df[group_behav_v_model_lr_corr_df['variable'] == 'const']
behav_model_lr_intercept_mean = behav_model_intercept_df['slope'].mean()
behav_model_lr_intercept_std = behav_model_intercept_df['slope'].std()
behav_model_lr_intercept_sem = behav_model_intercept_df['slope'].sem()

behav_model_slope_df = group_behav_v_model_lr_corr_df[group_behav_v_model_lr_corr_df['variable'] == 'cleaned_learning_rate_ideal_observer']
behav_model_lr_slope_mean = behav_model_slope_df['slope'].mean()
behav_model_lr_slope_std = behav_model_intercept_df['slope'].std()
behav_model_lr_slope_sem = behav_model_intercept_df['slope'].sem()

# Collating correlation coefficients at participant level
behav_model_corr_mean = behav_model_slope_df['correlation'].mean()
behav_model_corr_std = behav_model_slope_df['correlation'].std()
behav_model_corr_sem = behav_model_slope_df['correlation'].sem()

# One-sample t-tests on regression and correlation parameters
print(f'Running one-sample t-tests on regression and correlation parameters...')
behav_model_corr_ttest = stats.ttest_1samp(behav_model_slope_df['correlation'], popmean=0)
behav_model_slope_ttest_zero = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=0)
behav_model_slope_ttest_one = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=1)

behav_model_lr_coef_df = pd.DataFrame({
    'mean_intercept': [behav_model_lr_intercept_mean],
    'std_intercept': [behav_model_lr_intercept_std],
    'sem_intercept': [behav_model_lr_intercept_sem],
    'mean_slope': [behav_model_lr_slope_mean],
    'std_slope': [behav_model_lr_slope_std],
    'sem_slope': [behav_model_lr_slope_sem],
    'slope_tvalue_against_zero': [behav_model_slope_ttest_zero[0]],
    'slope_pvalue_against_zero': [behav_model_slope_ttest_zero[1]],
    'slope_tvalue_against_one': [behav_model_slope_ttest_one[0]],
    'slope_pvalue_against_one': [behav_model_slope_ttest_one[1]],
    'mean_corr': [behav_model_corr_mean],
    'std_corr': [behav_model_corr_std],
    'sem_corr': [behav_model_corr_sem],
    'corr_tvalue_against_zero': [behav_model_corr_ttest[0]],
    'corr_pvalue_against_zero': [behav_model_corr_ttest[1]],
})

# Perform group-level correlation and regression parameter assessment on learning rate across all task sessions
# Group level assessment of learning rate around change points between participants and normative theory
lr_by_cp_mean_behav = group_averaged_lr_around_cp_df.groupby('relative_position')['mean'].mean()
lr_by_cp_mean_model = group_averaged_lr_around_cp_df.groupby('relative_position')['mean_model'].mean()

group_level_corr_lr_around_cp = stats.pearsonr(x=lr_by_cp_mean_model, y=lr_by_cp_mean_behav, alternative="two-sided")

# Collate group-level correlation and regression parameter assessment on learning rate around change points - (participant-level analysis)
lr_around_cp_corr_mean = group_lr_corr_around_cp_df['correlation'].mean()
lr_around_cp_corr_std = group_lr_corr_around_cp_df['correlation'].std()
lr_around_cp_corr_sem = group_lr_corr_around_cp_df['correlation'].sem()

lr_around_cp_corr_ttest = stats.ttest_1samp(group_lr_corr_around_cp_df['correlation'], popmean=0)

behav_model_lr_around_cp_corr_df = pd.DataFrame({
'group_corr_coef': [group_level_corr_lr_around_cp[0]],
'group_corr_pvalue': [group_level_corr_lr_around_cp[1]],
'participant_level_mean_corr': [lr_around_cp_corr_mean],
'participant_level_std_corr': [lr_around_cp_corr_std],
'participant_level_sem_corr': [lr_around_cp_corr_sem],
'participant_level_tvalue_against_zero': [lr_around_cp_corr_ttest[0]],
'participant_level_pvalue_against_zero': [lr_around_cp_corr_ttest[1]],
})

# Run permutation cluster analysis on learning rate around change points
cluster_analysis_values_list = []

for id in group_filtered_lr_around_cp_df['participant'].unique():
    # Subset participant data
    participant_data = group_filtered_lr_around_cp_df[group_filtered_lr_around_cp_df['participant'] == id].copy()
    
    # Subset relevant columns
    participant_data = participant_data[['relative_position', 'cleaned_learning_rate_estimate', 'task_session', 'outcome_num',  'cum_outcome_num', 'change_point_count', 'participant']]

    # Subtract baseline (mean before change point)
    baseline_mean = participant_data[participant_data['relative_position'] < 0]['cleaned_learning_rate_estimate'].mean()
    participant_data['adjusted_lr'] = participant_data['cleaned_learning_rate_estimate'] - baseline_mean

    # Compute mean adjusted learning rate per relative position
    mean_per_position = participant_data.groupby('relative_position')['adjusted_lr'].mean().reset_index()
    mean_per_position['participant'] = id

    cluster_analysis_values_list.append(mean_per_position)

# Combine all participant data
cluster_analysis_df = pd.concat(cluster_analysis_values_list, ignore_index=True)
cluster_analysis_df.rename(columns={'adjusted_lr': 'cluster_data', 'relative_position': 'time_point'}, inplace=True)

cluster_based_permutation_results, significant_clusters_df = group_stat_analysis.compute_group_level_permutation_cluster_test(group_level_df=cluster_analysis_df, pivot_column='time_point', vars = ['cluster_data'], alpha=0.05)

# Perform group_level analysis of uncertainty parameters on learning rate
# Collate regression weights and then perform 1-sample and paired-sample t-tests
behav_conf_coef = group_uncertainty_params_on_lr_df[group_uncertainty_params_on_lr_df['variable'] == 'conf_zscore'].sort_values('participant').copy()
behav_cpp_coef = group_uncertainty_params_on_lr_df[group_uncertainty_params_on_lr_df['variable'] == 'cpp_zscore'].sort_values('participant').copy()

behav_conf_coef_mean = behav_conf_coef['coefficient'].mean()
behav_cpp_coef_mean = behav_cpp_coef['coefficient'].mean()
behav_conf_coef_sem = stats.sem(behav_conf_coef['coefficient'])
behav_cpp_coef_sem = stats.sem(behav_cpp_coef['coefficient'])

# t-tests against zero and between coefficients
t_conf = stats.ttest_1samp(behav_conf_coef['coefficient'], 0)
t_cpp = stats.ttest_1samp(behav_cpp_coef['coefficient'], 0)
t_between_coef = stats.ttest_rel(behav_conf_coef['coefficient'], behav_cpp_coef['coefficient'], alternative='two-sided')

uncertainty_params_ttest_df = pd.DataFrame({
    'variable': ['conf_zscore', 'cpp_zscore', 'between_coef'],
    'regression_coef': [behav_conf_coef_mean, behav_cpp_coef_mean, None],
    'regress_coef_sem': [behav_conf_coef_sem, behav_cpp_coef_sem, None],
    't_statistic': [t_conf.statistic, t_cpp.statistic, t_between_coef.statistic],
    'p_value': [t_conf.pvalue, t_cpp.pvalue, t_between_coef.pvalue]
})

# Save group-level csv
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'update_freq_df': group_average_update_percent_df,
    'behav_v_model_lr_corr_df': group_behav_v_model_lr_corr_df,
    'averaged_lr_around_cp_df': group_averaged_lr_around_cp_df,
    'lr_corr_around_cp_df': group_lr_corr_around_cp_df,
    'uncertainty_params_on_lr_df': group_uncertainty_params_on_lr_df,
    'uncertainty_params_ttest_df': uncertainty_params_ttest_df,
    'averaged_lr_session_df': group_averaged_lr_session_df,
    'behav_model_lr_coef_df': behav_model_lr_coef_df,
    'lr_around_cp_sig_clusters': significant_clusters_df,
    'uncertainty_params_ttest_df': uncertainty_params_ttest_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil'
)
