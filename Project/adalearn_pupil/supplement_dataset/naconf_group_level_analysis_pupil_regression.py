"""
This scripts runs peristimulus pupil regression analysis for a supplementary dataset (NACONF) within the Pupil task of the Adalearn project. The effect of uncertainty parameters (prior confidence and change point probability) is examined on both phasic- and tonic-associated pupil signal epochs. Permutation cluster-based analysis compares effects against zero. Plots are generated within the script. 
"""
import os
import sys
import pandas as pd
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil
import adalearn_analysis_group_level_functions_statistical_analysis as group_stats
import adalearn_pupil.supplement_dataset.naconf_ideal_observer_model_functions as model
import adalearn_pupil.supplement_dataset.naconf_pupil_functions as pupil
import adalearn_pupil.supplement_dataset.naconf_pupil_regression_functions as pupil_regress

from adalearn_pupil.supplement_dataset.naconf_participant_list import usable_participants
from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import NACONF_BASE_DIR, LIST_SELECTED_PARTICIPANT, TASK_SESSIONS, FROM_SESSION, TO_SESSION


task_session_df_list = []
regression_coefs_list = []
regression_nobc_coefs_list = []
outcome_epochs_time_df_list =[]
outcome_epoched_pupil_df_list = []
outcome_epochs_columns_df_list = []
outcome_blink_ratio_df_list = []
outcome_epochs_to_remove_df_list = []
outcome_epochs_time_nobc_df_list = []
outcome_epoched_pupil_nobc_df_list = []
outcome_epochs_columns_nobc_df_list = []
outcome_blink_ratio_nobc_df_list = []
outcome_epochs_to_remove_nobc_df_list = []


for participant in usable_participants:
    print(f"\n\n\nProcessing participant {participant}...")
    # Define the directory path for the participant
    directory_path = Path(f'{NACONF_BASE_DIR}/data/sub-{participant}')

    print(f"\nLoading task data and computing model derived variables...")
    task_session_df = model.load_task_behavioural_data(directory_path=directory_path, selected_participant=participant, task_sessions=TASK_SESSIONS)

    num_change_points, change_points_per_session, change_point_indices, task_session_df = model.change_point_analysis(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df)

    constants, P_C, P_MIN, P_MAX = model.load_constants(directory_path=directory_path, task_sessions=TASK_SESSIONS)

    task_session_df = model.run_inference_from_observations(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df, change_point_indices_df=change_point_indices, p_c=P_C, p_min=P_MIN, p_max=P_MAX, plot_surprise = False)

    task_session_df = model.compute_prior_conf_and_cpp_from_observations(participant_str=participant, directory_path=directory_path, task_session_df=task_session_df, change_point_indices_df=change_point_indices, p_c=P_C, p_min=P_MIN, p_max=P_MAX, estimate_type = 'mean', conf_type = 'prev-conf-pred', unit = 'uncertainty_sd', plot_prior_conf = False, plot_cpp = False)

    task_session_df_list.append(task_session_df)

    print(f"\nLoading, preprocessing and epoching pupil data...")
    preprocessed_pupil_df, pupil_trig_df, pupil_ses_df, blink_df, epoched_pupil_df, blink_ratio_epoch_df, epochs_to_remove = pupil.load_and_process_pupil_data(directory_path=directory_path, participant_str=participant, list_subject_selected=LIST_SELECTED_PARTICIPANT, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, dict_onsets=None, dict_num=None, session_list=None, outlier_limit_std=10, outlier_correct_average_amount=15, window_start_before=0.25, window_end_after=1.2, baseline_cor=[-0.25, 0], lpf=[5], select_eye=True, remove_epochs=False, z_score=True, plot_interpolation_process=False) 

    outcome_onsets, outcome_nums, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epochs_to_remove_df = pupil.all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant, list_subject_selected=LIST_SELECTED_PARTICIPANT, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, pupil_trig_df=pupil_trig_df, baseline_cor=[-0.25, 0], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    outcome_onsets_nobc, outcome_nums_nobc, outcome_epoched_pupil_nobc_df, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc, outcome_epochs_to_remove_df_nobc = pupil.all_outcome_pupil_epochs(directory_path=directory_path, participant_str=participant, list_subject_selected=LIST_SELECTED_PARTICIPANT, condition_task_sessions=TASK_SESSIONS, from_session=FROM_SESSION, to_session=TO_SESSION, pupil_trig_df=pupil_trig_df, baseline_cor=[], variable_df=None, variable_column=None, select_eye=True, remove_epochs=True, plot_averaged_pupil_response=False, plot_raster_pupil_response=False)

    # Processing epoch dataframes to make them aligned across participants (transposing them so time x epoch, after removing statistical)
    epoch_time_df, csv_pupil_epoch_df, columns_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_pupil_df, participant_str=participant)

    epoch_time_nobc_df, csv_pupil_epoch_nobc_df, columns_nobc_df = group_data_utils.process_epoched_df(epoched_df=outcome_epoched_pupil_nobc_df, participant_str=participant)


    dfs = [
    epoch_time_df,
    csv_pupil_epoch_df,
    columns_df,
    outcome_blink_ratio_df,
    outcome_epochs_to_remove_df,
    epoch_time_nobc_df,
    csv_pupil_epoch_nobc_df,
    columns_nobc_df,
    outcome_blink_ratio_df_nobc,
    outcome_epochs_to_remove_df_nobc
    ]

    df_lists = [
        outcome_epochs_time_df_list,
        outcome_epoched_pupil_df_list,
        outcome_epochs_columns_df_list,
        outcome_blink_ratio_df_list,
        outcome_epochs_to_remove_df_list,
        outcome_epochs_time_nobc_df_list,
        outcome_epoched_pupil_nobc_df_list,
        outcome_epochs_columns_nobc_df_list,
        outcome_blink_ratio_nobc_df_list,
        outcome_epochs_to_remove_nobc_df_list,
    ]

    for df, df_list in zip(dfs, df_lists):
        df['participant'] = participant
        df_list.append(df)

    print(f"\nRunning peristimulus regression analysis on baseline corrected pupil signal...")
    coefficients_df, pupil_regress_df, residuals_df = pupil_regress.pupil_peristimulus_regression_analysis(participant_str = participant, directory_path = directory_path, task_session_df = task_session_df, outcome_epoched_pupil_df = outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove_df, variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = True, plot_regression_analysis=False)

    regression_coefs_list.append(coefficients_df)

    print(f"\nRunning peristimulus regression analysis on non-baseline corrected pupil signal...")
    coefficients_df_nobc, pupil_regress_df_nobc, residuals_df_nobc = pupil_regress.pupil_peristimulus_regression_analysis(participant_str = participant, directory_path = directory_path, task_session_df = task_session_df, outcome_epoched_pupil_df = outcome_epoched_pupil_nobc_df, outcome_epochs_to_remove=outcome_epochs_to_remove_df_nobc, variables=['prior_conf', 'cpp'], covariates=None, baseline_corrected = False, plot_regression_analysis=False)

    regression_nobc_coefs_list.append(coefficients_df_nobc)

# Concatenate the dataframes in the lists into a single group-level dataframe
group_level_outcome_epochs_time_df = pd.concat(outcome_epochs_time_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_df = pd.concat(outcome_epoched_pupil_df_list, ignore_index=True)
group_level_outcome_epochs_columns_df = pd.concat(outcome_epochs_columns_df_list, ignore_index=True)
group_level_outcome_blink_ratio_df = pd.concat(outcome_blink_ratio_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_df = pd.concat(outcome_epochs_to_remove_df_list, ignore_index=True)
group_level_outcome_epochs_time_nobc_df = pd.concat(outcome_epochs_time_nobc_df_list, ignore_index=True)
group_level_outcome_epoched_pupil_nobc_df = pd.concat(outcome_epoched_pupil_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_columns_nobc_df = pd.concat(outcome_epochs_columns_nobc_df_list, ignore_index=True)
group_level_outcome_blink_ratio_nobc_df = pd.concat(outcome_blink_ratio_nobc_df_list, ignore_index=True)
group_level_outcome_epochs_to_remove_nobc_df = pd.concat(outcome_epochs_to_remove_nobc_df_list, ignore_index=True)
group_level_task_session_df = pd.concat(task_session_df_list, ignore_index=True)
group_level_regression_coefs_df = pd.concat(regression_coefs_list, ignore_index=True)
group_level_regression_nobc_coefs_df = pd.concat(regression_nobc_coefs_list, ignore_index=True)

# Create dataframes for storing data
group_level_regression_coefs_df['conf_cpp_diff'] = group_level_regression_coefs_df['prior_conf_zscore'] - group_level_regression_nobc_coefs_df['cpp_zscore']
group_level_regression_nobc_coefs_df['conf_cpp_diff'] = group_level_regression_nobc_coefs_df['prior_conf_zscore'] - group_level_regression_nobc_coefs_df['cpp_zscore']

coefs_df = group_pupil.compute_peristim_mean_sem_by_count(group_level_regression_coefs_df, value_cols=['cpp_zscore', 'prior_conf_zscore', 'conf_cpp_diff'])
coefs_nobc_df = group_pupil.compute_peristim_mean_sem_by_count(group_level_regression_nobc_coefs_df, value_cols=['cpp_zscore', 'prior_conf_zscore', 'conf_cpp_diff'])

# Permutation cluster analysis
baseline_corr_cluster_analysis_results, significant_clusters_df = group_stats.compute_group_level_permutation_cluster_test(group_level_regression_coefs_df, vars=['prior_conf_zscore', 'cpp_zscore'], alpha=0.05)
nonbaseline_corr_cluster_analysis_results, significant_clusters_nobc_df = group_stats.compute_group_level_permutation_cluster_test(group_level_regression_nobc_coefs_df, vars=['prior_conf_zscore', 'cpp_zscore'],  alpha=0.05)

# Between coefficient permutation cluster analysis
between_coef_permutation_results, between_coef_significant_clusters_df = group_stats.compute_group_level_permutation_cluster_test(group_level_df=group_level_regression_coefs_df, vars = ['conf_cpp_diff'], alpha=0.05)
between_coef_permutation_results_nobc, between_coef_significant_clusters_nobc_df = group_stats.compute_group_level_permutation_cluster_test(group_level_df=group_level_regression_nobc_coefs_df, vars=['conf_cpp_diff'], alpha=0.05)

# Save data to .csv files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'outcome_epochs_time_df': group_level_outcome_epochs_time_df,
    'outcome_epoched_pupil_df': group_level_outcome_epoched_pupil_df,
    'outcome_epochs_columns_df': group_level_outcome_epochs_columns_df,
    'outcome_blink_ratio_df': group_level_outcome_blink_ratio_df,
    'outcome_epochs_to_remove_df': group_level_outcome_epochs_to_remove_df,
    'outcome_epochs_time_nobc_df': group_level_outcome_epochs_time_nobc_df,
    'outcome_epoched_pupil_nobc_df': group_level_outcome_epoched_pupil_nobc_df,
    'outcome_epochs_columns_nobc_df': group_level_outcome_epochs_columns_nobc_df,
    'outcome_blink_ratio_nobc_df': group_level_outcome_blink_ratio_nobc_df,
    'outcome_epochs_to_remove_nobc_df': group_level_outcome_epochs_to_remove_nobc_df,
    'coefs_df': coefs_df,
    'coefs_nobc_df': coefs_nobc_df,
    'cluster_permutation_analysis': significant_clusters_df,
    'cluster_permutation_analysis_nobc': significant_clusters_nobc_df,
    'between_coef_significant_clusters_df': between_coef_significant_clusters_df,
    'between_coef_significant_clusters_nobc_df': between_coef_significant_clusters_nobc_df
},
directory_path=NACONF_BASE_DIR,
condition_str='01',
task='Pupil_NACONF') 