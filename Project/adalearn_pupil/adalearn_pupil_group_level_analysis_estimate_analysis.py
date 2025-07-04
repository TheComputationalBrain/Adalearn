"""
This script performs group-level analysis of estimates for performance of the Pupil task of the Adalearn project. It loads participant-level data, computes the association between participants' behavioural data and model estimates, and performs statistical tests to assess the significance of these associations. The results are saved to .csv files for further analysis. The following scripts need to be run before this script to load the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_model_parameters.py
"""
import pandas as pd
from scipy import stats
from pathlib import Path

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_participant_level_functions_comp as comp

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, exclude_zero


dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'change_point_indices_df', 
        'computational_var_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(Path(f'{base_dir}/data'))

excluded_participants = list(dataframes['exclude_participants_behav_cor_df']['participant'])

group_behav_v_model_estimate_corr_list = []

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

    print(f"\nAnalysing association between participants' and model estimates...")
    # Evaluating the association between participants' and model estimates
    behav_v_model_estimate_df = comp.behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, outcome_var='estimate', pred_var='ideal_observer', exclude_zero=exclude_zero, plot_behav_v_model_scatter=False, add_regression=True)

    behav_v_model_estimate_df['participant'] = participant_str

    group_behav_v_model_estimate_corr_list.append(behav_v_model_estimate_df)

# Collate lists to dataframes
group_behav_v_model_estimate_corr_df = pd.concat(group_behav_v_model_estimate_corr_list,ignore_index=True)

# Perform group-level correlation and regression parameter assessment
# Collating regression parameters at participant level
behav_model_intercept_df = group_behav_v_model_estimate_corr_df[group_behav_v_model_estimate_corr_df['variable'] == 'const']
behav_model_estimate_intercept_mean = behav_model_intercept_df['slope'].mean()
behav_model_estimate_intercept_std = behav_model_intercept_df['slope'].std()
behav_model_estimate_intercept_sem = behav_model_intercept_df['slope'].sem()

behav_model_slope_df = group_behav_v_model_estimate_corr_df[group_behav_v_model_estimate_corr_df['variable'] == 'ideal_observer']
behav_model_estimate_slope_mean = behav_model_slope_df['slope'].mean()
behav_model_estimate_slope_std = behav_model_intercept_df['slope'].std()
behav_model_estimate_slope_sem = behav_model_intercept_df['slope'].sem()

# Collating correlation coefficients at participant level
behav_model_corr_mean = behav_model_slope_df['correlation'].mean()
behav_model_corr_std = behav_model_slope_df['correlation'].std()
behav_model_corr_sem = behav_model_slope_df['correlation'].sem()

# One-sample t-tests on regression and correlation parameters
print(f'Running one-sample t-tests on regression and correlation parameters...')
behav_model_corr_ttest = stats.ttest_1samp(behav_model_slope_df['correlation'], popmean=0)
behav_model_slope_ttest_zero = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=0)
behav_model_slope_ttest_one = stats.ttest_1samp(behav_model_slope_df['slope'], popmean=1)

behav_model_estimate_coef_df = pd.DataFrame({
    'mean_intercept': [behav_model_estimate_intercept_mean],
    'std_intercept': [behav_model_estimate_intercept_std],
    'sem_intercept': [behav_model_estimate_intercept_sem],
    'mean_slope': [behav_model_estimate_slope_mean],
    'std_slope': [behav_model_estimate_slope_std],
    'sem_slope': [behav_model_estimate_slope_sem],
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

# Save group-level csv
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'behav_v_model_estimate_corr_df': group_behav_v_model_estimate_corr_df,
    'behav_v_model_estimate_coef_df': behav_model_estimate_coef_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil'
)