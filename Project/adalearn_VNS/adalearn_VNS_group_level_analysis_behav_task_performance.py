"""
This script performs group-level analysis of task performance data for the VNS task in the Adalearn project. It loads individual participant data, computes performance metrics, conducts statistical analyses, and saves the results to .csv files. 
"""
import sys
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_participant_level_functions_behav as behav
from adalearn_analysis_participant_level_functions_participant_info import vns_condition_characteristics

from adalearn_VNS.adalearn_VNS_participant_info_and_parameters import base_dir



participant_list = group_data_utils.get_participant_numbers(Path(f'{base_dir}/data'))

group_behavioural_df_list = []
group_mean_error_df_list = []
group_change_point_indices_list = []

# For loop, extracting data from behavioural files for each participant within data list
for participant in participant_list:
    participant_str = str(participant).zfill(2)
    print(f"\n\n\nProcessing participant {participant_str}...")
    directory_path = Path(f'{base_dir}/data/sub-{participant_str}')

    _, _, _, tsv_df, vns_info_dict = vns_condition_characteristics(participant=participant, directory_path=directory_path)

    for key in vns_info_dict.keys():
        condition_str = vns_info_dict[key]['condition']
        task_sessions = vns_info_dict[key]['task_sessions']  

        print(f"\nLoading task data and computing task performance for the {key} condition...")

        behavioural_df, _ = behav.load_behavioural_data( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

        behavioural_df['participant'] = participant_str
        behavioural_df['condition'] = key

        group_behavioural_df_list.append(behavioural_df)

        mean_abs_error_df = behav.behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_performance=False)

        mean_abs_error_df['coef_variation'] = (mean_abs_error_df['std_abs_error']/ mean_abs_error_df['mean_abs_error']) * 100

        mean_abs_error_df['participant'] = participant_str
        mean_abs_error_df['condition'] = key

        group_mean_error_df_list.append(mean_abs_error_df)

        print(f"\nExtracting change point indices...")

        num_change_points, change_points_per_session, change_point_indices, behavioural_df = behav.behavioural_change_point_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=behavioural_df, plot_histogram_zscores=False)

        change_point_indices['participant'] = participant_str
        change_point_indices['condition'] = key

        group_change_point_indices_list.append(change_point_indices)


# Concatenate the dataframes in the lists into a single group-level dataframe
group_level_behavioural_df = pd.concat(group_behavioural_df_list, ignore_index=True)
group_level_mean_error_df = pd.concat(group_mean_error_df_list, ignore_index=True)
group_level_change_point_indices_df = pd.concat(group_change_point_indices_list, ignore_index=True)


# Running linear mixed model on performance metrics
model_mean_error = mixedlm("mean_abs_error ~ condition + task_session + condition:task_session", data=group_level_mean_error_df, groups=group_level_mean_error_df["participant"])
result_mean_error= model_mean_error.fit()
print(result_mean_error.summary())

model_mean_cv = mixedlm("coef_variation ~ condition + task_session + condition:task_session", data=group_level_mean_error_df, groups=group_level_mean_error_df["participant"])
result_mean_cv = model_mean_cv.fit()
print(result_mean_cv.summary())

model_mean_score = mixedlm("score ~ condition + task_session + condition:task_session", data=group_level_mean_error_df, groups=group_level_mean_error_df["participant"])
result_mean_score = model_mean_score.fit()
print(result_mean_score.summary())

# Running factorial ANOVA on performance metrics
anova_mean_error = pg.rm_anova(
    data=group_level_mean_error_df,
    dv='mean_abs_error',
    within=['task_session', 'condition'],
    subject='participant',
    detailed=True
)

anova_mean_error['depend_var'] = 'mean_abs_error'

print('Outcome of factorial repeated measures ANOVA [task session and condition] on mean absolute error...')
print(f'\n{anova_mean_error}')

anova_mean_cv = pg.rm_anova(
    data=group_level_mean_error_df,
    dv='coef_variation',
    within=['task_session', 'condition'],
    subject='participant',
    detailed=True
)

anova_mean_cv['depend_var'] = 'coef_variation'

print('Outcome of factorial repeated measures ANOVA [task session and condition] on mean coefficient of variation...')
print(f'\n{anova_mean_cv}')

anova_mean_score = pg.rm_anova(
    data=group_level_mean_error_df,
    dv='score',
    within=['task_session', 'condition'],
    subject='participant',
    detailed=True
)

anova_mean_score['depend_var'] = 'score'

print('Outcome of factorial repeated measures ANOVA [task session and condition] on mean score...')
print(f'\n{anova_mean_score}')

anova_df = pd.concat([anova_mean_error, anova_mean_cv, anova_mean_score], ignore_index=True)

# Saving group level data to .csv files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    "behavioural_df": group_level_behavioural_df,
    "mean_abs_error_df": group_level_mean_error_df,
    "change_point_indices_df": group_level_change_point_indices_df,
    "behavioural_anova_df": anova_df
    }, 
directory_path=base_dir,
condition_str='combined',
task='VNS')