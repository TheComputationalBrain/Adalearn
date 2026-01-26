"""
This script evaluate pupil effects on shifted learning rate and the mediating influence of pupil responses on uncertainty-associated effects on shifted learning rates for participants undertaking the Pupil task in the adalearn project. It computes the coefficients for the mediating baseline-corrected and non-baseline-corrected pupil responses at a participant level, before evaluating the coefficients at a group level. The results are saved in .csv files for further analysis. The following script need to be run before this script to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_model_parameters.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_learning_rate_analysis.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_assessment.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_peristimulus_regression.py
- adalearn_pupil/adalearn_pupil_participant_level_analysis_update_lag_assessments.py

"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.metrics import mean_squared_error
import pingouin
import sys
import statsmodels.api as sm

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil
import adalearn_analysis_participant_level_functions_pupil as pupil

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR, EXCLUDE_ZERO



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'exclude_participants_pupil_analysis_df', 
        'shifted_estimate_results', 
        'change_point_indices_df', 
        'outcome_epochs_time_df', 
        'outcome_epoched_pupil_df',
        'outcome_epochs_columns_df', 
        'outcome_epochs_to_remove_df',
        'outcome_epochs_time_nobc_df', 
        'outcome_epoched_pupil_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc', 
        'uncertainty_params_ttest_df'],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(directory=Path(f'{BASE_DIR}/data'))

excluded_participants = list(set(
    list(dataframes['exclude_participants_behav_cor_df']['participant']) +
    list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])
))

pupil_lr_regress_raw_list = []
pupil_bc_effect = []
pupil_nobc_effect =[]
pupil_bc_around_cp_list = []
pupil_nobc_around_cp_list = []
pupil_v_lr_regress_coef_df_list = []
nobc_mediation_results_list = []
bc_mediation_results_list = []



# Running for all participants then excluding those identified due to excessive interpolation 
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{BASE_DIR}/data/sub-{participant_str}')
    else:
        continue

    # Getting individual participant data
    comp_df = dataframes['shifted_estimate_results'][dataframes['shifted_estimate_results']['participant']==participant].copy()
    comp_df = comp_df.reset_index()

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant']==participant].copy()

    epoch_data = dataframes['outcome_epoched_pupil_df'][dataframes['outcome_epoched_pupil_df']['participant'] == participant]

    epoch_nobc_data = dataframes['outcome_epoched_pupil_nobc_df'][dataframes['outcome_epoched_pupil_nobc_df']['participant'] == participant]

    epoch_time = dataframes['outcome_epochs_time_df'][dataframes['outcome_epochs_time_df']['participant']==participant]

    epoch_nobc_time = dataframes['outcome_epochs_time_nobc_df'][dataframes['outcome_epochs_time_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['outcome_epochs_columns_df'][dataframes['outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['outcome_epochs_to_remove_df'][dataframes['outcome_epochs_to_remove_df']['participant']==participant]
    
    # Processing epoched dataframes
    column_titles = list(analysed_epochs['columns'].values.astype(int))

    if len(column_titles)+len(epochs_to_remove) == len(comp_df):
        print(f'Correct number of epochs imported for participant {participant_str}')
    else:
        print(f'Incorrect number of epochs imported for participant {participant_str}')

    epoch_data = epoch_data.drop(columns=['participant'])
    epoch_data = epoch_data.T
    epoch_data.columns = column_titles
    epoch_data['time'] = epoch_time['time'].values.astype(float)

    epoch_nobc_data = epoch_nobc_data.drop(columns=['participant'])
    epoch_nobc_data = epoch_nobc_data.T
    epoch_nobc_data.columns = column_titles
    epoch_nobc_data['time'] = epoch_nobc_time['time'].values.astype(float)

    # Extracting cluster times to set window for regression and mediation analyses
    baseline_cor_start_time, baseline_cor_end_time = group_pupil.extract_cluster_times(
        dataframes['cluster_permutation_analysis'],
        epoch_data['time']
    )

    nobc_start_time, nobc_end_time = group_pupil.extract_cluster_times(
        dataframes['cluster_permutation_analysis_nobc'],
        epoch_nobc_data['time']
    )

    # Convert to list (if not already)
    epochs_to_remove_list = epochs_to_remove['epochs'].tolist()

    # Remove rows from the computational variable dataframe where the row has been taken out of the pupil data due to excessive interpolation
    data_comp_filtered = comp_df[~comp_df.index.isin(epochs_to_remove_list)].copy()

    # Preparing the pupil data
    epoch_data = epoch_data.set_index('time')

    # Baseline corrected response refers to the average of the pupil signal over the window of a significant effect from the peristimulus regression
    filtered_pupil_data_bc = epoch_data.loc[(epoch_data.index >=  baseline_cor_start_time) & (epoch_data.index <=  baseline_cor_end_time)]
    pupil_variable_bc = np.array(filtered_pupil_data_bc.mean())

    avg_pupil_bc_effect = pd.DataFrame({
        'participant': [participant_str],
        'avg_pupil_effect': [pupil_variable_bc.mean()]
    })

    pupil_bc_effect.append(avg_pupil_bc_effect)

    epoch_nobc_data = epoch_nobc_data.set_index('time')

    # The non-baseline corrected response refers to the average of the pupil signal from the beginning of the observed peristimulus effect to outcome presentation
    filtered_pupil_data_nobc = epoch_nobc_data.loc[(epoch_nobc_data.index >= nobc_start_time) & (epoch_nobc_data.index <= 0)]
    pupil_variable_nobc = np.array(filtered_pupil_data_nobc.mean())

    avg_pupil_nobc_effect = pd.DataFrame({
        'participant': [participant_str],
        'avg_pupil_effect': [pupil_variable_nobc.mean()]
    })

    pupil_nobc_effect.append(avg_pupil_nobc_effect)

    # Mediation analysis data frame
    mediation_analysis_df = data_comp_filtered[['task_session', 'outcome_num', 'prior_conf', 'cpp', 'learning_rate_shifted_prior_conf', 'learning_rate_shifted_cpp']].copy()
    mediation_analysis_df['pupil_bc'] = pupil_variable_bc
    mediation_analysis_df['pupil_nobc'] = pupil_variable_nobc

    # Checking that learning rate doesn't exceed the bound -0.6 and +1.3
    valid_range = (-0.6, 1.3)

    # Shifted columns
    shifted_var = ['learning_rate_shifted_prior_conf', 'learning_rate_shifted_cpp']

    # Trimming learning rates for each shift 
    for var in shifted_var:

        cleaned_mediation_analysis_df = mediation_analysis_df.copy()

        before_n = len(cleaned_mediation_analysis_df)

        cleaned_mediation_analysis_df = cleaned_mediation_analysis_df[
            cleaned_mediation_analysis_df[var].between(*valid_range)
        ]

        after_n = len(cleaned_mediation_analysis_df)

        removed_n = before_n - after_n

        print(f"Participant {participant_str}: Removed {removed_n} rows due to learning rate being outside the range {valid_range} for {var}.")


        # Explicitly drop trials with missing data in mediation variables
        before_dropna = len(cleaned_mediation_analysis_df)

        cleaned_mediation_analysis_df = cleaned_mediation_analysis_df.dropna(
            subset=[
                'prior_conf',
                'cpp',
                'pupil_bc',
                'pupil_nobc',
                'learning_rate_shifted_prior_conf',
                'learning_rate_shifted_cpp'
            ]
        )

        after_dropna = len(cleaned_mediation_analysis_df)

        print(
            f"Participant {participant_str}: "
            f"Dropped {before_dropna - after_dropna} rows due to NaNs in mediation variables."
            f" Remaining rows: {after_dropna}."
        )

        for col in cleaned_mediation_analysis_df.columns:
                cleaned_mediation_analysis_df[f'{col}_zscore'] = stats.zscore(cleaned_mediation_analysis_df[col], nan_policy='omit')

        # Running mediation analysis
        print(f'\nRunning mediation analysis...')

        if var == 'learning_rate_shifted_prior_conf':

            nobc_mediation_analysis = pingouin.mediation_analysis(data=cleaned_mediation_analysis_df, x='prior_conf_zscore', m=['pupil_nobc_zscore'], y='learning_rate_shifted_prior_conf_zscore', covar=None, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

            nobc_mediation_analysis['participant'] = participant_str

            nobc_mediation_results_list.append(nobc_mediation_analysis)

        elif var == 'learning_rate_shifted_cpp':

            bc_mediation_analysis = pingouin.mediation_analysis(data=cleaned_mediation_analysis_df, x='cpp_zscore', m=['pupil_bc_zscore'], y='learning_rate_shifted_cpp_zscore', covar=None, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

            bc_mediation_analysis['participant'] = participant_str

            bc_mediation_results_list.append(bc_mediation_analysis)


group_nobc_mediation_df = pd.concat(nobc_mediation_results_list, ignore_index=True)
group_bc_mediation_df = pd.concat(bc_mediation_results_list, ignore_index=True)

# Collating the mediation analysis
group_nobc_mediation_analysis_df = pd.DataFrame()

n_boot = 5000 

for path in group_nobc_mediation_df['path'].unique():
    data = group_nobc_mediation_df[group_nobc_mediation_df['path'] == path]
    coefs = data['coef'].values
    mean = data['coef'].mean()
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    boot_mean = np.mean(boot_means)
    boot_sem = np.std(boot_means, ddof=1)

    nobc_df_mediation = pd.DataFrame({
        'path': [path_label],
        'sample_mean': [mean],
        'boot_mean': [boot_mean],
        'sem': [boot_sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    group_nobc_mediation_analysis_df = pd.concat([group_nobc_mediation_analysis_df, nobc_df_mediation], ignore_index=True)

print(group_nobc_mediation_analysis_df)


group_bc_mediation_analysis_df = pd.DataFrame()

for path in group_bc_mediation_df['path'].unique():
    data = group_bc_mediation_df[group_bc_mediation_df['path'] == path]
    coefs = data['coef'].values
    mean = data['coef'].mean()
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    boot_mean = np.mean(boot_means)
    boot_sem = np.std(boot_means, ddof=1)

    bc_df_mediation = pd.DataFrame({
        'path': [path_label],
        'sample_mean': [mean],
        'boot_mean': [boot_mean],
        'sem': [boot_sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    group_bc_mediation_analysis_df = pd.concat([group_bc_mediation_analysis_df, bc_df_mediation], ignore_index=True)

print(group_bc_mediation_analysis_df)

# Saving dataframes
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'shifted_lr_uncert_nobc_mediation': group_nobc_mediation_analysis_df,
    'shifted_lr_uncert_bc_mediation': group_bc_mediation_analysis_df,
},
directory_path=BASE_DIR,
condition_str=CONDITION_STR,
task='Pupil')