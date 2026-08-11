"""
This script evaluates how latent uncertainty parameters predict participants' apparent learning rate across different levels of tonic pupil size. It uses a linear regression model to assess the relationship between latent uncertainty parameters and learning rate, on data split by tonic pupil size quartiles. Additional analysis explored how the association between change point probability and phasic pupil signals were modulated by tonic arousal state. The results are saved in .csv files for further analysis. The following script need to be run before this script to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_model_parameters.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_learning_rate_analysis.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_assessment.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_peristimulus_regression.py

"""
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import pingouin as pg
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import multipletests

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil
import adalearn_analysis_participant_level_functions_pupil as pupil

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
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc', 
        'uncertainty_params_ttest_df',
        'outcome_epochs_euclidean_distance_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(directory=Path(f'{BASE_DIR}/data'))

excluded_participants = list(set(
    list(dataframes['exclude_participants_behav_cor_df']['participant']) +
    list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])
))


pupil_lr_regress_raw_list, pupil_bc_effect, pupil_nobc_effect, pupil_position_effect_list, comp_filtered_list = [], [], [], [], []


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

    epoch_time = dataframes['outcome_epochs_time_df'][dataframes['outcome_epochs_time_df']['participant']==participant]

    epoch_nobc_time = dataframes['outcome_epochs_time_nobc_df'][dataframes['outcome_epochs_time_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['outcome_epochs_columns_df'][dataframes['outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['outcome_epochs_to_remove_df'][dataframes['outcome_epochs_to_remove_df']['participant']==participant]

    epoch_positions = dataframes['outcome_epochs_euclidean_distance_df'][dataframes['outcome_epochs_euclidean_distance_df']['participant']==participant]
    
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

    epoch_positions = epoch_positions.drop(columns=['participant'])
    epoch_positions = epoch_positions.T
    epoch_positions.columns = column_titles
    epoch_positions['time'] = epoch_time['time'].values.astype(float)

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
    data_comp_filtered = comp_df[~comp_df.index.isin(epochs_to_remove_list)]
    comp_filtered_list.append(data_comp_filtered)

    # Preparing the pupil data
    epoch_data = epoch_data.set_index('time')

    # Baseline corrected response refers to the average of the pupil signal over the window of a significant effect from the peristimulus regression
    filtered_pupil_data_bc = epoch_data.loc[(epoch_data.index >=  baseline_cor_start_time) & (epoch_data.index <=  baseline_cor_end_time)]
    pupil_variable_bc = np.array(filtered_pupil_data_bc.mean())

    avg_pupil_bc_effect = pd.DataFrame({
        'participant': participant_str,
        'avg_pupil_effect': pupil_variable_bc,
        'epoch': range(len(pupil_variable_bc))
    })

    pupil_bc_effect.append(avg_pupil_bc_effect)

    epoch_nobc_data = epoch_nobc_data.set_index('time')

    # The non-baseline corrected response refers to the average of the pupil signal from the beginning of the observed peristimulus effect to outcome presentation
    filtered_pupil_data_nobc = epoch_nobc_data.loc[(epoch_nobc_data.index >= nobc_start_time) & (epoch_nobc_data.index <= 0)]
    pupil_variable_nobc = np.array(filtered_pupil_data_nobc.mean())

    avg_pupil_nobc_effect = pd.DataFrame({
        'participant': participant_str,
        'avg_pupil_effect': pupil_variable_nobc,
        'epoch': range(len(pupil_variable_nobc))
    })

    pupil_nobc_effect.append(avg_pupil_nobc_effect)

    # Position data
    epoch_positions = epoch_positions.set_index('time')

    filtered_position_data_bc = epoch_positions.loc[(epoch_positions.index >=  baseline_cor_start_time) & (epoch_positions.index <=  baseline_cor_end_time)]
    filtered_position_data_nobc = epoch_positions.loc[(epoch_positions.index >= nobc_start_time) & (epoch_positions.index <= 0)]
    pupil_position_bc_effect = np.array(filtered_position_data_bc.mean())
    pupil_position_nobc_effect = np.array(filtered_position_data_nobc.mean())
    pupil_position_df = pd.DataFrame({
        'pupil_position_bc': pupil_position_bc_effect,
        'pupil_position_nobc': pupil_position_nobc_effect
    })

    avg_pupil_pos_across_effects = pd.DataFrame({
        'participant': [participant_str],
        'avg_position_bc_effect': [pupil_position_bc_effect.mean()],
        'avg_position_nobc_effect': [pupil_position_nobc_effect.mean()]
    })

    pupil_position_effect_list.append(avg_pupil_pos_across_effects)

pupil_bc_effect_df = pd.concat(pupil_bc_effect, ignore_index=True)
pupil_nobc_effect_df = pd.concat(pupil_nobc_effect, ignore_index=True)
pupil_position_effect_df = pd.concat(pupil_position_effect_list, ignore_index=True)
comp_filtered_df = pd.concat(comp_filtered_list, ignore_index=True)

# Merging dataframes for subsequent regression analyses
pupil_nobc_effect_df['participant'] = (
    pupil_nobc_effect_df['participant'].astype(int)
)
pupil_bc_effect_df['participant'] = (
    pupil_bc_effect_df['participant'].astype(int)
)
comp_filtered_df['participant'] = (
    comp_filtered_df['participant'].astype(int)
)

comp_filtered_df = (
    comp_filtered_df
    .sort_values(['participant', 'index'])
)

comp_filtered_df['epoch'] = (
    comp_filtered_df
    .groupby('participant')
    .cumcount()
)

merged_df = (
    comp_filtered_df
    .merge(
        pupil_nobc_effect_df,
        on=['participant', 'epoch'],
        how='left',
        suffixes=('', '_nobc')
    )
    .merge(
        pupil_bc_effect_df,
        on=['participant', 'epoch'],
        how='left',
        suffixes=('', '_bc')
    )
)

# Quartiling participants based on their average tonic pupil effect

quartile_df = (
    merged_df
    .groupby('participant', group_keys=False)
    .apply(group_data_utils.assign_quartile_group, new_column_name='pupil_group', quartile_column='avg_pupil_effect')
)

quartiles = quartile_df['pupil_group'].unique()

# Checking the number of trials per participant per pupil group
trial_counts_wide = quartile_df.groupby(['participant', 'pupil_group']).size().unstack('pupil_group')

# Computing the mean of the relevant variables for each participant and pupil group
participant_quartile_means = quartile_df.groupby(['participant', 'pupil_group'])[['cpp', 'prior_conf', 'avg_pupil_effect_bc']].mean()

group_quartile_means = participant_quartile_means.groupby('pupil_group').mean()
group_quartile_sds = participant_quartile_means.groupby('pupil_group').std()

# One-way ANOVA to test for differences in the means of the relevant variables across pupil groups
anova_tables = []

for var in ['cpp', 'prior_conf', 'avg_pupil_effect_bc']:
    aov = pg.rm_anova(
        data=participant_quartile_means.reset_index(),
        dv=var,
        within='pupil_group',
        subject='participant',
        detailed=True,
        correction=True
    )

    aov['variable'] = var
    anova_tables.append(aov)

anova_results_df = pd.concat(anova_tables, ignore_index=True)

# Running uncertainty - learning rate regression analysis for each pupil group
df_dict = {}
learning_rate_analysis_dict = {}

for pupil_group in quartile_df['pupil_group'].unique():
    group_data = quartile_df[quartile_df['pupil_group'] == pupil_group]

    df_dict[pupil_group] = group_data


for key, df in df_dict.items():

    # create dictionary for this pupil group
    learning_rate_analysis_dict[key] = {}

    for participant in df['participant'].unique():

        print(f"Running regression for participant {participant} in pupil group {key}...")
        # subset participant data first
        participant_df = df[df['participant'] == participant]

        regression_df = participant_df[
            ['cleaned_learning_rate_estimate', 'prior_conf', 'cpp']
        ].copy()

        df_clean = regression_df.dropna(
            subset=['cleaned_learning_rate_estimate']
        ).copy()

        # z-score variables
        df_clean['lr_zscore_estimate'] = stats.zscore(
            df_clean['cleaned_learning_rate_estimate'],
            nan_policy='omit'
        )

        df_clean['conf_zscore'] = stats.zscore(
            df_clean['prior_conf'],
            nan_policy='omit'
        )

        df_clean['cpp_zscore'] = stats.zscore(
            df_clean['cpp'],
            nan_policy='omit'
        )

        # regression
        df_clean['interaction'] = (
            df_clean['conf_zscore'] *
            df_clean['cpp_zscore']
        )

        X = df_clean[
            ['conf_zscore', 'cpp_zscore', 'interaction']
        ]

        y = df_clean['lr_zscore_estimate']

        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        coefficients = model.params.iloc[1:].values
        intercept = model.params.iloc[0]

        p_values = model.pvalues.values
        std_err_mean = model.bse.values
        conf_int = model.conf_int().values

        uncertainty_v_lr_df = pd.DataFrame({
            "pupil_group": key,
            "participant": participant,
            "variable": ["Intercept"] + X.columns[1:].tolist(),
            "coefficient": [intercept] + coefficients.tolist(),
            "p_value": p_values.tolist(),
            "std_err": std_err_mean.tolist(),
            "ci_lower": conf_int[:, 0].tolist(),
            "ci_upper": conf_int[:, 1].tolist()
        })

        # Store participant-specific results
        learning_rate_analysis_dict[key][participant] = {
            'uncert_lr_part_df': df_clean,
            'uncertainty_v_lr_df': uncertainty_v_lr_df,
            'model': model
        }

# Creating a combined dataframe for all participants in each pupil group

excluded_set = set(excluded_participants)

all_quartiles_regress_df = pd.concat(
    [
        learning_rate_analysis_dict[key][participant]['uncertainty_v_lr_df']
        for key in df_dict
        for participant in participant_list
        if participant not in excluded_set
    ],
    ignore_index=True
)

# Running one-way RM ANOVA to test for differences in the regression coefficients across pupil groups
variable_comparison_list = []
post_hoc_comparison_list = []

variables = ['cpp_zscore', 'conf_zscore']

for variable in variables:
    df = all_quartiles_regress_df[all_quartiles_regress_df['variable']==variable]

    aov = pg.rm_anova(data=df, dv='coefficient', within='pupil_group',
                       subject='participant', detailed=True, correction=True)
    
    aov['variable'] = variable
    variable_comparison_list.append(aov)

    if aov['p-GG-corr'][0] < 0.05:
        posthoc = pg.pairwise_tests(data=df, dv='coefficient', within='pupil_group',
                                subject='participant', padjust='fdr_bh', effsize='cohen')

        posthoc['variable'] = variable
        post_hoc_comparison_list.append(posthoc)

comparison_results_df = pd.concat(variable_comparison_list, ignore_index=True)
posthoc_results_df = pd.concat(post_hoc_comparison_list, ignore_index=True)

# Running regression analyses to assess how the association between change point probability and phasic pupil signals were modulated by tonic arousal state

beta_list = []

for key, df in df_dict.items():

    for participant in df['participant'].unique():

        participant_df = df[df['participant'] == participant].dropna(
            subset=['avg_pupil_effect_bc', 'cpp']
        )

        # z-score within participant
        participant_df['cpp_z'] = stats.zscore(
            participant_df['cpp']
        )

        participant_df['pupil_z'] = stats.zscore(
            participant_df['avg_pupil_effect_bc']
        )

        X = participant_df['cpp_z']
        y = participant_df['pupil_z']

        # add intercept
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        beta_df = pd.DataFrame({
            "pupil_group": [key],
            "participant": [participant],
            "intercept": [model.params['const']],
            "cpp_beta": [model.params['cpp_z']],
            "p_value": [model.pvalues['cpp_z']],
            "r_squared": [model.rsquared]
        })

        beta_list.append(beta_df)

cpp_pupil_beta_df = pd.concat(beta_list, ignore_index=True)

# Running one-way RM ANOVA to test for differences in the regression coefficients across pupil groups

cpp_pupil_anova = pg.rm_anova(data=cpp_pupil_beta_df, dv='cpp_beta', within='pupil_group', subject='participant', detailed=True, correction=True)

# T-tests to compare the regression coefficients between pupil groups to zero 

cpp_pupil_ttest_results = []

for group in cpp_pupil_beta_df['pupil_group'].unique():

    group_df = cpp_pupil_beta_df[cpp_pupil_beta_df['pupil_group'] == group]

    avg_beta = group_df['cpp_beta'].mean()
    std_beta = group_df['cpp_beta'].std()

    t_test = stats.ttest_1samp(
        group_df['cpp_beta'],
        0
    )

    cpp_pupil_ttest_results.append({
        'pupil_group': group,
        'mean_beta': avg_beta,
        'std_beta': std_beta,
        't_stat': t_test.statistic,
        'p_value': t_test.pvalue
    })

cpp_pupil_ttest_zero_df = pd.DataFrame(cpp_pupil_ttest_results)

# two-stage FDR correction (Benjamini-Hochberg)
reject, p_fdr, _, _ = multipletests(
    cpp_pupil_ttest_zero_df['p_value'],
    alpha=0.05,
    method='fdr_tsbh'
)

cpp_pupil_ttest_zero_df['p_fdr'] = p_fdr
cpp_pupil_ttest_zero_df['significant_fdr'] = reject

# Collating betas into specific dataframes
cpp_lr_betas = {}
cpp_pupil_betas = {}

for q in quartiles:

    quartile_cpp_lr_betas = (
        all_quartiles_regress_df.loc[
            (all_quartiles_regress_df['pupil_group'] == q) &
            (all_quartiles_regress_df['variable'] == 'cpp_zscore'),
            'coefficient'
        ]
        .values
    )

    quartile_cpp_pupil_betas = (
        cpp_pupil_beta_df.loc[
            cpp_pupil_beta_df['pupil_group'] == q,
            'cpp_beta'
        ]
        .values
    )

    cpp_lr_betas[q] = quartile_cpp_lr_betas
    cpp_pupil_betas[q] = quartile_cpp_pupil_betas

cpp_lr_betas_df = pd.DataFrame(dict(cpp_lr_betas))
cpp_pupil_betas_df = pd.DataFrame(dict(cpp_pupil_betas))

# Saving group level data to .csv files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    "tonic_quartiles_parameter_check_anova": anova_results_df,
    "tonic_quartiles_uncertainty_v_lr_regression": all_quartiles_regress_df,
    "tonic_quartiles_uncertainty_v_lr_regression_anova": comparison_results_df,
    "tonic_quartiles_uncertainty_v_lr_regression_posthoc": posthoc_results_df,
    "tonic_quartiles_cpp_pupil_betas_anova": cpp_pupil_anova,
    "tonic_quartiles_cpp_pupil_betas_ttest": cpp_pupil_ttest_zero_df, 
    "tonic_quartiles_cpp_lr_betas": cpp_lr_betas_df,
    "tonic_quartiles_cpp_pupil_betas": cpp_pupil_betas_df,
    }, 
directory_path=BASE_DIR,
condition_str=CONDITION_STR,
task='Pupil')