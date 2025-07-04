"""
This script runs correlation analysis to better understand individual differences in uncertainty parameter effects on learning rate obtained from the Pupil task in the Adalearn project. For uncertainty effects on learning rate and the pupil signals influenced by the same uncertainty effects, correlations are run against:
- participants' age
- participants' education level
- participants' average update frequency
- participants' estimate correlation with model posterior probability 
- individual volitility estimates
The following scripts need to be run before running this analysis to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics
- adalearn_pupil_group_level_analysis_pupil_assessment
- adalearn_pupil_group_level_analysis_individual_volatility
- adalearn_pupil_group_level_analysis_estimate_analysis
- adalearn_pupil_group_level_analysis_learning_rate_analysis
- adalearn_pupil_group_level_analysis_pupil_v_lr_and_mediation
"""
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_group_level_functions_statistical_analysis as group_stats 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str



dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'exclude_participants_pupil_analysis_df',
        'demographics_df',
        'individual_p_c_df',
        'update_freq_df', 
        'uncertainty_params_on_lr_df',
        'behav_v_model_estimate_corr_df',
        'average_bc_pupil_effect_signal',
        'average_nobc_pupil_effect_signal',
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')


# Extract key subsets from loaed dataframes
uncertainty_df = dataframes['uncertainty_params_on_lr_df']
conf_lr_coef = uncertainty_df[uncertainty_df['variable'] == 'conf_zscore']['coefficient']
cpp_lr_coef = uncertainty_df[uncertainty_df['variable'] == 'cpp_zscore']['coefficient']
est_model_corr = dataframes['behav_v_model_estimate_corr_df'][dataframes['behav_v_model_estimate_corr_df']['variable']=='ideal_observer']

demographics = dataframes['demographics_df']
update_freq = dataframes['update_freq_df']
individual_pc = dataframes['individual_p_c_df']

# Variables to correlate against the two uncertainty coefficients
x_vars = {
    'Age': demographics['Age'],
    'EducationLevel': demographics['EducationLevel'],
    'avg_update_percentage': update_freq['avg_update_percentage'],
    'optimal_p_c': individual_pc['optimal_p_c'],
    'est_corr': est_model_corr['correlation']
}

y_vars = {
    'conf_lr_coef': conf_lr_coef,
    'cpp_lr_coef': cpp_lr_coef
}

# Storing raw data in dataframes
raw_lr_cor_data_dict = {}

for x_key, x_data in x_vars.items():
    # Always add or overwrite — this ensures it's added on first encounter
    raw_lr_cor_data_dict[x_key] = x_data.reset_index(drop=True)

for y_key, y_data in y_vars.items():
    # Always add or overwrite — this ensures it's added on first encounter
    raw_lr_cor_data_dict[y_key] = y_data.reset_index(drop=True)

raw_lr_cor_data = pd.DataFrame(raw_lr_cor_data_dict)

# Compute correlations for learning rate uncertainty effects
results_lr = {}
for y_key, y_data in y_vars.items():
    for x_key, x_data in x_vars.items():
        coef, pval = group_stats.corr_with_alt(x_data, y_data)
        results_lr[(x_key, y_key)] = (coef, pval)

# Extract p-values
lr_pvals = [p for (_, p) in results_lr.values()]

# Apply FDR correction
lr_reject, lr_pvals_fdr = fdrcorrection(lr_pvals, alpha=0.05)

# Update results_lr with adjusted p-values
for i, key in enumerate(results_lr.keys()):
    coef, _ = results_lr[key]
    results_lr[key] = (coef, lr_pvals[i], lr_pvals_fdr[i], lr_reject[i])

# Prepare filtered dataframes for pupil signal uncertainty effects
exclude = dataframes['exclude_participants_pupil_analysis_df']['exclude_participant']

filtered_demographics = demographics[~demographics['ParticipantNumber'].isin(exclude)]
filtered_individual_pc = individual_pc[~individual_pc['participant'].isin(exclude)]
filtered_update_freq = update_freq[~update_freq['participant'].isin(exclude)]
filtered_estmodcor = est_model_corr[~est_model_corr['participant'].isin(exclude)]

# Targets for pupil signal
pupil_targets = {
    'bc_pupil': dataframes['average_bc_pupil_effect_signal']['avg_pupil_effect'],
    'nobc_pupil': dataframes['average_nobc_pupil_effect_signal']['avg_pupil_effect']
}

# Predictor variables after filtering
x_vars_filtered = {
    'Age': filtered_demographics['Age'],
    'EducationLevel': filtered_demographics['EducationLevel'],
    'avg_update_percentage': filtered_update_freq['avg_update_percentage'],
    'optimal_p_c': filtered_individual_pc['optimal_p_c'],
    'est_corr': filtered_estmodcor['correlation']
}

# Storing raw pupil correlations in dataframes
raw_pupil_cor_data_dict = {}

for x_key, x_data in x_vars_filtered.items():
    # Always add or overwrite — this ensures it's added on first encounter
    raw_pupil_cor_data_dict[x_key] = x_data.reset_index(drop=True)

for y_key, y_data in pupil_targets.items():
    # Always add or overwrite — this ensures it's added on first encounter
    raw_pupil_cor_data_dict[y_key] = y_data.reset_index(drop=True)

raw_pupil_cor_data = pd.DataFrame(raw_pupil_cor_data_dict)

# Compute correlations for pupil effects
results_pupil = {}
for y_key, y_data in pupil_targets.items():
    for x_key, x_data in x_vars_filtered.items():
        coef, pval = group_stats.corr_with_alt(x_data, y_data)
        results_pupil[(x_key, y_key)] = (coef, pval)

pupil_pvals = [p for (_, p) in results_pupil.values()]

pupil_reject, pupil_pvals_fdr = fdrcorrection(pupil_pvals, alpha=0.05)

for i, key in enumerate(results_pupil.keys()):
    coef, _ = results_pupil[key]
    results_pupil[key] = (coef, pupil_pvals[i], pupil_pvals_fdr[i], pupil_reject[i])

# Collate correlation coefficients into dataframes
lr_effects_df = group_data_utils.results_dict_to_df(results_lr)
pupil_effects_df = group_data_utils.results_dict_to_df(results_pupil)

# Saving group level data to .csv files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    "uncertainty_lr_effects_cor_df": lr_effects_df,
    "uncertainty_pupil_effects_cor_df": pupil_effects_df,
    "uncertainty_lr_cor_data": raw_lr_cor_data,
    "uncertainty_pupil_cor_data": raw_pupil_cor_data
    }, 
directory_path=base_dir,
condition_str=condition_str,
task='Pupil')