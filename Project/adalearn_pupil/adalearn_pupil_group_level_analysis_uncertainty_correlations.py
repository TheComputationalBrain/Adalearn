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

import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from statsmodels.stats.multitest import multipletests

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils 
import adalearn_analysis_group_level_functions_statistical_analysis as group_stats
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR



dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR, 
    substrings=[
        'exclude_participants_pupil_analysis_df',
        'demographics_df',
        'individual_p_c_df',
        'update_freq_df', 
        'uncertainty_params_on_lr_df',
        'behav_v_model_estimate_corr_df',
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

foucault_meyniel_dataframes = group_data_utils.load_sample_csvs(
    directory_path=Path(BASE_DIR, 'foucault_meyniel_2024_reanalysis'),
    substrings=[
        'foucault_meyniel_2024_comp_params_df',
        'foucault_meyniel_2024_estimate_v_model_corr_df',
        'foucault_meyniel_2024_update_freq_df',
        'foucault_meyniel_2024_ind_volatility_df',
        'foucault_meyniel_2024_uncert_lr_regress_coefs_df'
    ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-'
)

# Extract key subsets from loaded dataframes
# Present study
uncertainty_df = dataframes['uncertainty_params_on_lr_df']
conf_lr_coef = uncertainty_df[uncertainty_df['variable'] == 'conf_zscore']['coefficient'].copy()
cpp_lr_coef = uncertainty_df[uncertainty_df['variable'] == 'cpp_zscore']['coefficient'].copy()
est_model_corr = dataframes['behav_v_model_estimate_corr_df'][dataframes['behav_v_model_estimate_corr_df']['variable']=='ideal_observer'].copy()
update_freq = dataframes['update_freq_df'].copy()
individual_pc = dataframes['individual_p_c_df'].copy()

present_study_var_df = update_freq.copy()
present_study_var_df['est_model_corr'] = est_model_corr['correlation'].values
present_study_var_df['individual_pc'] = individual_pc['optimal_p_c']
present_study_var_df['conf_lr_coef'] = conf_lr_coef.values
present_study_var_df['cpp_lr_coef'] = cpp_lr_coef.values
present_study_var_df['study'] = 'present_study'

# Foucault & Meyniel 2024 study
foucault_meyniel_uncertainty_df = foucault_meyniel_dataframes['foucault_meyniel_2024_uncert_lr_regress_coefs_df'].copy()
foucault_meyniel_conf_lr_coef = foucault_meyniel_uncertainty_df[foucault_meyniel_uncertainty_df['variable'] == 'conf_zscore']['coefficient'].copy()
foucault_meyniel_cpp_lr_coef = foucault_meyniel_uncertainty_df[foucault_meyniel_uncertainty_df['variable'] == 'cpp_zscore']['coefficient'].copy()
foucault_meyniel_est_model_corr = foucault_meyniel_dataframes['foucault_meyniel_2024_estimate_v_model_corr_df'][foucault_meyniel_dataframes['foucault_meyniel_2024_estimate_v_model_corr_df']['variable']=='ideal_observer'].copy()
foucault_meyniel_update_freq = foucault_meyniel_dataframes['foucault_meyniel_2024_update_freq_df'].copy()
foucault_meyniel_individual_pc = foucault_meyniel_dataframes['foucault_meyniel_2024_ind_volatility_df'].copy()


average_update_freq_list = []

for participant in foucault_meyniel_update_freq['participant'].unique():
    task_sessions = foucault_meyniel_update_freq[foucault_meyniel_update_freq['participant'] == participant]

    average_update_freq_df = pd.DataFrame({
        'participant': [participant],
        'avg_update_percentage': [task_sessions['update_percentage'].mean()]
    })

    average_update_freq_list.append(average_update_freq_df)

foucault_meyniel_average_update_freq = pd.concat(average_update_freq_list, ignore_index=True)

foucault_meyniel_var_df = foucault_meyniel_average_update_freq.copy()
foucault_meyniel_var_df['est_model_corr'] = foucault_meyniel_est_model_corr['correlation'].values
foucault_meyniel_var_df['individual_pc'] = foucault_meyniel_individual_pc['optimal_p_c']
foucault_meyniel_var_df['conf_lr_coef'] = foucault_meyniel_conf_lr_coef.values
foucault_meyniel_var_df['cpp_lr_coef'] = foucault_meyniel_cpp_lr_coef.values
foucault_meyniel_var_df['study'] = 'foucault_meyniel_2024'

# Combine both DataFrames for comparison
combined_var_df = pd.concat([present_study_var_df, foucault_meyniel_var_df], ignore_index=True)

# Store dataframes in dictionary
study_dataframe_dict = {
    'present_study': present_study_var_df,
    'foucault_meyniel_2024': foucault_meyniel_var_df,
    'combined': combined_var_df
}

# Comparison of average variables between studies
print("Comparing latent uncertainty parameters between present study and Foucault & Meyniel 2024 study...")
exclude_cols = ['participant', 'study']

cols_to_test = [col for col in present_study_var_df.select_dtypes(include='number').columns 
                if col not in exclude_cols]

ttest_results = []

for col in cols_to_test:
    data1 = present_study_var_df[col]
    data2 = foucault_meyniel_var_df[col]

    # Run independent t-test
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)  
    
    # Degrees of freedom for Welch's t-test
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    df = (s1/n1 + s2/n2)**2 / (((s1/n1)**2 / (n1 - 1)) + ((s2/n2)**2 / (n2 - 1)))
    
    ttest_results.append({
        'column': col,
        't_stat': t_stat,
        'p_value': p_val,
        'df': df,
        'n1': n1,
        'n2': n2,
        'mean1': data1.mean(),
        'mean2': data2.mean()
    })

var_ttest_df = pd.DataFrame(ttest_results)

reject, pvals_corrected, _, _ = multipletests(var_ttest_df['p_value'], 
                                              alpha=0.05, 
                                              method='fdr_bh')

# Add adjusted p-values 
var_ttest_df['p_value_adj'] = pvals_corrected
var_ttest_df['reject_null'] = reject

# Correlations within each study dataframe
print("Running correlations on latent uncertainty parameters...")
correlation_results = []

target_vars = ['conf_lr_coef', 'cpp_lr_coef']
exclude_vars = target_vars + ['participant', 'study']  

for key, df in study_dataframe_dict.items():
    for target in target_vars:
        for col in df.columns:
            if col not in exclude_vars:
                x = df[col].dropna()
                y = df[target].dropna()  # correlate within same study dataframe

                # Align indices after dropping NaNs
                common_idx = x.index.intersection(y.index)
                x = x.loc[common_idx]
                y = y.loc[common_idx]

                if len(common_idx) > 1:  # need at least 2 points
                    r, p = stats.pearsonr(x, y)
                    correlation_results.append({
                        'study': key,
                        'target': target,
                        'predictor': col,
                        'r': r,
                        'p_value': p,
                        'n': len(common_idx)
                    })

correlation_df = pd.DataFrame(correlation_results)

# Apply multiple comparisons correction per study
adjusted_results = []
for study, subdf in correlation_df.groupby('study'):
    reject, pvals_corrected, _, _ = multipletests(subdf['p_value'], 
                                                  alpha=0.05, 
                                                  method='fdr_bh')
    subdf = subdf.copy()
    subdf['p_value_adj'] = pvals_corrected
    subdf['reject_null'] = reject
    adjusted_results.append(subdf)

correlation_df = pd.concat(adjusted_results, ignore_index=True)

# Saving group level data to .csv files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    "present_study_corr_var_df": present_study_var_df,
    "foucault_meyniel_2024_corr_var_df": foucault_meyniel_var_df,
    "present_v_foucault_meyniel_var_ttest_df": var_ttest_df,
    "uncertainty_determinants_corr_df": correlation_df,
    }, 
directory_path=BASE_DIR,
condition_str=CONDITION_STR,
task='Pupil')