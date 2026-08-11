"""
This script investigates whether the correlation between learning rate and change-point probability (cpp) or confidence can be increased by allowing trial-specific temporal shifts of the learning rate. The main approach is to optimize the alignment of the update sequeshiftnce with respect to the sequence of cpp (or confidence) and error for each participant, then measure the resulting correlation. The script also estimates a null distribution of correlations by shuffling cpp (or confidence) and error values with respect to update across participants, enabling statistical comparison.

Key functionalities:
- Computes original correlations between learning rate and cpp (and prior confidence) per participant.
- Optimizes trial-wise shifts in the update sequence to maximize the correlation between learning rate and cpp (or confidence).
- Estimates a null distribution of optimized correlations by randomizing cpp (or confidence) and error values across participants.
- Performs t-tests to assess whether the optimized correlation exceeds the null distribution.

Data requirements:
- Expects a CSV file with participant-level trial data, including columns for 'participant', 'cpp', 'estimate_errors', 'estimate_updates', 'learning_rate_estimate', 'prior_conf', and 'task_session'.
"""

import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from copy import deepcopy

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_comp_utilities as group_comp_utils 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR, CONDITION_STR



# define independent variables to analyze
INDEP_VARS = ['cpp', 'prior_conf']
REMOVE = True

dataframes = group_data_utils.load_sample_csvs(
    directory_path=BASE_DIR,
    substrings=[
        'computational_var_df', 
        'update_freq_df',
        ],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

shift_data = dataframes['computational_var_df'][['participant', 'task_session', 'outcome_num', 'outcome',
                                                 'estimate', 'estimate_updates', 'estimate_errors', 'was_updated_estimate',
                                                 'learning_rate_estimate', 'cleaned_learning_rate_estimate',
                                                 'ideal_observer', 'cpp', 'prior_conf']].copy(deep=True)

# measure original correlations
rhos_pearson, rhos_sper = {}, {}
for indep_var in INDEP_VARS:
    rhos_pearson[indep_var] = []
    rhos_sper[indep_var] = []
    for sub_id in shift_data['participant'].unique():
        sub_data = shift_data.loc[shift_data['participant'] == sub_id].copy(deep=True)
        rhos_pearson[indep_var].append(
            np.corrcoef(sub_data['learning_rate_estimate'], sub_data[indep_var])[0, 1])
        corr, pval = stats.spearmanr(sub_data['learning_rate_estimate'], sub_data[indep_var])
        rhos_sper[indep_var].append(corr)

# shift update to improve the correlation
shifts, corrs, null_means, null_shift_vectors, ttest_results = {}, {}, {}, {}, {}
all_results = []
group_summary_list = []
null_vectors = []

for indep_var in INDEP_VARS:
    print(f'\n\nProcessing latent variable: {indep_var}\n') 

    results = []  # collect all participant-level shift results
    participants = sorted(shift_data['participant'].unique())

    for sub_id in participants:
        print(f'Processing participant {sub_id}...')
        # Run your participant function
        res = group_comp_utils.process_participant_shift_update(sub_id, shift_data, indep_var, remove=REMOVE, error_col='estimate_errors', update_col='estimate_updates')
        results.append(res)

    # Unpack results (each res should be a tuple: (shift, corr, null_mean))
    shifts[indep_var], corrs[indep_var], null_means[indep_var], null_shift_vectors[indep_var] = zip(*results)

    # Integrate optimal shifts into main dataframe
    shift_data[f'optimal_shift_{indep_var}'] = np.nan
    for sub_id, shift_array in zip(participants, shifts[indep_var]):
        mask = shift_data['participant'] == sub_id
        if mask.sum() == len(shift_array):
            shift_data.loc[mask, f'optimal_shift_{indep_var}'] = shift_array
        else:
            print(f"Length mismatch for participant {sub_id}: "
                  f"{mask.sum()} trials vs {len(shift_array)} in shift array")

    # Store per-participant results in a dataframe
    participant_results = pd.DataFrame({
        'participant': participants,
        'latent_var': indep_var,
        'corr_original': rhos_pearson[indep_var],
        'corr_after_shift': corrs[indep_var],
        'null_mean': null_means[indep_var],
    })
    participant_results['corr_diff'] = (
        participant_results['corr_after_shift'] - participant_results['null_mean']
    )

    all_results.append(participant_results)

    # Compute group-level stats
    ttest = stats.ttest_1samp(np.array(corrs[indep_var])
                        - np.array(null_means[indep_var]), popmean=0
    )
    
    ttest_results[indep_var] = ttest

    # Storing null vectors
    null_shifts = []

    for participant, participant_null_vectors in zip(
            participants,
            null_shift_vectors[indep_var]):

        for null_iter, shift_vector in enumerate(participant_null_vectors):

            for trial, shifted in enumerate(shift_vector):

                null_shifts.append({
                    "participant": participant,
                    "latent_var": indep_var,
                    "null_iter": null_iter,
                    "trial": trial,
                    "shifted": shifted
                })

    null_shift_df = pd.DataFrame(null_shifts)
    null_vectors.append(null_shift_df)

    group_summary_list.append(pd.DataFrame([{
        'latent_var': indep_var,
        't_stat': ttest.statistic,
        'p_value': ttest.pvalue,
        'mean_corr_after_shift': participant_results['corr_after_shift'].mean(),
        'mean_null_corr': participant_results['null_mean'].mean(),
        'mean_original_corr': participant_results['corr_original'].mean(),
    }]))

    # Print summary info
    print(f'\n\nLatent variable: {indep_var}\n')
    print(f'Original corr mean = {np.mean(rhos_pearson[indep_var]):0.3f} ± '
          f'{pd.Series(rhos_pearson[indep_var]).sem():0.3f}')
    print(f'After shift mean   = {np.mean(corrs[indep_var]):0.3f} ± '
          f'{pd.Series(corrs[indep_var]).sem():0.3f}')
    print(f'Null mean          = {np.mean(null_means[indep_var]):0.3f} ± '
          f'{pd.Series(null_means[indep_var]).sem():0.3f}')

results_df = pd.concat(all_results, ignore_index=True)
group_summary_df = pd.concat(group_summary_list, ignore_index=True)
null_vectors_df = pd.concat(null_vectors, ignore_index=True)

# Average estimates shifted for each variable
print(f"\n\nOptimal estimate shift for CPP: {shift_data['optimal_shift_cpp'].mean():.2f}")
print(f"Optimal estimate shift for prior uncertainty: {shift_data['optimal_shift_prior_conf'].mean():.2f}")

# Agreement in shifts between cpp and prior uncertainty
agreement_list = []

for participant in shift_data['participant'].unique():
    participant_df = shift_data[shift_data['participant'] == participant].copy()
    
    # Compare 0/1 arrays elementwise
    agreement = (participant_df['optimal_shift_cpp'].values ==
                 participant_df['optimal_shift_prior_conf'].values)
    
    # Compute proportion of agreement for this participant
    agreement_mean = np.mean(agreement)
    agreement_list.append(agreement_mean)

agreement_series = pd.Series(agreement_list)

print(f"\nShift agreement between CPP and prior confidence:")
print(f"Mean: {np.mean(agreement_list):.2f}, SEM: {pd.Series(agreement_list).sem():.2f}")
print(f"Min: {np.min(agreement_list):.2f}, Max: {np.max(agreement_list):.2f}")

# Comparion of shifted correlation compared to null across variables
print('\nComparison of shifted correlation relative to null across variables:') 
cpp_diff_data = results_df[results_df['latent_var'] == 'cpp']['corr_diff']
prior_conf_diff_data = results_df[results_df['latent_var'] == 'prior_conf']['corr_diff']

# Paired t-test
t_stat_diff, p_val_diff = stats.ttest_rel(np.array(cpp_diff_data), np.array(prior_conf_diff_data))

print(f'Paired t-test between cpp and prior confidence correlation differences: t = {t_stat_diff:.3f}, p = {p_val_diff:.3f}')

print('\nComparison of absolute shifted correlation across variables:')
cpp_shift_corr_data = results_df[results_df['latent_var'] == 'cpp']['corr_after_shift']
prior_conf_shift_corr_data = results_df[results_df['latent_var'] == 'prior_conf']['corr_after_shift']

# Paired t-test
t_stat_shift, p_val_shift = stats.ttest_rel(np.array(cpp_shift_corr_data), np.array(prior_conf_shift_corr_data))

print(f'Paired t-test between cpp and prior confidence shifted correlations: t = {t_stat_shift:.3f}, p = {p_val_shift:.3f}')

# Store in DataFrame
ttest_df = pd.DataFrame({
    'test': ['paired_ttest_cpp_vs_prior_conf_diff', 'paired_ttest_cpp_vs_prior_conf_shifted_corr'],
    't_stat': [t_stat_diff, t_stat_shift],
    'p_value': [p_val_diff, p_val_shift],
    'n_participants': [len(cpp_diff_data), len(cpp_shift_corr_data)]
})

for indep_var in INDEP_VARS:

    indep_var_clean = indep_var if indep_var == 'cpp' else 'prior_conf'

    # initialize empty output columns
    shift_data[f'estimate_update_shifted_{indep_var_clean}'] = np.nan
    shift_data[f'learning_rate_shifted_{indep_var_clean}'] = np.nan

    # process participant-by-participant
    for sub_id in shift_data['participant'].unique():

        sub_mask = shift_data['participant'] == sub_id
        sub_data = shift_data.loc[sub_mask]

        # extract individual participant vectors
        updates = sub_data['estimate_updates'].to_numpy()
        errors = sub_data['estimate_errors'].to_numpy()
        shift_vec = sub_data[f'optimal_shift_{indep_var_clean}'].to_numpy().astype(int)

        # --- Apply the shift (new correct function)
        updates_shifted = group_comp_utils.apply_shift(updates, shift_vec, remove=REMOVE)

        # compute learning rate
        lr_shifted = updates_shifted / errors

        # --- Store results back into the DataFrame
        shift_data.loc[sub_mask, f'estimate_update_shifted_{indep_var_clean}'] = updates_shifted
        shift_data.loc[sub_mask, f'learning_rate_shifted_{indep_var_clean}'] = lr_shifted

# Correlation between optimization for cpp and prior uncertainty
fraction_shift_df_list = []

for participant in shift_data['participant'].unique():

    participant_df = shift_data[shift_data['participant'] == participant].copy()

    cpp_shift_mean = participant_df['optimal_shift_cpp'].mean()
    prior_conf_shift_mean = participant_df['optimal_shift_prior_conf'].mean()

    fraction_shift_df = pd.DataFrame({
        'participant': [participant],
        'cpp_shift_mean': [cpp_shift_mean],
        'prior_conf_shift_mean': [prior_conf_shift_mean]
    })

    fraction_shift_df_list.append(fraction_shift_df)    

group_fraction_shift_df = pd.concat(fraction_shift_df_list, ignore_index=True)

# Compute correlation and p-value
fraction_corr, fraction_p_val = stats.pearsonr(
    group_fraction_shift_df['cpp_shift_mean'],
    group_fraction_shift_df['prior_conf_shift_mean']
)
print(f'\nCorrelation between fraction of trials shifted for cpp and prior confidence: r = {fraction_corr:.3f}, p = {fraction_p_val:.3e}')

# Check correlation between fraction shifts and average update frequency
fraction_shift_corr_output = []

columns = ['cpp_shift_mean', 'prior_conf_shift_mean']

for col in columns:
    if col == 'cpp_shift_mean':
        latent_var = 'cpp_shift'
    else:
        latent_var = 'prior_conf_shift'

    fraction_var_df = group_fraction_shift_df[['participant', col]].copy()

    fraction_shift_corr_df = pd.merge(
        fraction_var_df,
        dataframes['update_freq_df'][['participant', 'avg_update_percentage']],
        on='participant'
    )

    r, p = stats.pearsonr(fraction_shift_corr_df['avg_update_percentage'], fraction_shift_corr_df[col])

    corr_output_df = pd.DataFrame({
        'latent_var': [latent_var],
        'fraction_shift_corr': [r],
        'fraction_shift_pval': [p]
    })

    fraction_shift_corr_output.append(corr_output_df)

fraction_shift_corr_df = pd.concat(fraction_shift_corr_output, ignore_index=True)

print('\nCorrelation between average update frequency and fraction of trials shifted:')
for var in fraction_shift_corr_df['latent_var'].unique():
    var_df = fraction_shift_corr_df[fraction_shift_corr_df['latent_var'] == var]
    print(f"{var}: r = {var_df['fraction_shift_corr'].values[0]:.3f}, p = {var_df['fraction_shift_pval'].values[0]:.3f}")

# Check correlation between shifted learning rate and average update frequency 
update_shift_corr_output = []

for var in results_df['latent_var'].unique():

    latent_var_df = results_df[results_df['latent_var'] == var].copy()

    update_shift_corr_df = pd.merge(
        latent_var_df[['participant', 'corr_diff']],
        dataframes['update_freq_df'][['participant', 'avg_update_percentage']],
        on='participant'
    )

    r, p = stats.pearsonr(update_shift_corr_df['avg_update_percentage'], update_shift_corr_df['corr_diff'])

    corr_output_df = pd.DataFrame({
        'latent_var': [var],
        'update_shift_corr': [r],
        'update_shift_pval': [p]
    })

    update_shift_corr_output.append(corr_output_df)

update_shift_corr_df = pd.concat(update_shift_corr_output, ignore_index=True)

print('\nCorrelation between average update frequency and improvement in correlation after shift:')
for var in update_shift_corr_df['latent_var'].unique():
    var_df = update_shift_corr_df[update_shift_corr_df['latent_var'] == var]
    print(f"{var}: r = {var_df['update_shift_corr'].values[0]:.3f}, p = {var_df['update_shift_pval'].values[0]:.3f}")

# Collecting null shifted vectors
null_shift_summary_df = pd.concat(
    [df for df in null_shift_dfs.values()],
    ignore_index=True
)

for col in null_shift_summary_df.columns:
    null_shift_summary_df[col] = null_shift_summary_df[col].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
    )

# Saving files to .csv
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'shifted_estimate_results': shift_data,
    'shift_optimization_results': results_df,
    'shift_optimization_group_summary': group_summary_df, 
    'between_variable_shift_optimization': group_fraction_shift_df,
    'fraction_shift_correlation': fraction_shift_corr_df,
    'update_shift_correlation': update_shift_corr_df,
    'lag_corr_difference_paired_ttest': ttest_df,
    'null_shifted_vectors_summary': null_shift_summary_df
    },
    directory_path=BASE_DIR,
    condition_str=CONDITION_STR,
    task='Pupil')