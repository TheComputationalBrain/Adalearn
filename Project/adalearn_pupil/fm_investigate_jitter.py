"""
This script investigates whether the correlation between learning rate and change-point
probability (cpp) or confidence can be increased by allowing trial-specific temporal
shifts of the learning rate. The main approach is to optimize the alignment of the
update sequence with respect to the sequence of cpp (or confidence) and error for each
participant, then measure the resulting correlation. The script also estimates a null
distribution of correlations by shuffling cpp (or confidence) and error values with
respect to update across participants, enabling statistical comparison.

Key functionalities:
- Computes original correlations between learning rate and cpp (and prior confidence)
    per participant.
- Optimizes trial-wise shifts in the update sequence to maximize the correlation
    between learning rate and cpp (or confidence).
- Estimates a null distribution of optimized correlations by randomizing cpp (or
    confidence) and error values across participants.
- Performs t-tests to assess whether the optimized correlation exceeds the null
    distribution.
- Visualizes original, optimized, and null correlations across participants.

Dependencies: pandas, numpy, scipy, joblib, matplotlib

Data requirements:
- Expects a CSV file with participant-level trial data, including columns for
    'participant', 'cpp', 'estimate_errors', 'estimate_updates', 'learning_rate_estimate',
    'prior_conf', and 'task_session'.

Usage:
- Adjust DATA_DIR and file paths as needed.
- Run the script to perform the analysis and generate plots.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


DATA_DIR = '~/nasShare/projects/ADALEARN/adalearn_pupil'
N_ITER = 10
N_NULL_ITER = 30
INDEP_VARS = ['cpp', 'prior_conf']

data = pd.read_csv(os.path.join(DATA_DIR, 'group_csv_files',
    'sample_level_analysis-ses_01-task_Adalearn_Pupil-computational_var_df.csv'))

# turn confidence into uncertainty
data['prior_conf'] = -data['prior_conf']

def compute_movable_position(array_len, sequence_len):
    """
    Get indices of position that can be moved
    
    The last position of each sequence cannot be moved because there is no next trial
    to take the value from.
    """
    return np.concatenate([np.arange(0, sequence_len - 1) + k*sequence_len
                           for k in np.arange(0, array_len/sequence_len)]).astype(int)


def shift_values(values, shift):
    """Shift the values as indicated in the shift
    
    When shift[i]=j, values[i] is replaced by values[i+j]
    """

    # prevent changes in the original vector
    values = values.copy()

    # shift the value
    for shift_val in np.unique(shift):
        if shift_val > 0:
            values[shift==shift_val] = values[
                np.flatnonzero(shift==shift_val)+int(shift_val)]

    return values


def compute_correlation_with_shifted_update(indep_var, error, update, shift, remove):
    """Compute correlated of learning rate and indep_var with shifted update"""

    update_shifted = shift_values(update, shift)
    lr_shifted = update_shifted / error

    if remove:
        # set to nan the values of update that have been shifted to a previous trial
        for shift_val in shift.unique():
            if shift_val > 0:
                lr_shifted[np.flatnonzero(shift==shift_val) + shift_val] = np.nan

    # use pandas to handle nans
    return pd.Series(indep_var).corr(pd.Series(lr_shifted))



def optimize_shift_update(indep_var, error, update, sequence_len, remove, n=N_ITER):
    """Return the optimized shift of update and corresponding correlation"""

    # get positions with can be shifted (not position 0, 1, and 2 in each sequence)
    shift = np.zeros(len(indep_var))
    allowed_shift_positions = compute_movable_position(len(shift), sequence_len)

    # optimize the shift through several iterations
    for _ in range(n):
        for idx in np.random.permutation(allowed_shift_positions):
            ref_corr = compute_correlation_with_shifted_update(indep_var, error,
                                                               update, shift, remove)
            # change the value of the shift
            shift[idx] = int(not shift[idx])
            if (compute_correlation_with_shifted_update(indep_var, error,
                                                        update, shift, remove)
                <= ref_corr):
                # revert the shift to the previous value
                shift[idx] = int(not shift[idx])
    return shift, compute_correlation_with_shifted_update(indep_var, error, update, shift, remove)



def null_optimized_corr_update(data, target_id, indep_var, remove, n=N_ITER, null_n=N_NULL_ITER):
    """Compute the null distribution of the correlation using the shifted update
    
    This null distribution is estimated by replacing indep_var and error values of the
    target participant with those of a randomly sampled (other) participant.
    """
    target_sub_data = data.loc[data['participant'] == target_id]
    null_ids = list(set(data['participant'].unique()) - {target_id})
    null_dist_corr = []

    # estimate the null distribution by sampling other pariticpants
    for id in np.random.choice(null_ids,
                               size=null_n, replace=False):
        null_sub_data = data.loc[data['participant'] == id].copy(deep=True)
        shift, corr = optimize_shift_update(
                            null_sub_data[indep_var].to_numpy(),
                            null_sub_data['estimate_errors'].to_numpy(),
                            target_sub_data['estimate_updates'].to_numpy(),
                            target_sub_data.shape[0]/target_sub_data['task_session'].max(),
                            remove,
                            n)
        null_dist_corr.append(corr)
    return null_dist_corr


def process_participant_shift_update(sub_id, data, indep_var, remove=False):
    """Process correlation analysis for a single participant with optimized shift of the update."""
    sub_data = data.loc[data['participant'] == sub_id].copy(deep=True)
    
    # get within participant correlation after shift
    shift, corr = optimize_shift_update(
                                 sub_data[indep_var].to_numpy(),
                                 sub_data['estimate_errors'].to_numpy(),
                                 sub_data['estimate_updates'].to_numpy(),
                                 sub_data.shape[0]/sub_data['task_session'].max(),
                                 remove)
    
    # get null distribution of correlation after shift
    null_dist_corr = null_optimized_corr_update(data, sub_id, indep_var, remove)
    
    return shift, corr, np.mean(null_dist_corr)


# measure original correlations
rhos_pearson, rhos_sper = {}, {}
for indep_var in INDEP_VARS:
    indep_var_name = indep_var if indep_var == 'cpp' else 'prior uncertainty'
    rhos_pearson[indep_var_name] = []
    rhos_sper[indep_var_name] = []
    for sub_id in data['participant'].unique():
        sub_data = data.loc[data['participant'] == sub_id].copy(deep=True)
        rhos_pearson[indep_var_name].append(
            np.corrcoef(sub_data['learning_rate_estimate'], sub_data[indep_var])[0, 1])
        corr, pval = spearmanr(sub_data['learning_rate_estimate'], sub_data[indep_var])
        rhos_sper[indep_var_name].append(corr)

## shift update to improve the correlation
shifts, corrs, null_means = {}, {}, {}
for indep_var in INDEP_VARS:
    print(f'\n\n### INDEP_VAR: {indep_var} ###\n') 
    indep_var_name = indep_var if indep_var == 'cpp' else 'prior uncertainty'
    results = Parallel(n_jobs=-1)(
        delayed(process_participant_shift_update)(sub_id, data, indep_var)
        for sub_id in data['participant'].unique()
    )

    # Extract results
    shifts[indep_var_name], corrs[indep_var_name], null_means[indep_var_name] = zip(*results)

    # stats
    ttest = ttest_1samp(np.array(corrs[indep_var_name])
                        - np.array(null_means[indep_var_name]), popmean=0)

    # report stats
    print(f'correlation learning rate vs. {indep_var_name}, original: ' +
          f'mean={np.mean(rhos_pearson[indep_var_name]):0.3f} ' +
          f'+/- {pd.Series(rhos_pearson[indep_var_name]).sem():0.3f} SEM')
    print(f'correlation learning rate vs. {indep_var_name} after shift: ' +
          f'mean={np.mean(corrs[indep_var_name]):0.3f} ' +
          f'+/- {pd.Series(corrs[indep_var_name]).sem():0.3f} SEM')
    print(f'correlation learning rate vs. {indep_var_name} null: ' +
          f'mean={np.mean(null_means[indep_var_name]):0.3f} ' +
          f'+/- {pd.Series(null_means[indep_var_name]).sem():0.3f} SEM')
    print(f'ttest: t={ttest.statistic:0.3f}, p={ttest.pvalue:0.1e}')

    # plot
    plt.figure()
    plt.plot(rhos_pearson[indep_var_name], label='original')
    plt.plot(np.array(corrs[indep_var_name]), label='shifted')
    plt.plot(np.array(null_means[indep_var_name]), label='shifted null')
    plt.xlabel('participants')
    plt.ylabel(f'correlation learning rate vs. {indep_var_name}')
    plt.title(f'shift update \n ttest: p={ttest.pvalue:0.3f}')
    plt.legend()

# plot shifts
plt.figure()
plt.subplot(1, 2, 1)
for indep_var in INDEP_VARS:
    indep_var_name = indep_var if indep_var == 'cpp' else 'prior uncertainty'
    plt.plot(np.mean(shifts[indep_var_name], axis=1), label=indep_var_name)
plt.legend()
plt.xlabel('participants')
plt.ylabel('fraction of shifted trials')
plt.subplot(1, 2, 2)
plt.plot(np.mean(shifts['cpp'], axis=1), np.mean(shifts['prior uncertainty'], axis=1),
         '.', label='one participant')
plt.legend()
plt.xlabel('fraction of shifted trials (cpp)')
plt.ylabel('fraction of shifted trials (prior uncertainty)')
between_subj_corr = np.corrcoef(np.mean(shifts["cpp"], axis=1),
                         np.mean(shifts["prior uncertainty"], axis=1))[0,1]
plt.title(f'correlation {between_subj_corr:0.3f}')
plt.tight_layout()

# Errorbar plot of paired difference between shifted and shifted null correlation
labels = []
diffs = []
sems = []
for indep_var in INDEP_VARS:
    indep_var_name = indep_var if indep_var == 'cpp' else 'prior uncertainty'
    diff = np.array(corrs[indep_var_name]) - np.array(null_means[indep_var_name])
    diffs.append(np.mean(diff))
    sems.append(pd.Series(diff).sem())
    labels.append(indep_var_name)

plt.figure()
plt.bar(labels, diffs, yerr=sems, capsize=8, color=['C0','C1'])
plt.ylabel('Shifted correlation - shifted null correlation')
plt.title('Paired difference: shifted vs. shifted null correlation')
plt.tight_layout()

# agreement in shifts between cpp and prior uncertainty
shift_aggrement = []
for shift_cpp, shift_unc in zip(shifts['cpp'], shifts['prior uncertainty']):
    shift_aggrement.append(np.mean(shift_cpp == shift_unc))
print('### Shift agreement between cpp and prior uncertainty ###')
print(f'Mean: {np.mean(shift_aggrement):0.3f}, SEM: {pd.Series(shift_aggrement).sem():0.3f}')
print(f'Min: {np.min(shift_aggrement):0.3f}, Max: {np.max(shift_aggrement):0.3f}')

plt.show()
