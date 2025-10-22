import numpy as np
import pandas as pd


N_ITER = 10
N_NULL_ITER = 30

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
    print(f'optimizing shift update...')
    shift, corr = optimize_shift_update(
                                 sub_data[indep_var].to_numpy(),
                                 sub_data['estimate_errors'].to_numpy(),
                                 sub_data['estimate_updates'].to_numpy(),
                                 sub_data.shape[0]/sub_data['task_session'].max(),
                                 remove)
    
    # get null distribution of correlation after shift
    null_dist_corr = null_optimized_corr_update(data, sub_id, indep_var, remove)
    
    return shift, corr, np.mean(null_dist_corr)

