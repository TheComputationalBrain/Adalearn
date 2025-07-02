import numpy as np
import scipy.stats
import sys

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference
import data_analysis_utils as dana
from data_analysis_utils import get_prev_estimates, get_values_with_prior

def expanded_if_needed(array, ndim_with_sessions=2):
    # if the array contains a single session and does not have a dimension
    # indexing the sessions, add one such dimension.
    if array.ndim < ndim_with_sessions:
        array = array[np.newaxis, :]
        did_expand = True
    else:
        did_expand = False
    return array, did_expand

def unexpanded(array):
    # remove the dimension previously added by expanded_if_needed()
    return np.squeeze(array, axis=0)

#
# Ada-Prob Model
#

P1_MIN = 0.1
P1_MAX = 0.9

def ideal_learner_prob_estimates_from_outcomes(outcomes, p_c,
    estimate_type="mean",
    p1_min=P1_MIN, p1_max=P1_MAX):
    if estimate_type == 'mean':
        do_inference_on_current_trial = True
        keys = ['mean', 'SD', 'dist']
    elif estimate_type == 'mean-pred':
        do_inference_on_current_trial = False
        keys = ['mean', 'SD', 'dist']
    elif estimate_type == 'median':
        do_inference_on_current_trial = True
        keys = ['mean', 'SD', 'dist', 'median']
    elif estimate_type == 'median-pred':
        do_inference_on_current_trial = False
        keys = ['mean', 'SD', 'dist', 'median']
        
    inf_dict = ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        p1_min=p1_min, p1_max=p1_max,
        do_inference_on_current_trial=do_inference_on_current_trial,
        keys=keys)
    return inf_dict[estimate_type]

def ideal_learner_prob_confidence_from_outcomes(outcomes, p_c,
    conf_type='prev-conf',
    p1_min=P1_MIN, p1_max=P1_MAX):
    if conf_type == 'prev-conf':
        do_prev = True
        do_inference_on_current_trial = True
    elif conf_type == 'prev-conf-pred':
        do_prev = True
        do_inference_on_current_trial = False
    elif conf_type == 'new-conf':
        do_prev = False
        do_inference_on_current_trial = True
    elif conf_type == 'new-conf-pred':
        do_prev = False
        do_inference_on_current_trial = False

    inf_dict = ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        do_inference_on_current_trial=do_inference_on_current_trial,
        p1_min=p1_min, p1_max=p1_max)
    sds = inf_dict['SD']

    if do_prev:
        sd_prior = np.sqrt(1/12 * (p1_max - p1_min) ** 2)
        if sds.ndim == 2:
            sds = np.insert(sds[:, :-1], 0, sd_prior, axis=1)
        elif sds.ndim == 1:
            sds = np.insert(sds[:-1], 0, sd_prior, axis=0)

    return -np.log(sds)

def ideal_learner_prob_cpp(outcomes, p_c, p1_min=P1_MIN, p1_max=P1_MAX):
    """cpp: probability of a change point on last trial given last outcome"""
    inf_dict = ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        do_inference_on_current_trial=True,
        p1_min=p1_min, p1_max=p1_max)
    # p(X_t | cp)
    p_1_given_cp = (p1_min + p1_max) / 2
    p_last_outcome_given_cp = np.where(outcomes == 1, p_1_given_cp, (1 - p_1_given_cp))
    # p(X_t | no cp)
    p_1_given_no_cp = dana.get_prev_estimates(inf_dict['mean']) # = mean of posterior dist on p1 at previous trial
    p_last_outcome_given_no_cp = np.where(outcomes == 1, p_1_given_no_cp, (1 - p_1_given_no_cp))
    # p(cp | X_t) = p(X_t | cp) p_c / (p(X_t | cp) p_c + p(X_t | no cp) (1-p_c))
    cpp = (p_last_outcome_given_cp * p_c
        / (p_last_outcome_given_cp * p_c + p_last_outcome_given_no_cp * (1 - p_c)))
    return cpp

def ideal_learner_prob_outcome_improbability(outcomes, p_c, p1_min=P1_MIN, p1_max=P1_MAX,
    do_inference_on_current_trial=True):
    inf_dict = ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        p1_min=p1_min, p1_max=p1_max,
        do_inference_on_current_trial=do_inference_on_current_trial)
    p_1_given_no_cp = dana.get_prev_estimates(inf_dict['mean']) # = mean of posterior dist on p1 at previous trial
    p_last_outcome_given_no_cp = np.where(outcomes == 1, p_1_given_no_cp, (1 - p_1_given_no_cp))
    return (1 - p_last_outcome_given_no_cp)

def ideal_learner_prob_outcome_surprise(outcomes, p_c, p1_min=P1_MIN, p1_max=P1_MAX,
    do_inference_on_current_trial=True):
    inf_dict = ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        p1_min=p1_min, p1_max=p1_max,
        do_inference_on_current_trial=do_inference_on_current_trial)
    p_1_given_no_cp = dana.get_prev_estimates(inf_dict['mean']) # = mean of posterior dist on p1 at previous trial
    p_last_outcome_given_no_cp = np.where(outcomes == 1, p_1_given_no_cp, (1 - p_1_given_no_cp))
    return -np.log(p_last_outcome_given_no_cp)


def ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
    keys=['mean', 'SD', 'dist'],
    p1_min=P1_MIN, p1_max=P1_MAX,
    do_inference_on_current_trial=True,
    resol=20):
    outcomes, should_unexpand = expanded_if_needed(outcomes)
    options = {
        'resol': resol,
        'p_c': p_c,
        'p1_min': p1_min,
        'p1_max': p1_max,
        'do_inference_on_current_trial': do_inference_on_current_trial,
    }
    inf_dict = {} # note: we always add dist, SD and mean to this dict since they are provided for free
    inf_dict['dist'] = np.empty(list(outcomes.shape) + [resol])
    for key in ['SD', 'mean']: 
        inf_dict[key] = np.empty(outcomes.shape)
    for key in keys:
        inf_dict[key] = np.empty(outcomes.shape)
    keys = list(inf_dict.keys())
    for key in keys:
        if key == 'dist':
            inf_dict[key] = np.empty(list(outcomes.shape) + [resol])
        else:
            inf_dict[key] = np.empty(outcomes.shape)
    for i_sess in range(outcomes.shape[0]):
        inference_out = run_inference(outcomes[i_sess, :], options=options)
        inference_out = inference_out[(1,)]
        # put the distribution grid in the last dimension
        inference_out['dist'] = np.swapaxes(inference_out['dist'], 0, 1)
        for key in keys:
            if key in ['mean', 'SD', 'dist']:
                inf_dict[key][i_sess, :] = inference_out[key]
            elif key == 'median':
                inf_dict[key][i_sess, :] = compute_dist_medians(inference_out['dist'])

    if should_unexpand:
        for key in keys:
            inf_dict[key] = unexpanded(inf_dict[key])
    return inf_dict

def compute_dist_means(dists):
    dists, should_unexpand = expanded_if_needed(dists, ndim_with_sessions=3)

    pgrid = get_pgrid_for_dist(dists)
    means = np.sum(pgrid * dists, axis=-1)

    if should_unexpand:
        means = unexpanded(means)
    return means

def compute_dist_medians(dists):
    dists, should_unexpand = expanded_if_needed(dists, ndim_with_sessions=3)

    pgrid = get_pgrid_for_dist(dists)
    cumdists = np.cumsum(dists, axis=-1)
    n_sessions = cumdists.shape[0]
    n_trials = cumdists.shape[1]
    medians = np.empty((n_sessions, n_trials))
    for i_sess in range(n_sessions):
        for i_trial in range(n_trials):
            dist = dists[i_sess, i_trial]
            cumdist = cumdists[i_sess, i_trial]
            i = np.searchsorted(cumdist, 0.5, side='left') 
            # returns i such that cumdistdist[i-1] <= 0.5 < cumdistdist[i]
            if i >= 1:
                # interpolate between i-1 and i
                weight_left = (cumdist[i] - 0.5) if dist[i-1] > 0 else 0
                weight_right = (0.5 - cumdist[i-1]) if dist[i] > 0 else 0
                median = (pgrid[i_sess, i_trial, i-1] * weight_left
                    + pgrid[i_sess, i_trial, i] *  weight_right) / (weight_left + weight_right)                
            else:
                # cumdist[0] >= 0.5, so we should take the lowest value, pgrid[0]
                median = pgrid[i_sess, i_trial, 0]
            medians[i_sess, i_trial] = median

    if should_unexpand:
        medians = unexpanded(medians)
    return medians

def get_pgrid_for_dist(dist):
    resol = dist.shape[-1]
    return np.zeros_like(dist) + np.linspace(0, 1, resol)


#
# Delta-rule
#

def delta_rule_estimates_from_outcomes(outcomes, lr, prior_estimate=0.5):
    outcomes, should_unexpand = expanded_if_needed(outcomes)
    estimates = np.empty(outcomes.shape)
    for t in range(outcomes.shape[1]):
        prev_estimate = estimates[:, t-1] if t > 0 else prior_estimate
        estimates[:, t] = prev_estimate + lr * (outcomes[:, t] - prev_estimate)

    if should_unexpand:
        estimates = unexpanded(estimates)
    return estimates


#
# Noisy delta-rule
#

def noisy_delta_rule_estimates_from_outcomes(outcomes, lr,
    noise_level_type="constant", # or "prop-error"
    noise_level_param=0, # this is sigma_epsilon if noise_level_type == "constant",
                         # and zeta if noise_level_type == "prop-error"
    prior_estimate=0.5,
    do_clip_to_bounds=False,
    estimate_bounds=(0, 1.0)):
    # Noisy delta rule model
    #     v[t] = v[t-1] + eta (x[t] - v[t-1]) + epsilon[t]
    # - with constant noise level (noise_level_type: "constant")
    #     epsilon[t] ~ N(0, sigma_epsilon)
    # - with noise level proportion to the error magnitude (noise_level_type: "prop-error")
    #     epsilon[t] = |x[t] - v[t-1]| epsilon2[t], epsilon2[t] ~ N(0, zeta)
    outcomes, should_unexpand = expanded_if_needed(outcomes)
    estimates = np.empty(outcomes.shape)
    for t in range(outcomes.shape[1]):
        # compute noise-free estimate for the next trial
        prev_estimate = estimates[:, t-1] if t > 0 else prior_estimate
        error = (outcomes[:, t] - prev_estimate)
        estimate_noise_free = prev_estimate + lr * error
        # add noise sample
        noise = scipy.stats.norm.rvs(loc=0, scale=1, size=estimate_noise_free.shape)
        if noise_level_type == "constant":
            # the noise s.d. is a constant equal to the given parameter value (sigma_epsilon).
            noise *= noise_level_param
        elif noise_level_type == "prop-error":
            # the noise s.d. is a proportional to the magnitude of the prediction error.
            # the coefficient of proportionality is the given parameter value (zeta).
            for i in range(noise.shape[0]):
                noise[i] *= noise_level_param * np.abs(error[i])
        estimates[:, t] = estimate_noise_free + noise
        if do_clip_to_bounds:
            estimates[:, t] = np.clip(estimates[:, t], estimate_bounds[0], estimate_bounds[1])

    if should_unexpand:
        estimates = unexpanded(estimates)
    return estimates
