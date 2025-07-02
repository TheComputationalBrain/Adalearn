import numpy as np
import os
import os.path as op
import pandas as pd
import pickle

ESTIMATE_PRIOR = 0.5

def get_prev_estimates(estimates, estimate_prior=ESTIMATE_PRIOR):
    estimates_with_prior = get_values_with_prior(estimates, prior=estimate_prior)
    if estimates.ndim == 2:
        return estimates_with_prior[:, :-1] #np.insert(estimates[:, :-1], 0, estimate_prior, axis=1)
    elif estimates.ndim == 1:
        return estimates_with_prior[:-1] #np.insert(estimates[:-1], 0, estimate_prior, axis=0)
    
def get_values_with_prior(values, prior):
    if values.ndim == 2:
        return np.insert(values, 0, prior, axis=1)
    elif values.ndim == 1:
        return np.insert(values, 0, prior, axis=0)