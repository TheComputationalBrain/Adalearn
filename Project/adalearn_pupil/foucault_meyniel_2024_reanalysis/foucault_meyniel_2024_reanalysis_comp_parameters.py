"""
This script reanalyses the uncertainty determinants of learning rate across participants in the Foucault & Meyniel 2024 study. It also derives update frequency and individual volatility estimates that will be used in future correlation analysis. The results are saved to .csv files. The data from the Foucault & Meyniel 2024 study needs to be downloaded first. Running the script automatically clones the repository: https://github.com/cedricfoucault/ada-learn.
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import scipy.optimize as so
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, project_dir, exclude_zero, ESTIMATE_PRIOR, P_C, P1_MIN, P1_MAX, RESOL

import adalearn_analysis_participant_level_functions_behav as behav
import adalearn_analysis_participant_level_functions_comp as comp
import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

model_function_dir = Path(f'{project_dir}/external_reference_functions').resolve()
sys.path.append(model_function_dir)
from model_learner import ideal_learner_prob_estimates_from_outcomes


# Cloning Foucault & Meyniel (2024) repository from git hub
# Define where the repository should live locally (home directory)
repo_name = "ada-learn"
repo_url = "https://github.com/cedricfoucault/ada-learn"
repo_path = Path.home() / repo_name  # Save in the user's home directory

# Check if repo exists; if not, clone it
if not repo_path.exists():
    print(f"Repository '{repo_name}' not found in home directory. Cloning from GitHub...")
    try:
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        print(f"Repository cloned to {repo_path}")
    except subprocess.CalledProcessError as e:
        print("Failed to clone repository. Please check your internet connection or Git installation.")
        raise e
else:
    print(f"Repository '{repo_name}' already exists at {repo_path}")

# Defining directory to save results
directory_path = os.path.join(base_dir, 'foucault_meyniel_2024_reanalysis')

# Check if it exists; if not, create it
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Created folder: {directory_path}")
else:
    print(f"Folder already exists: {directory_path}")

# Define file names
files = ["data_outcome-level.csv", "data_session-level.csv", "data_subject-level.csv", "task-parameters.csv"]

# Read each file into a dictionary of DataFrames
dataframes = {file: pd.read_csv(os.path.join(f'{repo_path}/data/structured-dataset/ada-learn_study', file)) for file in files}

# Access the DataFrames
foucault_meyniel_outcome = dataframes["data_outcome-level.csv"]
foucault_meyniel_session = dataframes["data_session-level.csv"]

# Match dataframes to structure of functions
foucault_meyniel_prob_task = foucault_meyniel_outcome[foucault_meyniel_outcome['task']=='ada-prob']
foucault_meyniel_prob_task.rename(columns={'session_idx': 'task_session'}, inplace=True)
foucault_meyniel_prob_task.rename(columns={'did_change_point_occur': 'change_point'}, inplace=True)
foucault_meyniel_prob_task.rename(columns={'outcome_idx': 'outcome_num'}, inplace=True)
foucault_meyniel_prob_task.rename(columns={'hidden_parameter': 'P1'}, inplace=True)

# Identify duplicated participant IDs
participant_ids = foucault_meyniel_prob_task['subject'].value_counts()
duplicate_ids = participant_ids[participant_ids > 15*75]
if duplicate_ids.empty:
    print("No duplicated participant IDs found.")
else:
    print(f"Duplicated participant IDs found: {duplicate_ids.index.tolist()}")


# Loop through each duplicated participant ID
for pid in duplicate_ids.index:  
    print(f"Processing participant: {pid}")  
    
    # Get all index positions for this participant
    idx_values = foucault_meyniel_prob_task[foucault_meyniel_prob_task['subject'] == pid].index.to_list()  
        
    # Compute the difference between consecutive index positions
    jumps = [idx_values[i] - idx_values[i-1] > 1 for i in range(1, len(idx_values))]

    if any(jumps):  
        split_index = idx_values[jumps.index(True) + 1] 
        print(f"Split found at index {split_index} for participant {pid}")

        # Only modify the rows where 'subject' is still `pid` **after** the split point
        foucault_meyniel_prob_task.loc[(foucault_meyniel_prob_task.index >= split_index) & (foucault_meyniel_prob_task['subject'] == pid), 'subject'] = f"{pid}_b"  

# Count unique participants
participants = foucault_meyniel_prob_task['subject'].unique()
print(f'{len(participants)} unique participants found in the Foucault & Meyniel (2024) dataset.')

# Setting random seed for reproducibility
np.random.seed(42)

participant_dfs = {}
subject_num = 0

for i in foucault_meyniel_prob_task['subject'].unique():
    subject_num += 1 
    data = foucault_meyniel_prob_task[foucault_meyniel_prob_task['subject'] == i].copy()
    data = data.reset_index(drop=True)
    data['rel_time'] = np.arange(0, len(data) * 1.5, 1.5)  
    participant_dfs[subject_num] = data  

# Reanalyzing the data with created functions
change_point_indices_sub_list = []
update_task_session_sub_list = []
computational_param_sub_list = []
estimate_v_model_corr_sub_list = []
uncert_outcome_sub_list = []
individual_volatility_sub_list = []  

for participant, behavioural_df in participant_dfs.items():
    # Running inference to obtain ideal observer model estimates
    for session in behavioural_df['task_session'].unique():
        session_data = behavioural_df[behavioural_df['task_session'] == session].copy()
        session_data['task_session'] = session  
        
        sequence = session_data['outcome'].values

        inference = ideal_learner_prob_estimates_from_outcomes(outcomes=sequence, p_c=P_C,  estimate_type='mean', p1_min=P1_MIN, p1_max=P1_MAX)

        behavioural_df.loc[behavioural_df['task_session'] == session, 'ideal_observer'] = inference

    # Change point indices
    num_change_points, change_points_per_session, change_point_indices, computational_var_df = behav.behavioural_change_point_analysis(
        participant_str=participant, condition_str="01", directory_path=directory_path,
        behavioural_df=behavioural_df, plot_histogram_zscores=False
    )
    change_point_indices['participant'] = participant
    computational_var_df['participant'] = participant
    change_point_indices_sub_list.append(change_point_indices)

    # Learning rate analysis
    print(f"\nComputing model parameters for participant: {participant}...")

    computational_var_df,_ = comp.run_inference_from_observations(
        participant_str=participant, condition_str='01', directory_path=directory_path,
        computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,
        p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, resol=RESOL,
        plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False
    )

    # Prior confidence & CPP
    computational_var_df = comp.compute_prior_conf_and_cpp_from_observations(
        participant_str=participant, condition_str='01', directory_path=directory_path,
        computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,
        resol=RESOL, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX,
        conf_type='prev-conf-pred', unit='confidence',
        plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False
    )

    computational_var_df = comp.behavioural_learning_rate_analysis(
        participant_str=participant, condition_str="01", directory_path=directory_path,
        computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,
        estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False
    )

    computational_var_df = comp.ideal_observer_learning_rate_analysis(
        participant_str=participant, condition_str="01", directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False
    )

    computational_param_sub_list.append(computational_var_df)

    # Uncertainty regression on learning rate
    print(f"\nComputing the effect of uncertainty parameters on learning rate for participant: {participant}...")
    learning_rate_analysis_dict = comp.uncertainty_on_learning_rate_participant_level(
        participant_str=participant, condition_str="01", directory_path=directory_path,
        computational_var_df=computational_var_df, n_bins=6, parameter_estimate='estimate',
        exclude_zero=exclude_zero, include_interaction=True, plot_corr_subplots=False
    )
    coef_df = learning_rate_analysis_dict['uncertainty_v_lr_df']
    coef_df['participant'] = participant
    uncert_outcome_sub_list.append(coef_df)

    # Participant estimate v model correlation
    behav_v_model_estimate_df = comp.behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant, condition_str="01", directory_path=directory_path, outcome_var='estimate', pred_var='ideal_observer', exclude_zero=exclude_zero, plot_behav_v_model_scatter=False, add_regression=True)

    behav_v_model_estimate_df['participant'] = participant

    estimate_v_model_corr_sub_list.append(behav_v_model_estimate_df)

    # Update frequency
    update_dict = {'participant': [], 'task_session': [], 'update_count': [], 'update_percentage': []}
   
    for session in computational_var_df['task_session'].unique():
        update_data = computational_var_df[computational_var_df['task_session'] == session].copy()
        sum_update = update_data['was_updated_estimate'].sum()

        update_dict['participant'].append(participant)
        update_dict['task_session'].append(session)
        update_dict['update_count'].append(sum_update)
        update_dict['update_percentage'].append(sum_update / len(update_data) * 100)

    update_df = pd.DataFrame(update_dict)
    update_task_session_sub_list.append(update_df)

    # Individual volatility estimation
    print(f"\nComputing individual volatility for participant: {participant}...")
    lb = np.array([0.0001])
    ub = np.array([0.9])
    bounds = [(lb, ub)]
    best_mse = np.inf
    fp_init = None
    n_initial_guesses = 100

    print(f'\nEstimating best initial guess...')
    for ii in range(n_initial_guesses):
        fp_random = lb + (ub - lb) * np.random.rand(len(lb))
        fp_random = np.maximum(np.minimum(fp_random, ub), lb)
        mse = comp.inference_and_mse(pars=fp_random, data=behavioural_df, p1_min=P1_MIN, p1_max=P1_MAX, resol=RESOL)
        if mse < best_mse:
            best_mse = mse
            fp_init = fp_random

    fp_init = np.array(fp_init)

    print(f'Running optimisation for participant: {participant} with initial guess: {fp_init}...')
    
    opt = so.minimize(comp.inference_and_mse, fp_init, args=(behavioural_df, P1_MIN, P1_MAX, RESOL),
                      method='Powell', bounds=bounds, options={'disp': False}) 

    x_min = opt['x']

    opt_results = pd.DataFrame({
        'participant': [participant], 
        'optimal_p_c': x_min
        })

    individual_volatility_sub_list.append(opt_results)


change_point_indices_df = pd.concat(change_point_indices_sub_list, ignore_index=True)
update_freq_df = pd.concat(update_task_session_sub_list, ignore_index=True)
estimate_v_model_corr_df = pd.concat(estimate_v_model_corr_sub_list, ignore_index=True)
computational_param_df = pd.concat(computational_param_sub_list, ignore_index=True)
uncert_outcome_df = pd.concat(uncert_outcome_sub_list, ignore_index=True)
individual_volatility_df = pd.concat(individual_volatility_sub_list, ignore_index=True)

# Save results to CSV files
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'foucault_meyniel_2024_change_point_indices_df': change_point_indices_df,
    'foucault_meyniel_2024_update_freq_df': update_freq_df,
    'foucault_meyniel_2024_estimate_v_model_corr_df': estimate_v_model_corr_df,
    'foucault_meyniel_2024_comp_params_df': computational_param_df,
    'foucault_meyniel_2024_uncert_lr_regress_coefs_df': uncert_outcome_df,
    'foucault_meyniel_2024_ind_volatility_df': individual_volatility_df
},
directory_path=directory_path,
condition_str="01",
task='Pupil'
)