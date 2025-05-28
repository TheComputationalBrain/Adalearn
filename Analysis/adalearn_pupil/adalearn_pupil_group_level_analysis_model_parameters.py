"""

"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, load_sample_csvs, save_sample_dataframes_to_csv 

from adalearn_analysis_participant_level_functions_behav import load_behavioural_data

from adalearn_analysis_participant_level_functions_comp import compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, compute_information_gain, behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis


from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, task_sessions, P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR, RESOL



dataframes = load_sample_csvs(directory_path=base_dir, substrings=['change_point_indices_df'])

participant_list = get_participant_numbers(base_dir)

group_computational_var_df_list = []

# For loop, extracting data from behavioural files for each participant within data list and then computing the model parameters for the sequences within each observed sequence
for participant in participant_list:
    participant_str = str(participant).zfill(2)
    print(f"\n\n\nProcessing participant {participant_str}...")
    directory_path = Path(f'{base_dir}/sub-{participant_str}')

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant].copy()

    _ , computational_var_df = load_behavioural_data( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, condition_task_sessions=task_sessions, plot_behav_data=False)

    print(f"\nComputing model parameters...")
    # Computing participants' estimate-derived learning rate
    computational_var_df = behavioural_learning_rate_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,  estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)

    # Computing model estimate-derived learning rate
    computational_var_df = ideal_observer_learning_rate_analysis( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)

    # Computing surprise of each observation
    computational_var_df,_ = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)

    # Computing prior confidence and change point probability of each observation
    computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_str,directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, resol=RESOL, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, conf_type='prev-conf', unit='confidence', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)

    # Computing the information gain of each observation
    computational_var_df = compute_information_gain(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, resol=20, p1_min=P1_MIN, p1_max=P1_MAX, plot_dkl=False, add_surprise_and_conf=True)

    computational_var_df['participant'] = participant_str

    group_computational_var_df_list.append(computational_var_df)

# Concatenate list into group-level dataframe
group_level_computational_var_df = pd.concat(group_computational_var_df_list, ignore_index=True)

# Save group-level csv
save_sample_dataframes_to_csv(dataframes={
    'computational_var_df': group_level_computational_var_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil'
)
