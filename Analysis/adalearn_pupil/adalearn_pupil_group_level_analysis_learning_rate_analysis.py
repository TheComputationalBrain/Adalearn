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

from adalearn_analysis_participant_level_functions_comp import learning_rate_by_change_point, behav_v_model_linear_analysis, uncertainty_on_learning_rate_participant_level, correlation_around_change_point

from adalearn_analysis_group_level_functions_data_utilities import get_participant_numbers, load_sample_csvs, save_sample_dataframes_to_csv 

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str


dataframes = load_sample_csvs(directory_path=base_dir, substrings=['change_point_indices_df', 'computational_var_df'])

participant_list = get_participant_numbers(base_dir)

group_behav_v_model_estimate_corr_list = []
group_behav_v_model_lr_corr_list = []
group_averaged_lr_round_cp_list = []
group_lr_corr_around_cp_list = []
group_uncertainty_params_on_lr_list = []

# For loop, loading the behavioural data and computed model parameters and then evaluating the effect on learning rate
for participant in participant_list:
    participant_str = str(participant).zfill(2)
    print(f"\n\n\nProcessing participant {participant_str}...")
    directory_path = Path(f'{base_dir}/sub-{participant_str}')

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant'] == participant].copy()
    
    computational_var_df = dataframes['computational_var_df'][dataframes['computational_var_df']['participant'] == participant].copy()

    print(f"\nAnalysing association between participants' and model estimates and learning rate...")
    # Evaluating the association between participants' and model estimates
    behav_v_model_estimate_df = behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, outcome_var='estimate', pred_var='ideal_observer', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)

    behav_v_model_estimate_df['participant'] = participant_str

    group_behav_v_model_estimate_corr_list.append(behav_v_model_estimate_df)

    # Evaluating the association between participants' and model learning rates
    behav_v_model_lr_df = behav_v_model_linear_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df
    , outcome_var='cleaned_learning_rate_estimate', pred_var='cleaned_learning_rate_ideal_observer', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)

    behav_v_model_lr_df['participant'] = participant_str

    group_behav_v_model_lr_corr_list.append(behav_v_model_lr_df)

    print(f"\nAnalysing association between participants' and model learning rates around change points...")
    # Computing the learning rate around change point (participants' estimates)
    filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp = learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=computational_var_df, change_point_indices=change_point_indices, learning_rate_parameter='estimate', pre=2, post=15, exclude_zero=False, plot_learning_rate_by_cp=False)

    # Computing the learning rate around change points (model estimates)
    filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp = learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=computational_var_df, change_point_indices=change_point_indices, learning_rate_parameter='ideal_observer', pre=2, post=15, exclude_zero = False, plot_learning_rate_by_cp=False)

    averaged_learning_rate_by_cp = pd.merge(left=averaged_behav_learning_rate_by_cp, right=averaged_model_learning_rate_by_cp, on=['relative_position'], how='outer', suffixes=('', '_model'))

    averaged_learning_rate_by_cp['participant'] = participant_str

    group_averaged_lr_round_cp_list.append(averaged_learning_rate_by_cp)

    # Evaluating the correlation between participants' and model estimates around change points
    change_point_avg_lrs_df, lr_around_cp_corr_df = correlation_around_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,filtered_learning_rate_by_cp=filtered_behav_learning_rate_by_cp, average_over='position', plot_correlation_round_cp=False)

    change_point_avg_lrs_df['participant'] = participant_str

    group_lr_corr_around_cp_list.append(lr_around_cp_corr_df)

    (f"\nComputing the effect of uncertainty parameters on learning rate...")
    # Linear regression for uncertainty parameters on participants' learning rate
    learning_rate_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='estimate', exclude_zero=False, plot_corr_subplots=False)

    # Linear regression for uncertainty parameters on model learnijng rate (normative account)
    learning_rate_model_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='ideal_observer', exclude_zero=False, plot_corr_subplots=False)

    group_uncertainty_params_on_lr = pd.merge(left=learning_rate_analysis_dict['uncertainty_v_lr_df'], right=learning_rate_model_analysis_dict['uncertainty_v_lr_df'], on=['variable'], how='outer', suffixes=('', '_model'))

    group_uncertainty_params_on_lr['participant'] = participant_str

    group_uncertainty_params_on_lr_list.append(group_uncertainty_params_on_lr)

# Collate lists to dataframes
group_behav_v_model_estimate_corr_df = pd.concat(group_behav_v_model_estimate_corr_list,ignore_index=True)
group_behav_v_model_lr_corr_df = pd.concat(group_behav_v_model_lr_corr_list, ignore_index=True)
group_averaged_lr_round_cp_df = pd.concat(group_averaged_lr_round_cp_list, ignore_index=True)
group_lr_corr_around_cp_df = pd.concat(group_lr_corr_around_cp_list, ignore_index=True)
group_uncertainty_params_on_lr_df = pd.concat(group_uncertainty_params_on_lr_list) 

# Save group-level csv
save_sample_dataframes_to_csv(dataframes={
    'sample_level_behav_v_model_estimate_corr_df': group_behav_v_model_estimate_corr_df,
    'sample_level_behav_v_model_lr_corr_df': group_behav_v_model_lr_corr_df,
    'sample_level_averaged_lr_round_cp_df': group_averaged_lr_round_cp_df,
    'sample_level_lr_corr_around_cp_df': group_lr_corr_around_cp_df,
    'sample_level_uncertainty_params_on_lr_df': group_uncertainty_params_on_lr_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil'
)
