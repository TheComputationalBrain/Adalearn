import plotly.io as pio
pio.renderers.default='browser'

from adalearn_analysis_participant_level_functions_comp import compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis, learning_rate_by_change_point, behav_v_model_linear_analysis, uncertainty_on_learning_rate_participant_level

from adalearn_analysis_participant_level_functions_data_utilities import plot_session_behav_comp, evaluation_of_learning_rate

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import behavioural_df, computational_var_df, change_point_indices

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, condition_str, from_session, to_session, task_sessions, P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR



computational_var_df = behavioural_learning_rate_analysis(computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, participant_str=participant_str, condition_str=condition_str, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



computational_var_df = ideal_observer_learning_rate_analysis(computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, participant_str=participant_str, condition_str=condition_str, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=computational_var_df, change_point_indices=change_point_indices, participant_str=participant_str, condition_str=condition_str, learning_rate_parameter='estimate', pre=2, post=10, plot_learning_rate_by_cp=False)



filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp = learning_rate_by_change_point(computational_var_df=computational_var_df, change_point_indices=change_point_indices, participant_str=participant_str, condition_str=condition_str, learning_rate_parameter='ideal_observer', pre=2, post=10, plot_learning_rate_by_cp=False)



computational_var_df, surprise_cat_df = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)



behav_v_model_estimate_corr, behav_v_model_estimate_regress = behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, model_var='ideal_observer', behav_var='estimate', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



behav_v_model_lr_corr, behav_v_model_lr_regress = behav_v_model_linear_analysis(computational_var_df=computational_var_df, participant_str=participant_str, condition_str=condition_str, model_var='cleaned_learning_rate_ideal_observer', behav_var='cleaned_learning_rate_estimate', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



plot_session_behav_comp(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, computational_df=computational_var_df, change_point_indices=change_point_indices, task_session=7, turn_on_plotting=False)



lr_evaluation_dict = evaluation_of_learning_rate(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, computational_var_df=computational_var_df, change_point_indices=change_point_indices, quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=False)



learning_rate_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, computational_var_df=computational_var_df, n_bins=6, parameter_estimate='estimate', exclude_zero=False, plot_corr_subplots=False)




# Attempt to calculate DKL 

import numpy as np
import sys
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Misc')
from IdealObserver import run_inference

from model_learner import ideal_learner_prob_estimates_from_outcomes, ideal_learner_prob_inference_from_outcomes, ideal_learner_prob_confidence_from_outcomes, ideal_learner_prob_cpp



def calculate_dkl_grid(posterior_grid, prior_grid):
    """
    Calculate the Kullback-Leibler divergence for each stimulus in the grid.
    """
    # Ensure no division by zero or log of zero
    epsilon = 1e-10
    prior_grid = np.clip(prior_grid, epsilon, 1)  # Avoid log(0)
    posterior_grid = np.clip(posterior_grid, epsilon, 1)  # Avoid division by 0


    prior_grid = prior_grid / np.sum(prior_grid, axis=0)  # Normalize prior
    posterior_grid = posterior_grid / np.sum(posterior_grid, axis=0)  # Normalize posterior

    # Calculate DKL for each stimulus (sum across grid points)
    dkl_values = np.sum(
        prior_grid * np.log2(prior_grid / posterior_grid), axis=0
    )  # Sum over grid points

    return dkl_values



# Define options for "run inference" function
options = {'p_c': 0.05, 'resol': 20, 'p1_min': 0.1, 'p1_max': 0.9}

# List to store data for each task session
task_session_data = []  # This will store dictionaries for each task session

# Loop through each task session
for task_session, session_data in computational_var_df.groupby('task_session'):
    seq = np.array(session_data['outcome'])

    # Run inference to compute surprise for the sequence of observations
    output = run_inference(seq, options=options)

    # Compute prior and posterior grids
    prior_grid = ideal_learner_prob_inference_from_outcomes(
        outcomes=seq, p_c=options['p_c'], keys=['mean', 'SD', 'dist'], p1_min=options['p1_min'], 
        p1_max=options['p1_max'], do_inference_on_current_trial=True, resol=options['resol']
    )
    
    prior_grid_array = prior_grid['dist']
    prior_grid_array_transposed = prior_grid_array.T  # Transpose to align dimensions
    
    post_grid_array = output[(1,)]['dist'][:, 1:]
    prior_grid_array= output[(1,)]['dist'][:, :-1]

    # Compute surprise, prior conf and cpp for each trial
    surprise = output['surprise']

    prior_conf = ideal_learner_prob_confidence_from_outcomes(
            seq, p_c = options['p_c'], conf_type='prev-conf', p1_min=options['p1_min'], p1_max=options['p1_max']
        )
    
    cpp = ideal_learner_prob_cpp(seq, p_c = options['p_c'], p1_min = options['p1_min'], p1_max = options['p1_max'])
    
    # Calculate DKL for this session
    dkl = calculate_dkl_grid(post_grid_array, prior_grid_array)
    
    # Create a dictionary for this task session with relevant variables
    session_df = pd.DataFrame({
        'task_session': [task_session]*len(dkl),  # Task session identifier
        'dkl': dkl,  # DKL values for this task session
        'surprise': surprise
    })
    
    # Append the dictionary to the list
    task_session_data.append(session_df)

# Convert the list of dictionaries to a pandas DataFrame
task_session_df = pd.concat(task_session_data)


unique_sessions = task_session_df['task_session'].unique()

fig = make_subplots(
    rows=len(unique_sessions), cols=1, shared_xaxes=True,
    subplot_titles=[f'Behavioural learning rate in task session {session}' for session in unique_sessions]
)

# Add DKL traces to each subplot
for session in unique_sessions:
    session_data = task_session_df[task_session_df['task_session'] == session]
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(session_data['dkl'])),  
        y=session_data['dkl'],  
        mode='lines+markers',
        line_color = 'blue',
        name=f'Information gain (DKL)',
    ),
    row=session,  
    col=1
    )

    fig.add_trace(go.Scatter(
        x=np.arange(len(session_data['surprise'])),  
        y=session_data['surprise'],  
        mode='lines+markers',
        line_color = 'red',
        name=f'Surprise',
    ),
    row=session,  
    col=1
    )

fig.update_layout(
    title='DKL Information Gain across Task Sessions',
    height=8000,
    showlegend=True 
)

fig.show()