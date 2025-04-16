"""
This script is used to apply the ideal observer model incorporated to extract latent parameters related to uncertainty based on the sequence of outcomes observed (housed within the dataframe created after loading the datafiles from the task). The Bayes-optimal model used can be found here: https://github.com/florentmeyniel/MinimalTransitionProbsModel and its original use in relation to the probability learning task (Adalearn) described in Foucault & Meyniel (2024) - Open Mind (utilised code in the called functions taken from: https://github.com/florentmeyniel/Ada-Learn). The following latent parameters are extracted from the model for further analysis: surprise, prior confidence, change point probability. Called functions also calculate information gain (Kullback-Leibler divergence), and learning rate (both for the individual participant and normative model). For prior confidence (-log(std)), the functions are adaptable to obtain the prior variance with different units (e.g. confidence can be inversed to prior uncertainty and expressed in various ways: as a simple inverse, as a decaying uncertainty weight, an uncertainty weight that also factors change point probability). The script is run for each participant individually, and the computed parameters for each sequence are added to the housing dataframe. The script also includes options to plot and inspect the results of this analysis.
"""

from adalearn_analysis_participant_level_functions_comp import compute_surprise_from_observations, compute_prior_conf_and_cpp_from_observations, compute_information_gain, behavioural_learning_rate_analysis, ideal_observer_learning_rate_analysis

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import computational_var_df, change_point_indices

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, condition_str, P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR



computational_var_df = behavioural_learning_rate_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,  estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



computational_var_df = ideal_observer_learning_rate_analysis( participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, estimate_prior=ESTIMATE_PRIOR, plot_learning_rate_by_ses=False)



computational_var_df, surprise_cat_df = compute_surprise_from_observations(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, plot_surprise=False, plot_posterior_conf=False, plot_posterior_dist_means=False)



computational_var_df = compute_prior_conf_and_cpp_from_observations(participant_str=participant_str, condition_str=condition_str,directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices, p_c=P_C, p1_min=P1_MIN, p1_max=P1_MAX, estimate_type='mean', conf_type='prev-conf', unit='confidence', plot_distribution_mean=False, plot_prior_conf=False, plot_cpp=False)



computational_var_df = compute_information_gain(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=computational_var_df, change_point_indices_df=change_point_indices,  p_c=P_C, resol=20, p1_min=P1_MIN, p1_max=P1_MAX, plot_dkl=False, add_surprise_and_conf=True)


