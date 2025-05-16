"""
This script runs participant level analysis on the computational parameters extracted from the normative model utilised in the description of uncertainty parameters in the probability learning task (Adalearn). Specifically, the called functions  examine the relationships between the computational parameters and the participants estimates and learning rates, as well as the relationships between the computational parameters themselves. The dynamics of computational parameters are also examined around features of the task (i.e. change points). The functions offer some adaptability to the user, such as the ability to exclude zero values from the analysis (seperate calls of the function are already included in the script to capture this and the returned object is typically ended with '_zero_excluded'), and to plot the results. Individual plots are saved to a created directory within the specified participant's directory, and the returned dataframes are saved to a created directory within the participant's directory. the user should ensure that the relevant directories are set up correctly before running the script. The script is designed to be run on a single participant at a time, mainly to inspect the data and obtain the data for further collation in later analysis. To load the data from the relevant csv files, the following scripts should be run before running this script:
- adalearn_pupil_participant_level_analysis_behav.py
- adalearn_pupil_participant_level_analysis_comp.py 
"""

from adalearn_analysis_participant_level_functions_comp import learning_rate_by_change_point, behav_v_model_linear_analysis, uncertainty_on_learning_rate_participant_level, comp_parameters_on_info_gain, correlation_around_change_point

from adalearn_analysis_participant_level_functions_data_utilities import plot_session_behav_comp, evaluation_of_learning_rate, save_dataframes_to_csv, load_csvs

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import directory_path, participant_str, condition_str



dataframes = load_csvs(directory_path=directory_path, condition_str=condition_str, substrings=['behavioural_df', 'computational_var_df', 'change_point_indices'])


plot_session_behav_comp(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=dataframes['behavioural_df'], computational_df=dataframes['computational_var_df'], change_point_indices=dataframes['change_point_indices'], task_session=20, turn_on_plotting=False)



behav_v_model_estimate_df = behav_v_model_linear_analysis(computational_var_df=dataframes['computational_var_df'], participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, outcome_var='estimate', pred_var='ideal_observer', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



lr_evaluation_dict = evaluation_of_learning_rate(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, behavioural_df=dataframes['behavioural_df'], computational_var_df=dataframes['computational_var_df'], change_point_indices=dataframes['change_point_indices'], quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=False)



filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp = learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=dataframes['computational_var_df'], change_point_indices=dataframes['change_point_indices'], learning_rate_parameter='estimate', pre=2, post=15, exclude_zero=False, plot_learning_rate_by_cp=False)



filtered_behav_learning_rate_by_cp_zero_excluded, averaged_behav_learning_rate_by_cp_zero_excluded = learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], change_point_indices=dataframes['change_point_indices'], learning_rate_parameter='estimate', pre=2, post=15, exclude_zero=True, plot_learning_rate_by_cp=False)



filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp = learning_rate_by_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,computational_var_df=dataframes['computational_var_df'], change_point_indices=dataframes['change_point_indices'], learning_rate_parameter='ideal_observer', pre=2, post=15, exclude_zero = False, plot_learning_rate_by_cp=False)



change_point_avg_lrs_df, lr_around_cp_corr_df = correlation_around_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,filtered_learning_rate_by_cp=filtered_behav_learning_rate_by_cp, average_over='position', plot_correlation_round_cp=False)



change_point_avg_lrs_df_zero_excluded, lr_around_cp_corr_df_zero_excluded = correlation_around_change_point(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path,filtered_learning_rate_by_cp=filtered_behav_learning_rate_by_cp_zero_excluded, average_over='position', plot_correlation_round_cp=False)



behav_v_model_lr_df = behav_v_model_linear_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_var='cleaned_learning_rate_estimate', pred_var='cleaned_learning_rate_ideal_observer', exclude_zero=False, plot_behav_v_model_scatter=False, add_regression=True)



behav_v_model_lr_df_zero_exclude = behav_v_model_linear_analysis(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], outcome_var='cleaned_learning_rate_estimate', pred_var='cleaned_learning_rate_ideal_observer', exclude_zero=True, plot_behav_v_model_scatter=False, add_regression=True)



learning_rate_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], n_bins=6, parameter_estimate='estimate', exclude_zero=False, plot_corr_subplots=False)



learning_rate_analysis_dict_exclude_zero = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], n_bins=6, parameter_estimate='estimate', exclude_zero=True, plot_corr_subplots=False)



learning_rate_model_analysis_dict = uncertainty_on_learning_rate_participant_level(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], n_bins=6, parameter_estimate='ideal_observer', exclude_zero=False, plot_corr_subplots=False)



computational_var_df, dkl_regression_df = comp_parameters_on_info_gain(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, computational_var_df=dataframes['computational_var_df'], plot_dkl_regression=False)



save_dataframes_to_csv(dataframes={
    'computational_var_df': computational_var_df,
    'averaged_behav_learning_rate_by_cp': averaged_behav_learning_rate_by_cp,
    'averaged_behav_learning_rate_by_cp_zero_excluded': averaged_behav_learning_rate_by_cp_zero_excluded,
    'averaged_model_learning_rate_by_cp': averaged_model_learning_rate_by_cp,
    'lr_around_cp_corr_df': lr_around_cp_corr_df,
    'lr_around_cp_corr_df_zero_excluded': lr_around_cp_corr_df_zero_excluded,
    'behav_v_model_estimate_df': behav_v_model_estimate_df,
    'behav_v_model_lr_df': behav_v_model_lr_df,
    'behav_v_model_lr_df_zero_exclude': behav_v_model_lr_df_zero_exclude,
    'uncertainty_v_lr_df': learning_rate_analysis_dict['uncertainty_v_lr_df'],
    'uncertainty_v_lr_df_zero_excluded': learning_rate_analysis_dict_exclude_zero['uncertainty_v_lr_df'],
    'uncertainty_v_model_lr_df': learning_rate_model_analysis_dict['uncertainty_v_lr_df'],
    'dkl_regression_df': dkl_regression_df
    }, 
    directory_path=directory_path, 
    participant_str=participant_str, 
    condition_str=condition_str, 
    task='Pupil'
    )