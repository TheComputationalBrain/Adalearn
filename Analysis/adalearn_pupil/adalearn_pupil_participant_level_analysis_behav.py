import plotly.io as pio
pio.renderers.default='browser'

from adalearn_pupil.adalearn_pupil_participant_characteristics import participant_str, directory_path, condition_str, task_sessions

from adalearn_analysis_participant_level_functions_behav import load_behavioural_data, behavioural_change_point_analysis, behavioural_performance_analysis



behavioural_df, computational_var_df = load_behavioural_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, condition_task_sessions=task_sessions, plot_behav_data=False)



num_change_points, change_points_per_session, change_point_indices, behavioural_df = behavioural_change_point_analysis(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, plot_histogram_zscores=False)



mean_abs_error_df = behavioural_performance_analysis(participant_str=participant_str, condition_str=condition_str, behavioural_df=behavioural_df, plot_performance=False)