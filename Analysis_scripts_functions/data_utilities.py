import pandas as pd
import sys
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from physiological_analysis_test import c1_trig_chan_df, c2_trig_chan_df

from pupil_analysis_test import c1_pupil_trig_df, c2_pupil_trig_df

from behavioural_analysis_test import c1_behavioural_df, c2_behavioural_df, c1_change_point_indices, c2_change_point_indices

from computational_analysis_test import c1_computational_var_df, c2_computational_var_df

from participant_task_information import participant_str, condition_1_str, condition_2_str

from data_utility_functions import check_trig_sync_between_files, evaluation_of_learning_rate, plot_session_behav_comp



c1_trig_check = check_trig_sync_between_files(participant_str=participant_str, condition_str=condition_1_str, behavioural_df=c1_behavioural_df, pupil_trig_df=c1_pupil_trig_df, physiological_triggers_df=c1_trig_chan_df, plot_outcomes_between_files=True)

c2_trig_check = check_trig_sync_between_files(participant_str=participant_str, condition_str=condition_2_str,behavioural_df=c2_behavioural_df, pupil_trig_df=c2_pupil_trig_df, physiological_triggers_df=c2_trig_chan_df, plot_outcomes_between_files=True)



c1_lr_evaluation_dict = evaluation_of_learning_rate(behavioural_df=c1_behavioural_df, computational_var_df=c1_computational_var_df, change_point_indices=c1_change_point_indices, quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=True)

c2_lr_evaluation_dict = evaluation_of_learning_rate(behavioural_df=c2_behavioural_df, computational_var_df=c2_computational_var_df, change_point_indices=c2_change_point_indices, quantile_baseline=0.2, parameter_estimate='estimate', baseline_method='quantile_over_sessions', plot_lr_evaluation=True)



plot_session_behav_comp(participant_str=participant_str, condition_str=condition_1_str, behavioural_df=c1_behavioural_df, computational_df=c1_computational_var_df, change_point_indices=c1_change_point_indices, task_session=1)

plot_session_behav_comp(participant_str=participant_str, condition_str=condition_2_str, behavioural_df=c2_behavioural_df, computational_df=c2_computational_var_df, change_point_indices=c2_change_point_indices, task_session=1)   