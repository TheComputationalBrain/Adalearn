import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from pupil_statistical_analysis_functions import pupil_regression_analysis

from participant_task_information import participant_str, condition_1_str, condition_2_str

from pupil_analysis_test import c1_pupil_trig_df, c1_outcome_epoched_pupil_df, c1_outcome_epoched_pupil_df_nobc, c1_outcome_blink_ratio_df, c1_outcome_epochs_to_remove, c2_pupil_trig_df, c2_outcome_epoched_pupil_df, c2_outcome_epoched_pupil_df_nobc, c2_outcome_blink_ratio_df, c2_outcome_epochs_to_remove

from computational_analysis_test import c1_computational_var_df, c2_computational_var_df



c1_conf_cpp_coef = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_1_str, comp_variable_df=c1_computational_var_df, outcome_epoched_pupil_df=c1_outcome_epoched_pupil_df, outcome_epochs_to_remove=c1_outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], include_outcome=False,plot_regression_analysis=True)

c2_conf_cpp_coef = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_2_str, comp_variable_df=c2_computational_var_df, outcome_epoched_pupil_df=c2_outcome_epoched_pupil_df, outcome_epochs_to_remove=c2_outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], include_outcome=False, plot_regression_analysis=True)

c1_conf_cpp_coef_nobc = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_1_str, comp_variable_df=c1_computational_var_df, outcome_epoched_pupil_df=c1_outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=c1_outcome_epochs_to_remove,include_outcome=False, variables=['conf_zscore', 'cpp_zscore'])

c2_conf_cpp_coef_nobc = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_2_str, comp_variable_df=c2_computational_var_df, outcome_epoched_pupil_df=c2_outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=c2_outcome_epochs_to_remove,include_outcome=False, variables=['conf_zscore', 'cpp_zscore'])