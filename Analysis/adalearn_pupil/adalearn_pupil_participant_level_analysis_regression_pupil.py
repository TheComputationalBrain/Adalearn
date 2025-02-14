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


from adalearn_analysis_participant_level_functions_regression_pupil import pupil_regression_analysis

from adalearn_pupil.adalearn_pupil_participant_level_analysis_pupil import pupil_trig_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df

from adalearn_pupil.adalearn_pupil_participant_characteristics import directory_path, participant_str, list_subject_selected, condition_str, from_session, to_session, task_sessions


conf_cpp_coef = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_str, comp_variable_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df, outcome_epochs_to_remove=outcome_epochs_to_remove, variables=['conf_zscore', 'cpp_zscore'], include_outcome=True,plot_regression_analysis=True)


conf_cpp_coef_nobc = pupil_regression_analysis(participant_str=participant_str, condition_str = condition_str, comp_variable_df=computational_var_df, outcome_epoched_pupil_df=outcome_epoched_pupil_df_nobc, outcome_epochs_to_remove=outcome_epochs_to_remove_nobc, variables=['conf_zscore', 'cpp_zscore'], include_outcome=True,plot_regression_analysis=True)

