import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

from computational_analysis_functions import compute_surprise_of_observations
from participant_task_characteristics import participant_str, condition_1_str, condition_2_str
from behavioural_analysis_test import adalearn_data_c1_df, adalearn_data_c2_df, c1_change_point_indices, c2_change_point_indices



c1_surprise, c1_post_dist_df, c1_surprise_cat_df = compute_surprise_of_observations(participant_str, condition_1_str, adalearn_data_c1_df, c1_change_point_indices, plot_surprise=True, plot_posterior_conf=True, plot_posterior_dist_means=True)
c2_surprise, c2_post_dist_df, c2_surprise_cat_df = compute_surprise_of_observations(participant_str, condition_2_str, adalearn_data_c2_df, c2_change_point_indices, plot_surprise=True, plot_posterior_conf=True, plot_posterior_dist_means=True)
