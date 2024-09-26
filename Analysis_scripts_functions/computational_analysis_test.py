import os
import pandas as pd
import numpy as np
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
pio.renderers.default='browser'

from computational_analysis_functions import compute_surprise_of_observations
from participant_task_characteristics import participant_str, condition_1_str
from behavioural_analysis_test import adalearn_data_c1_df



c1_surprise = compute_surprise_of_observations(participant_str, condition_1_str, adalearn_data_c1_df)
