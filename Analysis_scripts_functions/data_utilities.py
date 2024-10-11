import pandas as pd
import numpy as np
import os
import sys
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from pupil_analysis_test import c1_trig_data

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Analysis_scripts_functions')
from physiological_analysis_test import outcome_triggers_c1

from behavioural_analysis_test import adalearn_data_c1_df



def check_trig_sync_between_files(behavioural_df, pupil_trig_df, physiological_triggers_df):
    """
    Function to check that the number and order of triggers inserted matches across the different files.
    It aligns the triggers by 'task_session' and handles missing data by inserting NaN where necessary.

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. Variable of interest is 'outcome'. 
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session from pupil .txt file. Variable of interest is 'outcome'
    - physiological_triggers_df: pd.DataFrame, contains trigger data, associated timestamp, and data file number obtained from .edf file. Variable of interest is 'triggers'
    """
    filtered_physiological_df = physiological_triggers_df.groupby('task_session').apply(lambda x: x.iloc[1:-1] if len(x) > 2 else x).reset_index(drop=True)

    combined_outcome_df = pd.DataFrame({
        'behavioural': behavioural_df['outcome'].reset_index(drop=True),
        'pupil': pupil_trig_df['outcome'].reset_index(drop=True),
        'physiological': filtered_physiological_df['triggers'].reset_index(drop=True)
    })

    return combined_outcome_df


# Example usage:
trig_check = check_trig_sync_between_files(adalearn_data_c1_df, c1_trig_data, outcome_triggers_c1)




