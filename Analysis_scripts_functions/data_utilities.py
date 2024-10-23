import pandas as pd
import sys
import plotly.io as pio
pio.renderers.default='browser'

from vega_pupil_script import c1_pupil_trig_df

sys.path.append('/home/aaron/Documents/Python/Ada_Learn/Analysis/Analysis_scripts_functions')
from physiological_analysis_test import c1_outcome_triggers

from behavioural_analysis_test import c1_behavioural_df



def check_trig_sync_between_files(behavioural_df, pupil_trig_df, physiological_triggers_df):
    """
    Function to check that the number and order of triggers inserted matches across the different files.
    It aligns the triggers by 'task_session' and handles missing data by inserting NaN where necessary.

    Parameters:
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. Variable of interest is 'outcome'. 
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session from pupil .txt file. Variable of interest is 'outcome'
    - physiological_triggers_df: pd.DataFrame, contains trigger data, associated timestamp, and data file number obtained from .edf file. Variable of interest is 'triggers'


    ##### Currently doesn't match due to mismatch in task sessions in participant 3 (due to skip of the first task session file due to size). Also need to convert bipoac triggers to binary based on channels triggers taken from ######
    """
    filtered_physiological_df = physiological_triggers_df.groupby('task_session').apply(lambda x: x.iloc[1:-1] if len(x) > 2 else x).reset_index(drop=True)

    combined_outcome_df = pd.DataFrame({
        'behavioural': behavioural_df['outcome'].reset_index(drop=True),
        'pupil': pupil_trig_df['outcome'].reset_index(drop=True),
        'physiological': filtered_physiological_df['triggers'].reset_index(drop=True),
        'task_session': behavioural_df['task_session'].reset_index(drop=True),
        'outcome_num': behavioural_df['outcome_num'].reset_index(drop=True)
    })
    combined_outcome_df['cum_outcome_num'] = combined_outcome_df.index


    return combined_outcome_df


# Example usage:
trig_check = check_trig_sync_between_files(behavioural_df=c1_behavioural_df, pupil_trig_df=c1_pupil_trig_df, physiological_triggers_df=c1_outcome_triggers)



