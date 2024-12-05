import pandas as pd
import sys
import os
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'


def check_trig_sync_between_files(participant_str, condition_str, behavioural_df, pupil_trig_df, physiological_triggers_df, plot_outcomes_between_files = True):
    """
    Function to check that the number and order of triggers inserted matches across the different files.
    It aligns the triggers by 'task_session' and handles missing data by inserting NaN where necessary.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers)
    - behavioural_df: pd.DataFrame, contains parsed data from all .json adalearn data files for select variables. Variable of interest is 'outcome'. 
    - pupil_trig_df: pd.DataFrame, contains the trigger information (i.e. time, outcome) for each task session from pupil .txt file. Variable of interest is 'outcome'
    - physiological_triggers_df: pd.DataFrame, contains trigger data, associated timestamp, and data file number obtained from .edf file. Variable of interest is 'triggers'


    ##### Currently doesn't match due to mismatch in task sessions in participant 3 (due to skip of the first task session file due to size). Also need to convert bipoac triggers to binary based on channels triggers taken from ######
    """
    filtered_physiological_df = physiological_triggers_df.groupby('task_session').apply(lambda x: x.iloc[1:-1] if len(x) > 2 else x).reset_index(drop=True)

    combined_outcome_df = pd.DataFrame({
        'behavioural': behavioural_df['outcome'].reset_index(drop=True),
        'pupil': pupil_trig_df['outcome'].reset_index(drop=True),
        'physiological': filtered_physiological_df['outcome'].reset_index(drop=True),
        'task_session': behavioural_df['task_session'].reset_index(drop=True),
        'outcome_num': behavioural_df['outcome_num'].reset_index(drop=True)
    })
    combined_outcome_df['cum_outcome_num'] = combined_outcome_df.index
    
    combined_outcome_df['difference_flag'] = combined_outcome_df.apply(lambda x: len(set([x['behavioural'], x['pupil'], x['physiological']])) > 1, axis=1)

    if combined_outcome_df['difference_flag'].any():
        print(f"ERROR: Mismatch in outcomes derived from different files")
    else:
        print("MATCHED: Outcomes derived from different files are matched")
    
    combined_outcome_df_filled = combined_outcome_df.copy().fillna(-1)

    # Plot outcomes if requested
    if plot_outcomes_between_files:
        task_sessions = combined_outcome_df_filled['task_session'].unique()
        num_task_sessions = len(task_sessions)
        
        # Create subplots
        fig = make_subplots(rows=3, cols=num_task_sessions, subplot_titles=[f"Session {i+1}" for i in range(num_task_sessions)])

        color_map = {1: 'blue', 0: 'yellow', -1: 'red'}
        symbol_map = {1: 'circle', 0: 'circle', -1: 'cross'}

        # Plot each channel and task session
        for i, task_session in enumerate(task_sessions):
            session_data = combined_outcome_df_filled[combined_outcome_df_filled['task_session'] == task_session]

            for j, (col, label) in enumerate(zip(['behavioural', 'pupil', 'physiological'], ['Behavioural', 'Pupil', 'Physiological'])):
                valid_session_data = session_data[pd.notna(session_data[col])]
                colors = [color_map[val] for val in valid_session_data[col]]
                symbols = [symbol_map[val] for val in session_data[col]]
                
                fig.add_trace(
                    go.Scatter(
                        x=valid_session_data['cum_outcome_num'],
                        y=[0] * len(valid_session_data),  
                        mode='markers',
                        marker=dict(color=colors, symbol=symbols, size=10),
                        name=f"{label} {task_session}",
                    ),
                    row=j+1, col=i+1
                )

        fig.update_layout(height=900, width=900 * num_task_sessions, title_text=f"Outcome comparisons across files and task sessions for participant {participant_str} in condition {condition_str}", showlegend=False)
        
        fig.update_yaxes(title_text="Behavioural Outcome", row=1, col=1)
        fig.update_yaxes(title_text="Pupil Outcome", row=2, col=1)
        fig.update_yaxes(title_text="Physiological Outcome", row=3, col=1)

        fig.show()

    return combined_outcome_df





