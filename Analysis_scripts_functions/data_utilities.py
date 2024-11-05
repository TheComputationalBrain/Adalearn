import pandas as pd
import sys
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from physiological_analysis_test import c1_trig_chan_df, c2_trig_chan_df

from pupil_analysis_test import c1_pupil_trig_df, c2_pupil_trig_df

from behavioural_analysis_test import c1_behavioural_df, c2_behavioural_df



def check_trig_sync_between_files(behavioural_df, pupil_trig_df, physiological_triggers_df, plot_outcomes_between_files = True):
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

    # Option to plot outcomes between files across task sessions
    if plot_outcomes_between_files:
        task_sessions = combined_outcome_df['task_session'].unique()
        num_task_sessions = len(task_sessions)
        num_files = len(['behavioural', 'pupil', 'physiological'])

        # Create subplots 
        fig = make_subplots(rows=num_files, cols=num_task_sessions, subplot_titles=[f"Session {i+1}" for i in range(num_task_sessions)])

        # Plot each channel and task session
        for i, task_session in enumerate(task_sessions):
            session_data = combined_outcome_df[combined_outcome_df['task_session'] == task_session]

            # Define colors based on values
            behavioral_colors = ['blue' if val == 1 else 'yellow' for val in session_data['behavioural']]
            pupil_colors = ['blue' if val == 1 else 'yellow' for val in session_data['pupil']]
            physiological_colors = ['blue' if val == 1 else 'yellow' for val in session_data['physiological']]

            # Plot behavioural channel at y=0
            fig.add_trace(
                go.Scatter(x=session_data['cum_outcome_num'], y=[0] * len(session_data),  # Set y to 0
                        mode='markers', name=f"Behavioural {task_session}",
                        marker=dict(color=behavioral_colors)),
                row=1, col=i+1
            )

            # Plot pupil channel at y=0
            fig.add_trace(
                go.Scatter(x=session_data['cum_outcome_num'], y=[0] * len(session_data),  # Set y to 0
                        mode='markers', name=f"Pupil {task_session}",
                        marker=dict(color=pupil_colors)),
                row=2, col=i+1
            )

            # Plot physiological channel at y=0
            fig.add_trace(
                go.Scatter(x=session_data['cum_outcome_num'], y=[0] * len(session_data),  # Set y to 0
                        mode='markers', name=f"Physiological {task_session}",
                        marker=dict(color=physiological_colors)),
                row=3, col=i+1
            )

            # Highlight differences
            if session_data['difference_flag'].any():
                for row_num in range(1, 4):  # Rows 1, 2, and 3 for each channel
                    diff_indices = session_data[session_data['difference_flag']].index
                    fig.add_trace(
                        go.Scatter(x=session_data.loc[diff_indices, 'cum_outcome_num'],
                                y=[0] * len(diff_indices),  # Set y to 0 for differences as well
                                mode='markers', marker=dict(color='orange', symbol='cross', size=10),
                                name="Difference"),
                        row=row_num, col=i+1
                    )

        # Update layout
        fig.update_layout(height=900, width=9000, title_text="Outcome Comparisons Across Files and Task Sessions", showlegend = False), 
        
        fig.update_yaxes(title_text="Behavioural Outcome", row=1, col=1)
        fig.update_yaxes(title_text="Pupil Outcome", row=2, col=1)
        fig.update_yaxes(title_text="Physiological Outcome", row=3, col=1)

        fig.show()


    return combined_outcome_df


# Example usage:
trig_check = check_trig_sync_between_files(behavioural_df=c1_behavioural_df, pupil_trig_df=c1_pupil_trig_df, physiological_triggers_df=c1_trig_chan_df, plot_outcomes_between_files=True)



