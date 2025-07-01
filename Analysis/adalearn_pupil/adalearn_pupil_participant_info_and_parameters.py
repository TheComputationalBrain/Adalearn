"""
This script defines the participant analysed who have taken part in the Pupil task of the Adalearn project. To change participant, the user simply changes the participant number. In addition, the user can specify the base directory (base_dir: the top level location of the datafiles). Changes to the expected task structure can be made here too (e.g. the number of sessions, the session numbers, the condition). The script also includes the parameters used in the normative model (P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR).
"""

from pathlib import Path

import adalearn_analysis_participant_level_functions_participant_info as participant_info


base_dir = '/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Pupil_Ada_Learn/Data/Study'

selected_participant = 36

participant_str = str(selected_participant).zfill(2)
list_subject_selected = [selected_participant]
directory_path = Path(f'{base_dir}/sub-{participant_str}')
condition_str = '01' 
from_session =  1
to_session =  22
task_sessions = [str(session).zfill(2) for session in range(from_session, to_session + 1)]

# Read demographic .tsv file
tsv_df = participant_info.read_tsv_files(directory_path=directory_path)

# exclude zero updates in learning rate from analysis
exclude_zero = False

# Task information
P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
ESTIMATE_PRIOR = 0.5
RESOL = 20