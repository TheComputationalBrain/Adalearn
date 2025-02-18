from pathlib import Path

from adalearn_analysis_participant_level_functions_participant_info import read_tsv_files


base_dir = '/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Pupil_Ada_Learn/Data/Study'

selected_participant = 17

participant_str = str(selected_participant).zfill(2)
list_subject_selected = [selected_participant]
directory_path = Path(f'{base_dir}/sub-{participant_str}')
condition_str = '01' 
from_session =  1
to_session =  22
task_sessions = [str(session).zfill(2) for session in range(from_session, to_session + 1)]

# Read demogrpahic .tsv file
tsv_df = read_tsv_files(directory_path=directory_path)

# Task information
P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
ESTIMATE_PRIOR = 0.5