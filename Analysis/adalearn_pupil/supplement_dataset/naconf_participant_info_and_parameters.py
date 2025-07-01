"""
This script defines the participant analysed. To change participant, the user simply changes the participant number. In addition, the user can specify the base directory (base_dir: the top level location of the datafiles). Changes to the expected task structure can be made here too (e.g. the number of sessions, the session numbers, the condition). The script also includes the parameters used in the normative model (P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR).
"""

from pathlib import Path


NACONF_base_dir = '/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Analysis/adalearn_pupil/supplement_dataset'

selected_participant = 4

list_subject_selected = [selected_participant]
directory_path = Path(f'{NACONF_base_dir}/data/sub-{selected_participant}')
condition_str = '01' 
from_session =  1
to_session =  4
task_sessions = [str(session) for session in range(from_session, to_session +1)]

