"""
This script defines the participant analysed. To change participant, the user simply changes the participant number. In addition, the user can specify the base directory (base_dir: the top level location of the datafiles). Changes to the expected task structure can be made here too (e.g. the number of sessions, the session numbers, the condition). The script also includes the parameters used in the normative model (P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR).
"""

from pathlib import Path

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import BASE_DIR

NACONF_BASE_DIR = f'{BASE_DIR}/supplement_dataset'

SELECTED_PARTICIPANT = 4

LIST_SELECTED_PARTICIPANT = [SELECTED_PARTICIPANT]
PARTICIPANT_DIR = Path(f'{NACONF_BASE_DIR}/data/sub-{SELECTED_PARTICIPANT}')
CONDITION_STR = '01' 
FROM_SESSION =  1
TO_SESSION =  4
TASK_SESSIONS = [str(session) for session in range(FROM_SESSION, TO_SESSION +1)]

