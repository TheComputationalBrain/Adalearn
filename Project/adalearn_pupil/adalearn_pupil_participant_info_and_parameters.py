"""
This script defines the analysed participant who have taken part in the Pupil task of the Adalearn project. To change participant for participant level analysis, the user simply changes the participant number. In addition, the user can specify the base directory (base_dir: the top level location of the datafiles). Changes to the expected task structure can be made here too (e.g. the number of sessions, the session numbers, the condition). The script also includes the directories for the analysis and parameters used in the normative model (P_C, P1_MIN, P1_MAX, ESTIMATE_PRIOR).
"""

from pathlib import Path

import adalearn_analysis_participant_level_functions_participant_info as participant_info


#### Directorioes and task parameters
try:
    # Works when running from a .py file
    project_dir = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for Jupyter/interactive
    project_dir = Path.cwd()

PROJECT_DIR = str(project_dir)
BASE_DIR = str(Path(project_dir) / "adalearn_pupil")

# Task information
P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
ESTIMATE_PRIOR = 0.5
RESOL = 20

# exclude zero updates in learning rate from analysis
EXCLUDE_ZERO = False

##### Participant information for running participant level analysis
SELECTED_PARTICIPANT = 31

PARTICIPANT_STR = str(SELECTED_PARTICIPANT).zfill(2)
LIST_SELECTED_PARTICIPANT = [SELECTED_PARTICIPANT]
PARTICIPANT_DIR = Path(f'{BASE_DIR}/data/sub-{PARTICIPANT_STR}')
CONDITION_STR = '01' 
FROM_SESSION =  1
TO_SESSION =  22
TASK_SESSIONS = [str(session).zfill(2) for session in range(FROM_SESSION, TO_SESSION + 1)]