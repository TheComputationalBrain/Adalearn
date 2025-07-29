from pathlib import Path

from adalearn_analysis_participant_level_functions_participant_info import vns_condition_characteristics



base_dir = '/home/aarongreenhouse-tucknott/Documents/Python/Ada_Learn/Analysis/Project/adalearn_VNS'

selected_participant = 2

directory_path = Path(f'{base_dir}/data/sub-{str(selected_participant).zfill(2)}')


participant_str, list_subject_selected, directory_path, tsv_df, vns_info_dict = vns_condition_characteristics(participant=selected_participant, directory_path=directory_path)

# exclude zero updates in learning rate from analysis
exclude_zero = False

# Task information
P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
ESTIMATE_PRIOR = 0.5