from participant_info_functions import participant_characteristics

base_dir = '/home/aaron/Documents/Python/Ada_Learn/VNS_Ada_Learn/Data'

selected_participant = 4

participant_str, list_subject_selected, directory_path, tsv_df, high_vns_condition, low_vns_condition, condition_1_str, condition_2_str, c1_task_sessions, c1_from_session, c1_to_session, c2_task_sessions, c2_from_session, c2_to_session, c1_run_ranges, c2_run_ranges, c1_vns_task_sessions, c2_vns_task_sessions = participant_characteristics(participant=selected_participant, base_dir=base_dir)

# Task information
P_C = 0.05
P1_MIN = 0.1
P1_MAX = 0.9
ESTIMATE_PRIOR = 0.5