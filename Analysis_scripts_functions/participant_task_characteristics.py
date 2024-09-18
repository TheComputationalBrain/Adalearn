# Participant, condition, task session and stimulation characteristics

# Participant
participant = 1
participant_str = str(participant).zfill(2)
list_subject_selected = [participant]

# Conditions
condition_1 = 1
condition_1_str = str(condition_1).zfill(2)
c1_vns_intensity = 0.75

condition_2 = 2
condition_2_str = str(condition_2).zfill(2)
c2_vns_intensity = 0.25

# Condition task sessions
c1_from_session, c1_to_session = 1, 20
c1_first_task_session_str = str(c1_from_session).zfill(2)
c1_last_task_session_str = str(c1_to_session).zfill(2)
c1_task_sessions = [str(session).zfill(2) for session in range(c1_from_session, c1_to_session + 1)]
c1_run_ranges = [
    {'run_start': '01', 'run_end': '04'},
    {'run_start': '05', 'run_end': '08'},
    {'run_start': '09', 'run_end': '12'},
    {'run_start': '13', 'run_end': '16'},
    {'run_start': '17', 'run_end': '20'}
]

c2_from_session, c2_to_session = 1, 16
c2_first_task_session_str = str(c2_from_session).zfill(2)
c2_last_task_session_str = str(c2_to_session).zfill(2)
c2_task_sessions = [str(session).zfill(2) for session in range(c2_from_session, c2_to_session + 1)]
c2_run_ranges = [
    {'run_start': '01', 'run_end': '04'},
    {'run_start': '05', 'run_end': '08'},
    {'run_start': '09', 'run_end': '12'},
    {'run_start': '13', 'run_end': '16'}
]

# Task sessions in which VNS occurred 
c1_vns_task_session = [1,3,6,8,9,11,14,16,18,20]
formatted_c1_vns_task_session = [f"{num:02}" for num in c1_vns_task_session]

c2_vns_task_session = [2,4,6,8,9,11,13,15]
formatted_c2_vns_task_session = [f"{num:02}" for num in c2_vns_task_session]

# Theoretical stimulation parameters
stimfreq     = 30
ONduration   = 30
OFFduration = 300
minnumpulse  = (ONduration+2) / (1/stimfreq) # Two stimulations added to files to signal start and end of task in the physiological data files
print('The stimulation parameters should be:')
print('Stimulation frequency :', stimfreq, 'Hz')
print('ON duration:', ONduration, 's')
print('OFF duration:', OFFduration, 's')
print('Minimum number of pulses per stimulation', minnumpulse)