"""
This script reads .tsv files within each participants data directory folder to obtain demographic information for the sample in the Pupil task of the Adalearn project. A dataframe is created with the demographic information for each participant. 
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_participant_level_functions_participant_info as participant_info
import adalearn_analysis_group_level_functions_data_utilities as data_utils
import adalearn_analysis_group_level_functions_data_utilities as group_data_utils

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, task_sessions



base_path = Path(f'{base_dir}/data')
participant_list = group_data_utils.get_participant_numbers(Path(f'{base_dir}/data'))
excluded_participants, exclude_participants_behav_cor_df, _ = group_data_utils.check_behavioural_responses(participant_list=participant_list, base_dir=base_dir, condition_str=condition_str, task_sessions=task_sessions, update_percent=10, update_change=0.01, estimate_range=0.5, exclusion_cor_threshold=0.1)

group_demographic_df = pd.DataFrame()

for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{base_dir}/data/sub-{participant_str}')
        tsv_df = participant_info.read_tsv_files(directory_path=directory_path)
        group_demographic_df = pd.concat([group_demographic_df, tsv_df], ignore_index=True)
    else:
        continue

group_demographic_df.sort_values('ParticipantNumber')

print(f'The sample size is n = {len(group_demographic_df)}')

male_count = (group_demographic_df['Gender'] == 'male').sum()
female_count = (group_demographic_df['Gender'] == 'female').sum()
print(f'The number of males in the sample is {male_count} and the number of females in the sample is {female_count}')

right_hand_count = (group_demographic_df['Handedness'] == 'Right').sum()
left_hand_count = (group_demographic_df['Handedness'] == 'Left').sum()
print(f'The number of right handed participants in the sample is {right_hand_count} and the number of left handed participants in the sample is {left_hand_count}')

disorder_count = (group_demographic_df['DisorderHistory'] == 'yes').sum()
print(f' The number of participants reporting a history of psychological/neurological disorders was {disorder_count} out of {len(group_demographic_df)}')

avg_education_dur = group_demographic_df['EducationLevel'].mean()
std_education_dur = group_demographic_df['EducationLevel'].std()
print(f'The mean (+/- std) duration of higher education amongst participants was {avg_education_dur:.0f} +/- {std_education_dur:.0f} years')

math_training_count = (group_demographic_df['MathsTraining'] == 'yes').sum()
print(f' The number of participants reporting additonal math training for education/occupation was {math_training_count} out of {len(group_demographic_df)}')

group_demographic_df['DOB'] = pd.to_datetime(group_demographic_df['DOB'], format='%Y-%m-%d')
today = datetime.today()
group_demographic_df['Age'] = group_demographic_df['DOB'].apply(lambda dob: (today - dob).days / 365.25)
avg_age = group_demographic_df['Age'].mean()
std_age = group_demographic_df['Age'].std()
print(f'The average age of the sample is {avg_age:.0f} +/- {std_age:.0f} years')

data_utils.save_sample_dataframes_to_csv(dataframes={
    'demographics_df': group_demographic_df,
    'exclude_participants_behav_cor_df': exclude_participants_behav_cor_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil')