from pathlib import Path
import pandas as pd
from datetime import datetime

from adalearn_analysis_participant_level_functions_participant_info import read_tsv_files
from adalearn_pupil.adalearn_pupil_participant_characteristics import base_dir

group_demographic_df = pd.DataFrame()

base_path = Path(base_dir)

excluded_participants = ['sub-09']
print(f'Participants excluded from analysis: {excluded_participants}')

for sub_folder in base_path.iterdir():
    if sub_folder.name not in excluded_participants:
        tsv_df = read_tsv_files(directory_path=sub_folder)
        group_demographic_df = pd.concat([group_demographic_df, tsv_df], ignore_index=True)

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