"""
This script reads .tsv files within each participants data directory folder to obtain demographic information for the sample in the VNS task of the Adalearn project. A dataframe is created with the demographic information for each participant recorded on each individual session. 
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import adalearn_analysis_group_level_functions_data_utilities as data_utils
import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
from adalearn_analysis_participant_level_functions_participant_info import vns_condition_characteristics


from adalearn_VNS.adalearn_VNS_participant_info_and_parameters import base_dir



base_path = Path(f'{base_dir}/data')
participant_list = group_data_utils.get_participant_numbers(Path(f'{base_dir}/data'))

vns_info_dict_list = []
tsv_df_list = []
excluded_participants_list = []

for participant in participant_list:

    directory_path = Path(f'{base_dir}/data/sub-{str(participant).zfill(2)}')

    print(f'\n\nProcessing participant {str(participant).zfill(2)}...')
    _, _, _, tsv_df, vns_info_dict = vns_condition_characteristics(participant=participant, directory_path=directory_path)

    vns_info_dict_list.append(vns_info_dict)
    tsv_df_list.append(tsv_df)

# Combine all participant tsv dataframes into one
group_demographic_df = pd.concat(tsv_df_list, ignore_index=True)

group_demographic_df.sort_values('ParticipantNumber')

# Ensuring age is correctly stored in the dataframe
group_demographic_df['DOB'] = pd.to_datetime(group_demographic_df['DOB'], format='%Y-%m-%d').copy()
today = datetime.today()
group_demographic_df['Age'] = group_demographic_df['DOB'].apply(lambda dob: (today - dob).days / 365.25)

# Generating a column describing the duration from surgery to each session
group_demographic_df['SurgeryDate'] = pd.to_datetime(group_demographic_df['SurgeryDate'], format='%Y-%m-%d')
group_demographic_df['ExperimentDate'] = pd.to_datetime(group_demographic_df['ExperimentDate'], format='%Y-%m-%d')
group_demographic_df['DurationFromSurgery'] = (
    (group_demographic_df['ExperimentDate'] - group_demographic_df['SurgeryDate']).dt.days
)

# Printing out participant characteristics
unique_participants_df = group_demographic_df.drop_duplicates(subset='ParticipantNumber')

# Sample size
print(f"The sample size is n = {len(unique_participants_df)}")

# Sex split
male_count = (unique_participants_df['Gender'] == 'male').sum()
female_count = (unique_participants_df['Gender'] == 'female').sum()
print(f'The number of males in the sample is {male_count} and the number of females in the sample is {female_count}')

# Age
avg_age = unique_participants_df['Age'].mean()
std_age = unique_participants_df['Age'].std()
print(f'The average age of the sample is {avg_age:.0f} +/- {std_age:.0f} years')

# Handedness
right_hand_count = (unique_participants_df['Handedness'] == 'Right').sum()
left_hand_count = (unique_participants_df['Handedness'] == 'Left').sum()
print(f'The number of right handed participants in the sample is {right_hand_count} and the number of left handed participants in the sample is {left_hand_count}')

# Pathology
pathology = (unique_participants_df['Pathology'].unique())
for pathologies in pathology:
    print(f"The number of participants diagnosed with {pathologies} was: {len(unique_participants_df[unique_participants_df['Pathology']==pathologies])}")

# Duration from surgery
avg_duration = unique_participants_df['DurationFromSurgery'].mean()
std_duration = unique_participants_df['DurationFromSurgery'].std()
print(f'The average duration from surgery to the start of the experiment date {avg_duration:.0f} +/- {std_duration:.0f} days')

high_stim_data = []
low_stim_data = []

# Stimultaion parameters
for participant in group_demographic_df['ParticipantNumber'].unique():
    data = group_demographic_df[group_demographic_df['ParticipantNumber'] == participant]

    max_row = data.loc[data['StimulationIntensity'].idxmax()]
    max_stim = pd.DataFrame(max_row).T
    min_row = data.loc[data['StimulationIntensity'].idxmin()]
    min_stim = pd.DataFrame(min_row).T

    high_stim_data.append(max_stim)
    low_stim_data.append(min_stim)

# Combine high and low stimulation data
high_stim_df = pd.concat(high_stim_data, ignore_index=True)
low_stim_df = pd.concat(low_stim_data, ignore_index=True)

print(f'\nIn the high stimulation condition the stimulation parameters were:')
print(f'Average stimulation intensity: {high_stim_df["StimulationIntensity"].mean():.2f} +/- {high_stim_df["StimulationIntensity"].std():.2f}')
print(f'Average stimulation frequency: {high_stim_df["StimulationFrequency"].mean():.2f} +/- {high_stim_df["StimulationFrequency"].std():.2f}')
print(f'Average stimulation pulse width: {high_stim_df["PulseWidth"].mean():.2f} +/- {high_stim_df["PulseWidth"].std():.2f}')
print(f'Duration stimulator on: {high_stim_df["DurationON"].mean():.2f} +/- {high_stim_df["DurationON"].std():.2f} seconds')
print(f'Duration stimulator off: {high_stim_df["DurationOFF"].mean():.2f} +/- {high_stim_df["DurationOFF"].std():.2f} seconds')

print(f'\nIn the low stimulation condition the stimulation parameters were:')
print(f'Average stimulation intensity: {low_stim_df["StimulationIntensity"].mean():.2f} +/- {low_stim_df["StimulationIntensity"].std():.2f}')
print(f'Average stimulation frequency: {low_stim_df["StimulationFrequency"].mean():.2f} +/- {low_stim_df["StimulationFrequency"].std():.2f}')
print(f'Average stimulation pulse width: {low_stim_df["PulseWidth"].mean():.2f} +/- {low_stim_df["PulseWidth"].std():.2f}')
print(f'Duration stimulator on: {low_stim_df["DurationON"].mean():.2f} +/- {low_stim_df["DurationON"].std():.2f} seconds')
print(f'Duration stimulator off: {low_stim_df["DurationOFF"].mean():.2f} +/- {low_stim_df["DurationOFF"].std():.2f} seconds')

data_utils.save_sample_dataframes_to_csv(dataframes={
    'demographics_df': group_demographic_df,
},
directory_path=base_dir,
condition_str='combined',
task='VNS')