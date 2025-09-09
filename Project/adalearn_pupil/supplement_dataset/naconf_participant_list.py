from pathlib import Path
import pickle

from adalearn_pupil.supplement_dataset.naconf_participant_info_and_parameters import NACONF_BASE_DIR


filename = 'phasic_tonic_pupil_all_sub.pickle'
directory = Path(f'{NACONF_BASE_DIR}/data/{filename}')

# Open the file in read-binary ('rb') mode
with open(directory, 'rb') as f:
    data = pickle.load(f)

list_of_participants = list(data['phasic_conf'].keys())
print(f"List of participants with pupil data: {list_of_participants}")

usable_participants = [key for key, value in data['correct_data'].items() if value > 50]
print(f"List of participants with usable epochs > 50%: {usable_participants}")