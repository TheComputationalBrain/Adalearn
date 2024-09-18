""" Main analysis script of Ada_learn VNS physiological data. Analysis perfomed for both conditions for each individual. Data for each individual saved to a spreadsheet with each run through."""


import pandas as pd
import numpy as np
import mne
import os
import scipy as sc
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'

from physiological_signal_functions import load_biopac_data, process_emg_data, extract_adalearn_triggers_biopac, extract_task_session_emg_data, identify_vns_from_emg, process_ecg_data

from participant_task_characteristics import participant_str, condition_1_str, condition_2_str, c1_run_ranges, c1_vns_task_session, c2_run_ranges, c2_vns_task_session, stimfreq, ONduration, OFFduration, minnumpulse


# Loading biopac data using function in physiological_signal_functions
biopac_data_c1, stim_channel, ecg_channel, trig_channel = load_biopac_data(participant_str, condition_1_str, c1_run_ranges)
biopac_data_c2, stim_channel, ecg_channel, trig_channel = load_biopac_data(participant_str, condition_2_str, c2_run_ranges)


# Processing EMG data and storing it in a dataframe using function in physiological_signal_functions
processed_biopac_c1 = process_emg_data(biopac_data_c1, stim_channel)
processed_biopac_c2 = process_emg_data(biopac_data_c2, stim_channel)


# Extracting adalearn outcome presentation triggers within bipoac data using function in physiological_signal_functions 
outcome_triggers_c1, c1_trig_summary = extract_adalearn_triggers_biopac(biopac_data_c1, trig_channel, plot=True)
outcome_triggers_c2, c2_trig_summary = extract_adalearn_triggers_biopac(biopac_data_c2, trig_channel, plot=True)


# Reducing EMG data to individual dataframes related to the duration of a task session and storing this within a dictionary using a function within physiological_signal_functions 
c1_task_sess_dict = extract_task_session_emg_data(processed_biopac_c1, c1_trig_summary, plot=True)
c2_task_sess_dict = extract_task_session_emg_data(processed_biopac_c2, c2_trig_summary, plot=True)

# Checking the length of the dictionary and shape of each data frame within the dictionary for condition 1
print(f"In condition {condition_1_str} the number of task sessions is: {len(c1_task_sess_dict)}")
for key, df in c1_task_sess_dict.items():
    print(f"Task Session {key} shape: {df.shape}")

# Checking the length of the dictionary and shape of each data frame within the dictionary for condition 2
print(f"In condition {condition_2_str} the number of task sessions is: {len(c2_task_sess_dict)}")
for key, df in c2_task_sess_dict.items():
    print(f"Task Session {key} shape: {df.shape}")


# Identification of VNS activation and deactivation based on EMG signal using function in physiological_signal_functions
c1_vns_start_times = identify_vns_from_emg(c1_task_sess_dict, c1_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.3)
c2_vns_start_times = identify_vns_from_emg(c2_task_sess_dict, c2_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.1)


# Analysis of heart rate and R-R interval across each task session using function in physiological_signal_functions
c1_rpeaks_dict = process_ecg_data(biopac_data_c1, ecg_channel, c1_trig_summary, plot = True)
c2_rpeaks_dict = process_ecg_data(biopac_data_c2, ecg_channel, c2_trig_summary, plot = True)