""" Main analysis script of Ada-learn VNS physiological data. Script loads and performs various functions on physiological signals collected during data collection to assess VNS activation (and deactivation) and changes in heart rate. ****TO UPDATE**** ****Data for each individual saved to a spreadsheet with each run through.****"""


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


from physiological_signal_functions import load_biopac_data, process_emg_data, biopac_trigger_extraction, extract_task_session_emg_data, process_ecg_data, r_r_interval_by_change_point


from participant_task_information import directory_path, participant_str, condition_1_str, condition_2_str, c1_run_ranges, c2_run_ranges

from behavioural_analysis_test import c1_change_point_indices, c2_change_point_indices



c1_biopac_df, c1_stim_channel, c1_ecg_channel, c1_trig_channel = load_biopac_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_1_str, run_ranges=c1_run_ranges)

c2_biopac_df, c2_stim_channel, c2_ecg_channel, c2_trig_channel = load_biopac_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_2_str, run_ranges=c2_run_ranges)



c1_processed_biopac = process_emg_data(biopac_data=c1_biopac_df, stim_channels=c1_stim_channel, emg_by_file_plot = False)

c2_processed_biopac = process_emg_data(biopac_data=c2_biopac_df, stim_channels=c2_stim_channel, emg_by_file_plot = False)



c1_raw_trig_file_df, c1_trig_chan_df, c1_task_session_summary_df, c1_missing_triggers = biopac_trigger_extraction(biopac_data=c1_biopac_df, run_ranges=c1_run_ranges, plot_triggers_by_channel=False, plot_trigs_by_task_session=False)

c2_raw_trig_file_df, c2_trig_chan_df, c2_task_session_summary_df, c2_missing_triggers = biopac_trigger_extraction(biopac_data=c2_biopac_df, run_ranges=c2_run_ranges, plot_triggers_by_channel=False, plot_trigs_by_task_session=False)



c1_emg_task_sess_dict = extract_task_session_emg_data(participant_str=participant_str, condition_str=condition_1_str, emg_data_df=c1_processed_biopac, task_session_summary=c1_task_session_summary_df, emg_signal='rectified_emg', plot_task_session_emg=False)

c2_emg_task_sess_dict = extract_task_session_emg_data(participant_str=participant_str, condition_str=condition_2_str, emg_data_df=c2_processed_biopac, task_session_summary=c2_task_session_summary_df, emg_signal='rectified_emg', plot_task_session_emg=False)



c1_ecg_file, c1_ecg_session, c1_r_r_interval_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_1_str, biopac_data=c1_biopac_df, ecg_channels=c1_ecg_channel, task_session_summary=c1_task_session_summary_df, ecg_trace='cardiac_cycles_filtered', ectopic_removal_method='malik', interpolation_method='linear', apply_filter=True, plot_raw_ecg_file=False, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False)

c2_ecg_file, c2_ecg_session, c2_r_r_interval_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_2_str, biopac_data=c2_biopac_df, ecg_channels=c2_ecg_channel, task_session_summary=c2_task_session_summary_df, ecg_trace='cardiac_cycles_filtered', ectopic_removal_method='malik', interpolation_method='linear', apply_filter=True, plot_raw_ecg_file=False, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False)



c1_r_r_interval_by_cp, c1_r_r_intervals_time_binned_cp = r_r_interval_by_change_point(participant_str=participant_str, condition_str=condition_1_str, rpeaks_session_dict=c1_r_r_interval_dict, change_point_indices=c1_change_point_indices, pre = 2, post = 10, time_bin_size = 0.5, plot_r_r_intervals_by_cp=False)

c2_r_r_interval_by_cp, c2_r_r_intervals_time_binned_cp = r_r_interval_by_change_point(participant_str=participant_str, condition_str=condition_2_str, rpeaks_session_dict=c2_r_r_interval_dict, change_point_indices=c2_change_point_indices, pre = 2, post = 10, time_bin_size = 0.5, plot_r_r_intervals_by_cp=False)



#c1_vns_start_times = identify_vns_from_emg(task_session_dict=c1_emg_task_sess_dict, vns_task_session_keys=c1_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.3)

#c2_vns_start_times = identify_vns_from_emg(task_session_dict=c2_emg_task_sess_dict, vns_task_session_keys=c2_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.1)
