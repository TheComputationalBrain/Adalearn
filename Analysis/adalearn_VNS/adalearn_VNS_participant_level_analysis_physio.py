""" Main analysis script of Ada-learn VNS physiological data. Script loads and performs various functions on physiological signals collected during data collection to assess VNS activation (and deactivation) and changes in heart rate. ****TO UPDATE**** ****Data for each individual saved to a spreadsheet with each run through.****"""


import pandas as pd
import numpy as np
import neurokit2 as nk
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


from adalearn_analysis_participant_level_functions_physio import load_biopac_data, process_emg_data, biopac_trigger_extraction, extract_task_session_emg_data, process_ecg_data, ecg_markers_by_change_point, vns_activation_id, ecg_response_to_vns

from adalearn_VNS_participant_characteristics import directory_path, participant_str, condition_1_str, condition_2_str, c1_run_ranges, c2_run_ranges, c1_vns_task_sessions, c2_vns_task_sessions

from adalearn_VNS_participant_level_analysis_behav import c1_change_point_indices, c2_change_point_indices



c1_biopac_df, c1_stim_channel, c1_ecg_channel, c1_trig_channel, c1_sampling_rate = load_biopac_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_1_str, run_ranges=c1_run_ranges)

c2_biopac_df, c2_stim_channel, c2_ecg_channel, c2_trig_channel, c2_sampling_rate = load_biopac_data(directory_path=directory_path, participant_str=participant_str, condition_str=condition_2_str, run_ranges=c2_run_ranges)



c1_processed_biopac = process_emg_data(participant_str=participant_str, condition_str=condition_1_str, biopac_data=c1_biopac_df, stim_channels=c1_stim_channel, emg_by_file_plot = False)

c2_processed_biopac = process_emg_data(participant_str=participant_str, condition_str=condition_2_str, biopac_data=c2_biopac_df, stim_channels=c2_stim_channel, emg_by_file_plot = False)



c1_raw_trig_file_df, c1_trig_chan_df, c1_task_session_summary_df, c1_missing_triggers = biopac_trigger_extraction(participant_str=participant_str, condition_str=condition_1_str,biopac_data=c1_biopac_df, run_ranges=c1_run_ranges, plot_triggers_by_channel=False, plot_trigs_by_task_session=False)

c2_raw_trig_file_df, c2_trig_chan_df, c2_task_session_summary_df, c2_missing_triggers = biopac_trigger_extraction(participant_str=participant_str, condition_str=condition_2_str,biopac_data=c2_biopac_df, run_ranges=c2_run_ranges, plot_triggers_by_channel=False, plot_trigs_by_task_session=False)



c1_emg_task_sess_dict = extract_task_session_emg_data(participant_str=participant_str, condition_str=condition_1_str, emg_data_df=c1_processed_biopac, task_session_summary=c1_task_session_summary_df, emg_signal='rectified_emg', plot_task_session_emg=False)

c2_emg_task_sess_dict = extract_task_session_emg_data(participant_str=participant_str, condition_str=condition_2_str, emg_data_df=c2_processed_biopac, task_session_summary=c2_task_session_summary_df, emg_signal='rectified_emg', plot_task_session_emg=False)



c1_vns_start_end_markers = vns_activation_id(participant_str=participant_str, condition_str=condition_1_str, emg_task_sess_dict=c1_emg_task_sess_dict, sampling_rate=c1_sampling_rate, window_size_adjustment=1, vns_task_sessions=c1_vns_task_sessions, threshold=0.20, frequency='900.00-1000.00Hz', plot_start_end_vns_power=True, plot_start_end_vns_emg=True)


c2_vns_start_end_markers = vns_activation_id(participant_str=participant_str, condition_str=condition_2_str, emg_task_sess_dict=c2_emg_task_sess_dict, sampling_rate=c2_sampling_rate, window_size_adjustment=1, vns_task_sessions=c2_vns_task_sessions, threshold=0.20, frequency='900.00-1000.00Hz', plot_start_end_vns_power=False, plot_start_end_vns_emg=False)



c1_ecg_file, c1_ecg_session, c1_rpeak_session_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_1_str, biopac_data=c1_biopac_df, ecg_channels=c1_ecg_channel, task_session_summary=c1_task_session_summary_df, sampling_rate=c1_sampling_rate, low_cutoff=40, high_cutoff=None,  ecg_trace='filtered_ecg', rpeak_artifact_correction=False,  plot_raw_ecg_file=False, plot_ecg_filter_process=False, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False)

c2_ecg_file, c2_ecg_session, c2_rpeak_session_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_2_str, biopac_data=c2_biopac_df, ecg_channels=c2_ecg_channel, task_session_summary=c2_task_session_summary_df, sampling_rate=c2_sampling_rate, low_cutoff=40, high_cutoff=None,  ecg_trace='filtered_ecg', rpeak_artifact_correction=False,  plot_raw_ecg_file=False, plot_ecg_filter_process=False, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False)



c1_r_r_interval_by_cp, c1_r_r_intervals_time_binned_cp = ecg_markers_by_change_point(participant_str=participant_str, condition_str=condition_1_str, rpeaks_session_dict=c1_rpeak_session_dict, change_point_indices=c1_change_point_indices, ecg_parameter='instant_hr', pre=2, post=10, time_bin_size=1, plot_ecg_intervals_by_cp=False)

c2_r_r_interval_by_cp, c2_r_r_intervals_time_binned_cp = ecg_markers_by_change_point(participant_str=participant_str, condition_str=condition_2_str, rpeaks_session_dict=c2_rpeak_session_dict, change_point_indices=c2_change_point_indices, ecg_parameter='r_r_intervals', pre=2, post=10, time_bin_size=1, plot_ecg_intervals_by_cp=False)



c1_vns_hr_summary_df, c1_inside_vns_dict, c1_outside_vns_dict, c1_avg_hr = ecg_response_to_vns(participant_str=participant_str, condition_str=condition_1_str, vns_start_end_markers=c1_vns_start_end_markers, rpeak_session_dict=c1_rpeak_session_dict, plot_hr_with_vns=True)

c2_vns_hr_summary_df, c2_inside_vns_dict, c2_outside_vns_dict, c2_avg_hr = ecg_response_to_vns(participant_str=participant_str, condition_str=condition_2_str, vns_start_end_markers=c2_vns_start_end_markers, rpeak_session_dict=c2_rpeak_session_dict, plot_hr_with_vns=True)