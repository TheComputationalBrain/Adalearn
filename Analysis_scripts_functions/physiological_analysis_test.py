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



c1_emg_task_sess_dict = extract_task_session_emg_data(condition_str=condition_1_str, emg_data_df=c1_processed_biopac, task_session_summary=c1_task_session_summary_df, plot_task_session_emg=False)

c2_emg_task_sess_dict = extract_task_session_emg_data(condition_str=condition_2_str, emg_data_df=c2_processed_biopac, task_session_summary=c2_task_session_summary_df, plot_task_session_emg=False)



c1_ecg_file, c1_ecg_session, c1_r_r_interval_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_1_str, biopac_data=c1_biopac_df, ecg_channels=c1_ecg_channel, task_session_summary=c1_task_session_summary_df, ecg_trace='cardiac_cycles_filtered', ectopic_removal_method='malik', interpolation_method='linear', apply_filter=True, plot_raw_ecg_file=False, plot_rpeaks_session=False, plot_r_r_interval_session=True, plot_instant_hr_session=True)

c2_ecg_file, c2_ecg_session, c2_r_r_interval_dict = process_ecg_data(participant_str=participant_str, condition_str=condition_2_str, biopac_data=c2_biopac_df, ecg_channels=c2_ecg_channel, task_session_summary=c2_task_session_summary_df, ecg_trace='cardiac_cycles_filtered', ectopic_removal_method='malik', interpolation_method='linear', apply_filter=True, plot_raw_ecg_file=False, plot_rpeaks_session=False, plot_r_r_interval_session=False, plot_instant_hr_session=False)



c1_r_r_interval_by_cp, c1_r_r_intervals_time_binned_cp = r_r_interval_by_change_point(participant_str=participant_str, condition_str=condition_1_str, rpeaks_session_dict=c1_r_r_interval_dict, change_point_indices=c1_change_point_indices, pre = 2, post = 10, time_bin_size = 0.5, plot_r_r_intervals_by_cp=True)

c2_r_r_interval_by_cp, c2_r_r_intervals_time_binned_cp = r_r_interval_by_change_point(participant_str=participant_str, condition_str=condition_2_str, rpeaks_session_dict=c2_r_r_interval_dict, change_point_indices=c2_change_point_indices, pre = 2, post = 10, time_bin_size = 0.5, plot_r_r_intervals_by_cp=False)



#c1_vns_start_times = identify_vns_from_emg(task_session_dict=c1_emg_task_sess_dict, vns_task_session_keys=c1_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.3)

#c2_vns_start_times = identify_vns_from_emg(task_session_dict=c2_emg_task_sess_dict, vns_task_session_keys=c2_vns_task_session, sampling_rate = 2000, peak_window_size = 0.5, average_window_size = 5, threshold_increase = 1.1)





import neurokit2 as nk

good_ecg_signal = c1_ecg_session[1].reset_index(drop=True).copy()
bad_ecg_signal = c2_ecg_session[20].reset_index(drop=True).copy()


ECG_NeuroKit_1 = nk.ecg_clean(good_ecg_signal['cardiac_cycles'], sampling_rate=2000, method="neurokit")
ECG_NeuroKit_2 = nk.ecg_clean(bad_ecg_signal['cardiac_cycles'], sampling_rate=2000, method="neurokit")

_, rpeaks_1 = nk.ecg_peaks(good_ecg_signal['cardiac_cycles'], sampling_rate=2000)

_, rpeaks_2 = nk.ecg_peaks(bad_ecg_signal['cardiac_cycles'], sampling_rate=2000)

fig = go.Figure()
fig.add_trace(go.Scatter(x = good_ecg_signal.index, y= good_ecg_signal['cardiac_cycles']))
rpeak_values = good_ecg_signal['cardiac_cycles'].iloc[rpeaks_1['ECG_R_Peaks']].values
fig.add_trace(go.Scatter(x = rpeaks_1['ECG_R_Peaks'], y = rpeak_values, mode='markers'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = bad_ecg_signal.index, y= bad_ecg_signal['cardiac_cycles']))
rpeak_values = bad_ecg_signal['cardiac_cycles'].iloc[rpeaks_2['ECG_R_Peaks']].values
fig.add_trace(go.Scatter(x = rpeaks_2['ECG_R_Peaks'], y = rpeak_values, mode='markers'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = good_ecg_signal.index, y= ECG_NeuroKit_1))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = good_ecg_signal.index, y= ECG_NeuroKit_2))
fig.show()



from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values


def preprocess_r_r_intervals(rpeaks_session_dict, low_rri=300, high_rri=2000, ectopic_removal_method="malik", interpolation_method="linear"):
    """
    Function to process identified r-r intervals. Uses functions from hrvanalysis to remove outliers, ectopic beats and interpolate across nan values. 

    Parameters:
    -

    Returns:
    -
    
    """
    processed_r_r_dict = {}

    for session, df in rpeaks_session_dict.items():
    
        print(f"Processing task session {session}:")   

        r_r_intervals = df['cleaned_r_r_intervals']
        
        #rr_intervals_without_outliers = remove_outliers(rr_intervals=r_r_intervals, low_rri=low_rri, high_rri=high_rri)

        rr_intervals_ectopic_beats = remove_ectopic_beats(rr_intervals=r_r_intervals, method=ectopic_removal_method)

        rr_interval_interpolated = interpolate_nan_values(rr_intervals=rr_intervals_ectopic_beats, interpolation_method=interpolation_method)

        processed_r_r_intervals = pd.DataFrame({
            'interpolated_rr': rr_interval_interpolated, 
            'rpeaks_indices': df['rpeaks_indices'],
            'rpeaks_time': df['rpeaks_time'],
            'rpeak_rel_time': df['rpeak_rel_time']
        }) 

        processed_r_r_dict[session] = processed_r_r_intervals

    return processed_r_r_dict



processed_r_r_dict = preprocess_r_r_intervals(rpeaks_session_dict=c1_r_r_interval_dict, low_rri=0.3, high_rri=2, ectopic_removal_method="malik", interpolation_method="linear")


fig = go.Figure()
fig.add_trace(go.Scatter(x=processed_r_r_dict[1]['rpeak_rel_time'], y=processed_r_r_dict[1]['interpolated_rr']))
fig.show()
