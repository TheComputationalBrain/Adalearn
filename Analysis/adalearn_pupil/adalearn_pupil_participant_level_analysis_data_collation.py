"""
This scripts calls and runs functions that extracts the data from the participant level analysis before saving the dataframes to .csv files within the specified participant's data directory. Running this script will run all participant level analysis scripts for a given participant (the user can just run this script to save time running each other script individually - but if they want to save plots, they will need to make sure that the option to plot the data is activated on the relevant script. For example, if i want to plot interpolation of the pupil for each participant, then the option to plot the interpolated pupil data must be activated on the pupil analysis script at the participant level). A note of caution, if all plots are activated it may cause some memory issues given the display of plots using Plotly projected to the browser. If plots have been created, the script also creates a .pdf file for the specified participant that display select plots from the analysis. The script is designed to be run for a single participant at a time.  
"""

from adalearn_pupil.adalearn_pupil_participant_characteristics import participant_str, condition_str, directory_path

from adalearn_analysis_participant_level_function_data_collation import create_participant_pdf, save_dfs_to_csv



save_dfs_to_csv(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path, task='Pupil', analysis_type='all')



create_participant_pdf(participant_str=participant_str, condition_str=condition_str, directory_path=directory_path)