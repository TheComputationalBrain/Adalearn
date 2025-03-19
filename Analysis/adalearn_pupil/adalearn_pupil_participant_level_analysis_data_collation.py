import os

from adalearn_pupil.adalearn_pupil_participant_characteristics import participant_str, condition_str, directory_path

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import behavioural_df, mean_abs_error_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp import computational_var_df, filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp, filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp, filtered_behav_learning_rate_by_cp_zero_excluded, averaged_behav_learning_rate_by_cp_zero_excluded, filtered_model_learning_rate_by_cp_zero_excluded, averaged_model_learning_rate_by_cp_zero_excluded, behav_v_model_estimate_df, change_point_avg_lrs_df, lr_around_cp_corr_df, change_point_avg_lrs_df_zero_excluded, lr_around_cp_corr_df_zero_excluded, behav_v_model_lr_df, behav_v_model_lr_df_zero_exclude, lr_evaluation_dict, learning_rate_analysis_dict, learning_rate_analysis_dict_exclude_zero, dkl_regression_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_pupil import pupil_trig_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc, pupil_amp_around_cp_df, pupil_avg_around_cp_df, pupil_baseline_around_cp_df  

from adalearn_pupil.adalearn_pupil_participant_level_analysis_regression_pupil import conf_cpp_coef, pupil_regress_df, conf_cpp_coef_nobc, pupil_regress_df_nobc, mediation_output, mediation_output_nobc






def save_dfs_to_csv(participant_str, condition_str,  directory_path, task = 'pupil'):
    """
    Function to save dataframes to .csv files to be loaded and used for group level analysis. 
    """
    # Ensure the save directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Dictionary mapping variable names to their corresponding DataFrames
    dataframes = {
        # Behavioural analyses
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behavioural-df": behavioural_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mean-abs-error-df": mean_abs_error_df,
        # Computational analyses
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_computational-var-df": computational_var_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp": filtered_behav_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp": averaged_behav_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-model-lr-by-cp": filtered_model_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-model-lr-by-cp": averaged_model_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-estimate": behav_v_model_estimate_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp-zero-excluded": filtered_behav_learning_rate_by_cp_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp-zero-excluded": averaged_behav_learning_rate_by_cp_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-model-lr-by-cp-zero-excluded": filtered_model_learning_rate_by_cp_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-model-lr-by-cp-zero-excluded": averaged_model_learning_rate_by_cp_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-estimate": behav_v_model_estimate_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp": change_point_avg_lrs_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-corr": lr_around_cp_corr_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-zero-excluded": change_point_avg_lrs_df_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-around-cp-corr-zero-excluded": lr_around_cp_corr_df_zero_excluded,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-lr": behav_v_model_lr_df, 
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behav-v-model-lr-zero-excluded": behav_v_model_lr_df_zero_exclude, 
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_lr-by-session": lr_evaluation_dict['avg_lr_session_df'],
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-df": learning_rate_analysis_dict['uncertainty_v_lr_df'], 
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-df-zero-excluded": learning_rate_analysis_dict_exclude_zero['uncertainty_v_lr_df'],
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_dkl-regression-analysis": dkl_regression_df,
        # Pupil analyses
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-trig-df": pupil_trig_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df": outcome_epoched_pupil_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df": outcome_blink_ratio_df,
        #f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epochs-to-remove": outcome_epochs_to_remove,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df-nobc": outcome_epoched_pupil_df_nobc,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df-nobc": outcome_blink_ratio_df_nobc,
        #f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epochs-to-remove-nobc": outcome_epochs_to_remove_nobc,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-amplitude-around-cp": pupil_amp_around_cp_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-average-around-cp": pupil_avg_around_cp_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-baseline-around-cp": pupil_baseline_around_cp_df,
        # Pupil regression analyses
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef": conf_cpp_coef,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df": pupil_regress_df,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef-nobc": conf_cpp_coef_nobc,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df-nobc": pupil_regress_df_nobc,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-amplitude": mediation_output,
        f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-average": mediation_output_nobc
    }

    # Save each DataFrame to a CSV file
    for name, df in dataframes.items():
        if isinstance(df, dict):
            # Skip dictionaries (or handle differently if needed)
            print(f"Skipping {name} (dict object)")
            continue
        file_path = os.path.join(directory_path, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

save_dfs_to_csv(participant_str=participant_str, condition_str=condition_str, directory_path=desktop_path, task='Pupil')


