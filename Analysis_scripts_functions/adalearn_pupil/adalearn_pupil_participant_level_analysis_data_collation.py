import os

from adalearn_pupil_participant_characteristics import tsv_df, participant_str, condition_str, directory_path
from adalearn_pupil_participant_level_analysis_behav import behavioural_df, mean_abs_error_df
from adalearn_pupil_participant_level_analysis_comp import computational_var_df, filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp, filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp, behav_v_model_estimate_corr, behav_v_model_estimate_regress, behav_v_model_lr_corr, behav_v_model_lr_regress, lr_evaluation_dict, learning_rate_analysis_dict
from adalearn_pupil_participant_level_analysis_pupil import pupil_trig_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc
from adalearn_pupil_participant_level_analysis_regression_pupil import conf_cpp_coef, conf_cpp_coef_nobc






def save_dfs_to_csv(participant_str, condition_str, directory_path):
    """
    Function to 
    """
    # Ensure the save directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Dictionary mapping variable names to their corresponding DataFrames
    dataframes = {
        f"sub-{participant_str}_ses-{condition_str}_tsv-df": tsv_df, # Not a dataframe
        f"sub-{participant_str}_ses-{condition_str}_behavioural-df": behavioural_df,
        f"sub-{participant_str}_ses-{condition_str}_mean-abs-error-df": mean_abs_error_df,
        f"sub-{participant_str}_ses-{condition_str}_computational-var-df": computational_var_df,
        f"sub-{participant_str}_ses-{condition_str}_filtered-behav-lr_by_cp": filtered_behav_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_averaged-behav-lr_by_cp": averaged_behav_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_filtered-model-lr_by_cp": filtered_model_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_averaged_model_lr_by_cp": averaged_model_learning_rate_by_cp,
        f"sub-{participant_str}_ses-{condition_str}_behav-v-model-estimate-corr": behav_v_model_estimate_corr,
        f"sub-{participant_str}_ses-{condition_str}_behav-v-model-estimate-regress": behav_v_model_estimate_regress, # Not a dataframe
        f"sub-{participant_str}_ses-{condition_str}_behav-v-model-lr-corr": behav_v_model_lr_corr, # Not a dataframe
        f"sub-{participant_str}_ses-{condition_str}_behav-v-model-lr-regress": behav_v_model_lr_regress, # Not a dataframe
        f"sub-{participant_str}_ses-{condition_str}_lr-by-session": lr_evaluation_dict['avg_lr_session_df'],
        f"sub-{participant_str}_ses-{condition_str}_learning-rate-analysis-dict": learning_rate_analysis_dict, # Not a dataframe
        f"sub-{participant_str}_ses-{condition_str}_pupil-trig-df": pupil_trig_df,
        f"sub-{participant_str}_ses-{condition_str}_outcome-epoched-pupil-df": outcome_epoched_pupil_df,
        f"sub-{participant_str}_ses-{condition_str}_outcome-blink-ratio-df": outcome_blink_ratio_df,
        f"sub-{participant_str}_ses-{condition_str}_outcome-epochs-to-remove": outcome_epochs_to_remove,
        f"sub-{participant_str}_ses-{condition_str}_outcome_epoched_pupil_df_nobc": outcome_epoched_pupil_df_nobc,
        f"sub-{participant_str}_ses-{condition_str}_outcome-blink-ratio-df-nobc": outcome_blink_ratio_df_nobc,
        f"sub-{participant_str}_ses-{condition_str}_outcome-epochs-to-remove-nobc": outcome_epochs_to_remove_nobc,
        f"sub-{participant_str}_ses-{condition_str}_conf-cpp-coef": conf_cpp_coef,
        f"sub-{participant_str}_ses-{condition_str}_conf-cpp-coef-nobc": conf_cpp_coef_nobc
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




