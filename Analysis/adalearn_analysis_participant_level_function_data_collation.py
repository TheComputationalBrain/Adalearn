import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np  

from adalearn_pupil.adalearn_pupil_participant_characteristics import participant_str, condition_str, directory_path

from adalearn_pupil.adalearn_pupil_participant_level_analysis_behav import behavioural_df, mean_abs_error_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_comp_analysis import computational_var_df, filtered_behav_learning_rate_by_cp, averaged_behav_learning_rate_by_cp, filtered_model_learning_rate_by_cp, averaged_model_learning_rate_by_cp, filtered_behav_learning_rate_by_cp_zero_excluded, averaged_behav_learning_rate_by_cp_zero_excluded, behav_v_model_estimate_df, change_point_avg_lrs_df, lr_around_cp_corr_df,  change_point_avg_lrs_df_zero_excluded, lr_around_cp_corr_df_zero_excluded, behav_v_model_lr_df, behav_v_model_lr_df_zero_exclude, lr_evaluation_dict, learning_rate_analysis_dict, learning_rate_analysis_dict_exclude_zero, learning_rate_model_analysis_dict, dkl_regression_df

from adalearn_pupil.adalearn_pupil_participant_level_analysis_pupil import pupil_trig_df, outcome_epoched_pupil_df, outcome_blink_ratio_df, outcome_epochs_to_remove, outcome_epoched_pupil_df_nobc, outcome_blink_ratio_df_nobc, outcome_epochs_to_remove_nobc, pupil_amp_around_cp_df, pupil_avg_around_cp_df, pupil_baseline_around_cp_df  

from adalearn_pupil.adalearn_pupil_participant_level_analysis_regression_pupil import conf_cpp_coef, pupil_regress_df, conf_cpp_coef_nobc, pupil_regress_df_nobc, mediation_output, mediation_output_nobc, mediation_output_zero_excluded, mediation_output_nobc_zero_excluded


def save_dfs_to_csv(participant_str, condition_str,  directory_path, task = 'pupil', analysis_type = ['all']):
    """
    Function to save dataframes to .csv files to be loaded and used for group level analysis.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, path to the participant level directory
    - task: string, defines the task being analysed
    - analysis_type: list of strings, specifies which analyses to save ('pupil', 'computational', 'behavioural', etc.). If you want to save all you can simply put 'all'

    Returns:
    - Saves dataframes as dedicated .csv files in the participant level data directory
    """
    # Ensure the save directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Dictionary mapping variable names to their corresponding DataFrames
    dataframes = {
        # Behavioural analyses
        'behavioural': { 
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_behavioural-df": behavioural_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mean-abs-error-df": mean_abs_error_df,
        },
        # Computational analyses
        'computational': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_computational-var-df": computational_var_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp": filtered_behav_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp": averaged_behav_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-model-lr-by-cp": filtered_model_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-model-lr-by-cp": averaged_model_learning_rate_by_cp,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_filtered-behav-lr-by-cp-zero-excluded": filtered_behav_learning_rate_by_cp_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_averaged-behav-lr-by-cp-zero-excluded": averaged_behav_learning_rate_by_cp_zero_excluded,
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
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_uncertainty-v-lr-model-df": learning_rate_model_analysis_dict['uncertainty_v_lr_df'],
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_dkl-regression-analysis": dkl_regression_df,
        },
        # Pupil analyses
        'pupil': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-trig-df": pupil_trig_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df": outcome_epoched_pupil_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df": outcome_blink_ratio_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-epoched-pupil-df-nobc": outcome_epoched_pupil_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_outcome-blink-ratio-df-nobc": outcome_blink_ratio_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-amplitude-around-cp": pupil_amp_around_cp_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-average-around-cp": pupil_avg_around_cp_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-baseline-around-cp": pupil_baseline_around_cp_df,
        },
        # Pupil v computational variables regression analyses
        'pupil_comp_regression': {
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef": conf_cpp_coef,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df": pupil_regress_df,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_conf-cpp-coef-nobc": conf_cpp_coef_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_pupil-regress-df-nobc": pupil_regress_df_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-amplitude": mediation_output,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-average": mediation_output_nobc,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-amplitude-zero-excluded": mediation_output_zero_excluded,
            f"sub-{participant_str}_ses-{condition_str}_task-Adalearn-{task}_mediation-analysis-pupil-average-zero-excluded": mediation_output_nobc_zero_excluded,
        },
    }

    # Select the correct analysis type
    if analysis_type == "all":
        selected_dataframes = {k: v for category in dataframes.values() for k, v in category.items()}
    elif analysis_type in dataframes:
        selected_dataframes = dataframes[analysis_type]
    else:
        print(f"Invalid analysis_type '{analysis_type}'. Choose from 'pupil', 'computational', 'behavioural', etc. or 'all'.")
        return

    # Save each DataFrame to a CSV file
    save_dir = os.path.join(directory_path, f"ses-{condition_str}", "csv_files")

    os.makedirs(save_dir, exist_ok=True)

    for name, df in selected_dataframes.items():
        if isinstance(df, dict):
            print(f"Skipping {name} (dict object)")
            continue
        file_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")





def create_pdf_page_dictionary(participant_str, condition_str, directory_path, task = 'Pupil'):
    """
    Function to create a dictionary of pages/images/text for the participant PDF report.

    Parammeters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, path to the participant level directory
    - task: string, defines the task being analysed

    Returns:
    - pdf_page_dict: dictionary, where the keys are page numbers and the values are tuples of (image_path, text)
    """
    if task == 'Pupil':
        title = ''
    else:
        title = f'- condition: {condition_str}'

    pdf_page_dict = {
    "page_1": {
        "images": [],
        "text": f"Overview of plots for participant: {participant_str} in Adalearn: {task} ",
        "nrows": 0,
        "ncols": 0,
    },
    "page_2": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_absolute-error-across-task-sessions.png')],
        "text": "Absolute error across task sessions",
        "nrows": 1,
        "ncols": 1,
    },
    "page_3": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_task-session-behav-model-variables.png')],
        "text": "Behavioural and computational variables across one example task session",
        "nrows": 1,
        "ncols": 1,
    },
    "page_4": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_linear-analysis-of-model-estimate-(all-outcomes).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots',  f'sub-{participant_str}_ses-{condition_str}_linear-analysis-of-model-learning-rate-(all-outcomes).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_linear-analysis-of-model-learning-rate-(no-update-trials-removed).png')],
        "text": "Linear relationships between participant's estimate and derived learning rate against the ideal observer model",
        "nrows": 2,
        "ncols": 2,
    },
    "page_5": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_behavioural-learning-rate-by-change-point.png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots',  f'sub-{participant_str}_ses-{condition_str}_behavioural-learning-rate-by-change-point-(no-update-trials-removed).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_model-learning-rate-by-change-point.png')],
        "text": "Behaviourally- and model-derived learning rates around change point",
        "nrows": 2,
        "ncols": 2,
    },
    "page_6": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_evaluation-of-learning-rate.png')],
        "text": "Evaluation of behavioural learning rate",
        "nrows": 1,
        "ncols": 1,
    },
    "page_7": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_uncertainty-parameters-on-behavioural-learning-rate-(all-outcomes).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_uncertainty-parameters-on-behavioural-learning-rate-(no-update-trials-removed).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_uncertainty-parameters-on-model-learning-rate-(all-outcomes).png')],
        "text": "Effect of uncertainty parameters on behavioural learning rate",
        "nrows": 2,
        "ncols": 2,
    },
    "page_8": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-all-outcomes.png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-all-outcomes-heatmap.png')],
        "text": "Pupil response: all stimuli",
        "nrows": 1,
        "ncols": 2,
    },
    "page_9": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-by-colour.png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-by-prior-conf-quartile.png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-by-cpp-quartile.png')],
        "text": "Pupil response: against parameters",
        "nrows": 2,
        "ncols": 2,
    },
    "page_10": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_baseline-corrected-pupil-by-comp-var-peristimulus-regression.png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_non-baseline-corrected-pupil-by-comp-var-peristimulus-regression.png')],
        "text": "Peristimulus pupil regression analysis",
        "nrows": 1,
        "ncols": 2,
    },
    "page_11": {
        "images": [os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-amplitude-uncertainty-lr-mediation-(all-outcomes).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-amplitude-uncertainty-lr-mediation-(no-update-trials-removed).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-mean-uncertainty-lr-mediation-(all-outcomes).png'), os.path.join(directory_path, f'ses-{condition_str}', 'plots', f'sub-{participant_str}_ses-{condition_str}_pupil-mean-uncertainty-lr-mediation-(no-update-trials-removed).png')],
        "text": "Mediation analysis",
        "nrows": 2,
        "ncols": 2,
    },
    }
    

    return pdf_page_dict



def create_participant_pdf(participant_str, condition_str, directory_path, task = 'Pupil'):
    """
    Function to generate a PDF report from the pdf dictionary.

    Parameters:
    - participant_str: string, identifier of participant number (set to 2 numbers)
    - condition_str: string, identifier of condition order (set to 2 numbers) ** May need to change to denote actual condition in Adalearn VNS study **
    - directory_path: Path, path to the participant level directory
    - task: string, defines the task being analysed

    Returns:
    - Saves a PDF report in the participant level directory
    """
    pdf_page_dict = create_pdf_page_dictionary(participant_str, condition_str, directory_path=directory_path, task=task)
    
    # Construct the output path with the file name
    output_path = os.path.join(directory_path, f"ses-{condition_str}", "pdf_file", f"sub-{participant_str}_ses-{condition_str}_plot-overview.pdf")

    # Ensure the output directory exists (but not the file path itself)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Now create the PDF and save it to the file
    with PdfPages(output_path) as pdf:
        for page_key, page_data in pdf_page_dict.items():
            images = page_data["images"]
            text = page_data["text"]
            nrows = page_data["nrows"]
            ncols = page_data["ncols"]

            if nrows == 0 or ncols == 0:
                # Create a text-only page
                fig, ax = plt.subplots(figsize=(8.5, 11))  # A4 size
                ax.axis('off')
                ax.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', wrap=True)
                pdf.savefig(fig)
                plt.close(fig)
            else:
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.5, 11))
                axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
                
                for ax in axes:
                    ax.axis("off")

                for i, image_path in enumerate(images):
                    if i < len(axes) and os.path.exists(image_path):
                        img = mpimg.imread(image_path)
                        axes[i].imshow(img)
                        axes[i].axis("off")
                    else:
                        print(f"Warning: Image {image_path} not found or invalid.")

                # Add text at the bottom of the page
                fig.suptitle(text, fontsize=12, y=0.92, wrap=True)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"PDF saved successfully at {output_path}")