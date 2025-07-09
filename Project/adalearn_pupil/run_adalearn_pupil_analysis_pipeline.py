"""
This is a metascript that runs analysis scripts for the Pupil task in the Adalearn project in the correct order, producing all results which are stored as .csv files.
"""

from pathlib import Path
import subprocess
import datetime

def run_script(script_path):
    """
    Function that runs a Python script located at `script_path` as a subprocess.

    Parameters:
    - script_path : pathlib.Path, the full path to the Python script to be executed.

    Behavior:
    - Executes the script using a subprocess with `python script_path`.
    - Sets the current working directory to the script’s own folder to ensure relative imports work.
    - Prints start and finish timestamps for tracking.
    - If the script fails (non-zero exit), prints the error and stops the pipeline with `exit(1)`.
    """
    print(f"[{datetime.datetime.now()}] Running: {script_path.name}...")

    try:
        subprocess.run(
            ["python", str(script_path)],
            check=True,
            cwd=script_path.parent  # Set working directory to script's location
        )
        print(f"[{datetime.datetime.now()}] Finished: {script_path.name}\n")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.datetime.now()}] Error while running: {script_path.name}")
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    script_names = [
    "adalearn_pupil_group_level_participant_demographics.py",
    "adalearn_pupil_group_level_analysis_behav_task_performance.py",
    "adalearn_pupil_group_level_analysis_model_parameters.py",
    "adalearn_pupil_group_level_analysis_estimate_analysis.py",
    "adalearn_pupil_group_level_analysis_learning_rate_analysis.py",
    "adalearn_pupil_group_level_analysis_pupil_assessment.py",
    "adalearn_pupil_group_level_analysis_pupil_peristimulus_regression.py",
    "adalearn_pupil_group_level_analysis_pupil_v_lr_and_mediation.py",
    "adalearn_pupil_group_level_analysis_individual_volatility.py",
    "adalearn_pupil_group_level_analysis_uncertainty_correlations.py"
]
    scripts_in_order = [base_dir / script for script in script_names]

    for script in scripts_in_order:
        run_script(script)
