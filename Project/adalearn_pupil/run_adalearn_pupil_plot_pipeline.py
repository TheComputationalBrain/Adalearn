"""
This is a metascript that runs plot scripts for the Pupil task in the Adalearn project, producing all plot files.
"""

from pathlib import Path
import subprocess
import datetime

def run_plot(plot_path):
    """
    Function that runs a Python plot script located at `plot_path` as a subprocess.

    Parameters:
    - plot_path : pathlib.Path, the full path to the Python plot script to be executed.

    Behavior:
    - Executes the plot script using a subprocess with `python plot_path`.
    - Sets the current working directory to the plot script’s own folder to ensure relative imports work.
    - Prints start and finish timestamps for tracking.
    - If the plot script fails (non-zero exit), prints the error and stops the pipeline with `exit(1)`.
    """
    print(f"[{datetime.datetime.now()}] Running: {plot_path.name}...")

    try:
        subprocess.run(
            ["python", str(plot_path)],
            check=True,
            cwd=plot_path.parent  # Set working directory to plot's location
        )
        print(f"[{datetime.datetime.now()}] Finished: {plot_path.name}\n")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.datetime.now()}] Error while running: {plot_path.name}")
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    plot_script_names = [
    "adalearn_pupil_plots_task_performance_and_estimate_analysis.py",
    "adalearn_pupil_plots_learning_rate_analysis.py",
    "adalearn_pupil_plots_pupil _and_ peristimulus_analysis.py",
    "adalearn_pupil_plots_pupil _v_lr_and_mediation_analysis.py",
    "adalearn_pupil_plots_lag_assessment.py",
    "adalearn_pupil_plots_shifted_mediation_analysis.py",
    "supplement_dataset/naconf_plot_pupil_regression.py",
    ]
    plots_in_order = [base_dir / plot for plot in plot_script_names]

    for plot in plots_in_order:
        run_plot(plot)
