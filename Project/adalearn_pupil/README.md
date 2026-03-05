# ADALEARN: Pupil

This folder contains the  materials for running and analyzing the data within the **adalearn_pupil** study.

The **adalearn_pupil** study investigates how pupil responses track uncertainty-driven adaptive changes in learning rate within a dynamically changing probability learning task.  
The preprint for the study can be found here:

**doi: https://doi.org/10.64898/2026.02.26.708148** 

---
## Task

Files for running the task can be found in the `task/` folder. 

---
## Data

The raw data used within the study can be found within the `data/` folder. The folder is structured across participants. 

Derivatives of the raw data used in analyses can be found saved in **.csv** files within the `group_csv_files/` folder. 

Supplementary datasets used within the study are contained in their own dedicated folder: `supplement_dataset/`

> ⚠️ A cleaned and structured dataset is not yet included, but will be added in future updates to simplify usage.

---
## Analyses

The scripts for running the analysis are designated by the title convention: **"adalearn_pupil_group_level_analysis_"**

Given the dependency of some analysis scripts on the presence of relevant .csv files, the analysis scripts have a natural order in which they should be run (this is specified in the docstring at the top of each script). For ease, a meta_script is available that can produce all of the data by executing the files in the correct order. This meta-script can be run in the terminal after navigating to its directory: 

```python
python run_adalearn_pupil_analysis_pipeline.py
```
---
## Plotting 

All figures generated for the study can be found in the `plots/` folder. The scripts for plotting are designated by the title convention: **"adalearn_pupil_plots_"**

A meta-script is available for the production of all plots in the analysis. This can also be run from the terminal. For example, in the adalearn_pupil folder: 

```python
python run_adalearn_pupil_plot_pipeline.py
```
