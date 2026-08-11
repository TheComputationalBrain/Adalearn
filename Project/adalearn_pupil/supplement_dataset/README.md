# ADALEARN: Pupil

This folder contains the  materials for running and analyzing the **supplementary dataset** used within the **adalearn_pupil** study.

---
## Data

The raw data used for the supplementary analsysis within the study can be found within the `data/` folder. The folder is structured across participants. 

Derivatives of the raw data used in analyses can be found saved in **.csv** files within the `group_csv_files/` folder. 

---
## Analyses

The scripts for running the analysis are designated by the title convention: **"naconf_group_level_analysis_"** (though one file also contains the modules used for the analysis).  

The analysis of this dataset is run within the meta-level script of main study:  

```python
python run_adalearn_pupil_analysis_pipeline.py
```
---
## Plotting 

All figures generated for the study can be found in the `plots/` folder. The scripts for plotting are designated by the title convention: **"naconf_plots_"**

A meta-script is available for the production of all plots in this analysis: 

```python
python run_adalearn_pupil_plot_pipeline.py
```
