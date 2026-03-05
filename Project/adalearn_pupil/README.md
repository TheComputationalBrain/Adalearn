
Given the dependency of some scripts on the presence of relevant .csv files, the analysis scripts have a natural order in which they should be run (this is specified in the docstring at the top of each script). To make it easier to run, for certain projects (e.g. adalearn_pupil) a meta_script is available that can produce all of the data by executing the files in the correct order. This meta-script can be run in the terminal after navigating to its directory. For example, in the adalearn_pupil folder: 

```python
python run_adalearn_pupil_analysis_pipeline.py
```

In addition, a meta-script may be available for the production of all plots in the analysis. This can also be run from the terminal. For example, in the adalearn_pupil folder: 

```python
python run_adalearn_pupil_plot_pipeline.py
```
