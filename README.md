# ADALEARN Project

This repository contains core materials for running and analyzing the studies in the **ADALEARN** project.

The ADALEARN project investigates mechanisms that drive adaptive changes in learning rate within a dynamically changing probabilistic learning task, with a specific focus on the putative role of **noradrenaline**.

The project includes three separate studies:
- [`adalearn_pupil`](#adalearn_pupil)
- [`adalearn_VNS`](#adalearn_vns)
- [`adalearn_atomoxetine`](#adalearn_atomoxetine)

---

## Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Data](#data)
- [Analyses](#analyses)
- [Requirements](#requirements)
- [Contact](#contact)

---

## Overview

The repository is organized into folders corresponding to each of the three studies. Each study folder includes:

- The task implementation
- Raw and processed data
- Analysis scripts

Because the studies share a common methodological framework, general-purpose functions (used across all studies) are placed at the root level of the repository. External dependencies or helper functions from outside the ADALEARN project are stored in the [`external_reference_functions`](#external_reference_functions) folder.

<div align="center">
  <img src="https://www.dl.dropboxusercontent.com/scl/fi/1hru1i5q3rw7v7j5hfk78/repo_structure.png?rlkey=2h41m8lrr5himi5tbw5zmcut4&st=7xglvhu1&dl=0" width="600">
</div>

---

## Tasks

The same probabilistic learning task is used across all three studies. It is adapted from the original implementation (for more information on the task, please see the original repository):
- Paper: [https://doi.org/10.1162/opmi_a_00139](https://doi.org/10.1162/opmi_a_00139)
- GitHub: [cedricfoucault/ada-learn](https://github.com/cedricfoucault/ada-learn)

In this repository, the task has been extended to support:
- Pupil diameter data collection
- Heartbeat/physiological signal recording

This is achieved via **WebSocket communication**, which allows task events to be streamed in real time for synchronization with physiological recordings.

---

## Data

Each study contains a `data/` folder, which holds behavioral and physiological recordings in their original acquisition format. Data is organized into participant-specific subfolders.

The data is typically split into:
- `behav/`: Behavioral task output
- `eye/`: Eye-tracking or pupil diameter recordings (if applicable)
- `physio/`: Physiological recordings such as ECG, EMG, etc. (used in `adalearn_VNS` and `adalearn_atomoxetine`)

> ⚠️ A cleaned and structured dataset is not yet included, but will be added in future updates to simplify usage.

---

## Analyses

Analysis scripts are provided in the `analysis/` folder of each study. Each script corresponds to a specific figure or result in the project.

- Scripts are written in Python and include docstrings describing their purpose.
- Each script can be run independently.
- Results (e.g., plots, CSV outputs) are saved in structured output directories.

Given the dependency of some scripts on the presence of relevant .csv files, the analysis scripts have a natural order in which they should be run (this is specified in the docstring at the top of each script). To make it easier to run, for certain projects (e.g. adalearn_pupil) a meta_script is available that can produce all of the data by executing the files in the correct order. This meta-script can be run in the terminal after navigating to its directory. For example, in the adalearn_pupil folder: 

```python
python run_adalearn_pupil_analysis_pipeline.py
```

In addition, a meta-script may be available for the production of all plots in the analysis. This can also be run from the terminal. For example, in the adalearn_pupil folder: 

```python
python run_adalearn_pupil_plot_pipeline.py
```

---

## Requirements

To install all required dependencies:

1. Make sure you have **Python 3.11**
2. Run the following from the root of the repository:

```bash
pip install -r requirements.txt
```

---

## Contact

email: agreenhousetucknott@gmail.com
