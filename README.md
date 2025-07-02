## **Project**

This repository contains essential elements for running and analysing the studies within the ADALEARN project. The ADALEARN project seeks to understand mechanisms driving adaptive changes in learning rate within a dynamically changing probability learning task, with a specific focus on the putative role of noradrenaline. The project contains 3 studies: [adalearn_pupil](#adalearn_pupil); [adalearn_VNS](#adalearn_VNS); [adalearn_atomoxetine](#adalearn_atomoxetine). 


## **Table of contents**
- [Overview of the repository](#overview-of-the-repository)
- [Tasks](#tasks)
- [Data](#data)
- [Analyses](#analyses)
- [Requirements](#requirements)
- [Contact](#contact)


## **Overview of the repository**

The project repository contains folders relating to each of the three studies of the project. Each study folder contains essential elements defining the task, data and analysis scripts that can be used to replicate and build upon this work. Within each folder additional elements related to the specific study may also be found (e.g. supplementary datasets, applications, etc). The three studies share a common methodological framework and therefore draw upon general functions that are presented within the top level of the repository. External functions that have not been created specifically within this project, but are essential to the analysis, are also shared within the folder ['external_reference_functions'](#external_reference_functions).     


## **Tasks**

The tasks utilised for each of the studies is held within the task folder. The task is common to all studies and is largely based upon the original presented in previous work (paper: https://doi.org/10.1162/opmi_a_00139; github: https://github.com/cedricfoucault/ada-learn) - for more specific information about the task, please read the README of the original project. 

The original task could be simply executed within a web browser. The tasks have been updated to make them compatible with the collection and analysis of pupil diameter signal and heart beat recordings. To achieve this, the tasks utilise a websocket to extract real-time events from the task protocol within the physiological recordings.   


## **Data**

The data folder within each study contains all of the behavioural data that was collected for that study.

The data is presented in its original acquisition format, which is contained in a participant-specific folder. This data is used within the analysis scripts held in this repository. The data for each participant is split based on the type of data collected: [behav](#behav); [eye](#eye); [physio](#physio) (relating to adalearn_VNS and adlearn_atomoxetine only).

An aggregated structured dataset has not been generated to ease comprehension and manipulation (though will be forthcoming).  


## **Analyses**

The data analyses are implemented by Python scripts stored in each of the studies folder. These allow the user to reproduce all the results and figures. Each script performs one analysis. A docstring describing the analysis is provided at the top of the file.

To run all of the analysis...(to complete)


## **Requirements**

To run the analysis code, you will need to install Python and the necessary Python modules. The required modules are listed in the [`requirements.txt`](requirements.txt) file. You can install them using `pip`
by running the following command in the terminal from the root of the repository:

```
pip install -r requirements.txt
```

The results of the paper were produced using Python 3.11 and the module versions listed in [`requirements.txt`](requirements.txt). Using these versions ensures that the code runs correctly and that the results are exactly reproduced. 


## **Contact**

email: agreenhousetucknott@gmail.com
