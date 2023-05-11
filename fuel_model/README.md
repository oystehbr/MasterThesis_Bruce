# Fuel model folder ⛽︎

This folder contains everything related to the fuel part of my master's thesis. 

### folder: code 🔐
This subfolder contains the code for data preprocessing, merging of data sets, and creating models. 
The data is not included as it is confidential, but the methodology is provided. 
The most important files within this folder are explained below:

#### file: debugger_merging_data.py ⛙
This script merges the two data sources, fuel- and GPS data, and cleans the data for use in the model. 

#### file: fast_produce_of_model.py 💨
This script creates multiple models by excluding one data set at a time as the test set, as explained in the master's thesis. 
The results and necessary information are reported and saved in the _quick_run_all_ folder. 

#### gps_fuel_data_bruce.ipynb 📝
This Jupyter Notebook contains manual analysis and was used to understand the fuel model and its data. 

#### file: helper.py 🆘
This script contains helper function that are used in multiple other files. 

### folder: quick_run_all:
Contains results and necessary information reported from the runs of _code/fast_produce_of_model.py_

### folder: dummy_figures 💪🏼
This subfolder contains figures saved during the development process. 

