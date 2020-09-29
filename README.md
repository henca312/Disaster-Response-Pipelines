# Disaster-Response-Pipelines

This is my repository containing all project files for the Disaster-Response-Pipeline.

# Motivation

I've used data from Figure Eight to develop and establish a machine learning algorithm to classify textual data. This machine learning algorith 
can predict entered messeages in a web application and identify the relevant categories. To realize the machine learning algorithm i've used several notesbooks in Jupyter:

[1] ETL Pipeline Preparation.py
[2] ML Pipeline Preparation.py
[3] train_classifier.py
[4] process_data.py
[5] run.py

# How does the app work?

## Extraction | Transform | Load Process

First of all we need a way to clean up and prepare the provided data to make it useful for further analysis. Within in the file ETL Piple Preparation you can find the code that is used to clean up the provided data and store the result in a database. This coding is reused in the data/process_data.py. There you can digg in to the different steps of the ETL process, from loading, cleaning and finally storing the data into a database.

## Machine Learning Pipeline

After we've provided a clear database that can be used for analysis we need to develeop the Machine Learning Pipeline. In the first step i've created the ML Pipeline Preparation.py. This was used for development. The final Machine Learning algorithm can be found in models/train_classifier.py. This file loads the data, processes the data for the Machine Learning Pipeline and trained by an optimized Model. 

## Result Machine Learning Pipeline

The Machine Learning Pipeline / Model archieves the overall resutlts:
* 
*
*

