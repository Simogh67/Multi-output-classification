# Multioutput-classification

## Summary 
In this repository, a machine learning pipeline is designed to classify real messages that were sent during disaster events (the data comes from Appen, https://appen.com/). 
The task is a multioutput classification problem, and we leverage RandomFroest classifiers to classify the messages.

## File descriptions: 

* process_data.py: This script is for Extract, Transform, and Load process. Here, we read the dataset, clean the data, and then store it in an SQLite database.
* train_classifier.py: This code trains our ML model by using the data stored in the SQLite database. 
* data: This folder contains sample messages and categories datasets in csv format.
* app: the run.py should be used to trigger the web app.

Here is the tree-based file structure of the project. 

* app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app


* data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data


* models

|- train_classifier.py

|- classifier.pkl # saved model

* README.md

## How to run:

Please run the following commonds: 

* running  ETL pipeline:

**python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**

running ML pipeline:

**python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**

Finally run the web app:

**python run.py**
