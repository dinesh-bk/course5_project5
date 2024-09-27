# from flask import Flask, session, jsonify, request
# import pandas as pd
# import numpy as np
# import pickle
# import os
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import json

# from data_preprocess import data_preprocess

# import logging

# # Configure logging to display messages and save messages
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         #logging.StreamHandler(),  # This handler sends logs to the console
#         logging.FileHandler('./logs/data_training.log', mode='w')  # This handler writes logs to a file
#     ]
# )

# ###################Load config.json and get path variables
# with open('config.json','r') as f:
#     config = json.load(f) 

# # Training data
# training_dataset = os.path.join(config['output_folder_path'], "finaldata.csv")

# # Preprocess the data 
# X, y = data_preprocess(training_dataset)

# #################Function for training the model
# def train_model(X, y):
#     logging.info("Reading data")
    
#     #use this logistic regression for training
#     logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                                intercept_scaling=1, l1_ratio=None, max_iter=100,
#                                multi_class='auto', n_jobs=None, penalty='l2',
#                                random_state=0, solver='liblinear', tol=0.0001, verbose=0,
#                                warm_start=False)
    
#     logging.info("Training data")
#     #fit the logistic regression to your data
#     model = logit.fit(X,y)

#     #write the trained model to your workspace in a file called trainedmodel.pkl
#     logging.info("Writing model to workspace")
#     logging.info("Checking for model saving directory")
#     # Path to save the model
#     model_path = os.path.join(config['output_model_path']) 
#     if not os.path.exists(model_path):
#         logging.info("Creating model saving directory")
#         os.makedirs(model_path)  # Create the folder if it doesn't exist
#     file_path = os.path.join(model_path, "trainedmodel.pkl")

#     # Write the file 
#     with open(file_path, 'wb') as file:
#         pickle.dump(model, file)
#         logging.info(f"Model successfuly saved in as {file_path}")

# if __name__=='__main__':
#     train_model(X,y)
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from data_preprocess import data_preprocess

import logging

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('./logs/data_training.log', mode='w')
    ]
)

###################Load config.json and get path variables
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

# Training data
training_dataset = os.path.join(config['output_folder_path'], "finaldata.csv")
logging.info(f"Training dataset path: {training_dataset}")

# Preprocess the data 
logging.info("Starting data preprocessing")
X, y = data_preprocess(training_dataset)
logging.info("Data preprocessing completed")

#################Function for training the model
def train_model(X, y):
    logging.info("Initializing logistic regression model")
    
    # Use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)
    
    logging.info("Training the model")
    # Fit the logistic regression to your data
    model = logit.fit(X, y)
    logging.info("Model training completed")

    # Write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Preparing to save the trained model")
    logging.info("Checking for model saving directory")
    model_path = os.path.join(config['output_model_path'])
    if not os.path.exists(model_path):
        logging.info("Model saving directory does not exist. Creating directory.")
        os.makedirs(model_path)  # Create the folder if it doesn't exist
    file_path = os.path.join(model_path, "trainedmodel.pkl")

    logging.info(f"Saving model to {file_path}")
    # Write the file 
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
        logging.info(f"Model successfully saved as {file_path}")

if __name__ == '__main__':
    logging.info("Starting model training process")
    train_model(X, y)
    logging.info("Model training process completed")