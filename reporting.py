import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

from diagnostics import model_predictions
from data_preprocess import data_preprocess

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.StreamHandler(),  # This handler sends logs to the console
        logging.FileHandler('./logs/model_reporting.log', mode='w')  # This handler writes logs to a file
    ]
)

############### Load config.json and get path variables
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

# Data in testdata folder
test_data_path = os.path.join(config['test_data_path'])

# Practicemodels folder to save the plot
output_model_path = os.path.join(config['output_model_path'])

############## Function for reporting
def score_model():
    """
    Calculate the confusion matrix using the test data and the deployed model,
    and save the confusion matrix plot to a specified directory.

    - Loads the test data and deployed model.
    - Computes the confusion matrix.
    - Plots the confusion matrix and saves it as a PNG file.
    """
    logging.info("Starting model scoring process")
    
    test_dataset_path = os.path.join(test_data_path, "testdata.csv")
    
    logging.info(f"Loading test data from {test_dataset_path}")
    X_test, y_test = data_preprocess(test_dataset_path)
    logging.info("Test data loaded and preprocessed successfully")
    
    logging.info("Getting model predictions")
    y_pred = model_predictions(X_test)
    
    if y_pred is None:
        logging.error("Model predictions could not be obtained. Exiting.")
        return
    
    # Compute confusion matrix
    logging.info("Computing confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot to a specific folder
    output_dir = output_model_path
    if not os.path.exists(output_dir):
        logging.info(f"Creating directory {output_dir}")
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'confusionmatrix.png')
    plt.savefig(output_path)
    logging.info(f"Confusion matrix plot saved to {output_path}")
    
    logging.info("Model scoring process completed")

if __name__ == '__main__':
    score_model()