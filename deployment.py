from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This handler sends logs to the console
        logging.FileHandler('./logs/model_deployment.log', mode='w')  # This handler writes logs to a file
    ]
)

# Load config.json and get path variables
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

# Ingested data folder 
ingested_file_path = os.path.join(config['output_folder_path'])
logging.info(f"Ingested files path: {ingested_file_path}")

# Practice models folder
latest_score_path = os.path.join(config['output_model_path'])
trained_model_path = os.path.join(config["output_model_path"])
logging.info(f"Latest score path: {latest_score_path}")
logging.info(f"Trained model path: {trained_model_path}")

#################### Function for deployment
def store_model_into_pickle():
    """
    Copies the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file 
    into the deployment directory specified in the configuration.
    """
    # Ingest file path
    ingested_file = os.path.join(ingested_file_path, "ingestfiles.txt")
    logging.info(f"Ingested file path: {ingested_file}")
    
    # Score file path
    score_file = os.path.join(latest_score_path, "latestscore.txt")
    logging.info(f"Score file path: {score_file}")
    
    # Model file path
    model_file = os.path.join(trained_model_path, "trainedmodel.pkl")
    logging.info(f"Model file path: {model_file}")
    
    # Path to save those files 
    prod_deployment_path = os.path.join(config["prod_deployment_path"])
    logging.info(f"Deployment path: {prod_deployment_path}")
    
    if not os.path.exists(prod_deployment_path):
        logging.info("Deployment directory does not exist. Creating directory.")
        os.makedirs(prod_deployment_path)  # Create the folder if it doesn't exist
    
    logging.info("Copying files to deployment directory")
    shutil.copy2(ingested_file, prod_deployment_path)
    shutil.copy2(score_file, prod_deployment_path)                         
    shutil.copy2(model_file, prod_deployment_path)
    logging.info("Files successfully copied to deployment directory")

if __name__ == '__main__':
    logging.info("Starting the model deployment process")
    store_model_into_pickle()
    logging.info("Model deployment process completed")


