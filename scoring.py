import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
import json
import logging

from data_preprocess import data_preprocess
from load_model import load_model

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This handler sends logs to the console
        logging.FileHandler('./logs/model_scoring.log', mode='w')  # This handler writes logs to a file
    ]
)

# Load config.json and get path variables
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

# Test data file path   
file_path = os.path.join(config['test_data_path'], "testdata.csv")
logging.info(f"Test data file path: {file_path}")

# Extract the data 
logging.info("Starting data preprocessing")
X_test, y_test = data_preprocess(file_path)
logging.info("Data preprocessing completed")

# Model path
output_model_path = os.path.join(config["output_model_path"])
model_file = os.path.join(output_model_path, "trainedmodel.pkl")
logging.info(f"Model file path: {model_file}")

# Load model
logging.info("Loading the trained model")
model = load_model(model_file)
logging.info("Model loaded successfully")

def score_model(X_test, y_test, model):
    """
    Load the trained model and test data, compute the F1 score on the test data,
    and write the score to the latestscore.txt file.
    """
    # Predict using the model
    logging.info("Predicting with the trained model")
    y_pred = model.predict(X_test)
    # Calculate the F1 score
    logging.info("Calculating F1 score")
    f1 = f1_score(y_test, y_pred)
    logging.info(f"F1 score calculated: {f1}")

    # Write the F1 score to latestscore.txt in output_model_path
    score_file = os.path.join(output_model_path, "latestscore.txt")
    logging.info(f"Writing F1 score to {score_file}")
    with open(score_file, 'w') as file:
        file.write(f"{f1}")
        logging.info("F1 score successfully written to file")
        
    return f1

if __name__ == '__main__':
    logging.info("Starting model scoring process")
    score_model(X_test, y_test, model)
    logging.info("Model scoring process completed")
