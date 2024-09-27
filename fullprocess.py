import os
import json
import subprocess
import pandas as pd
from datetime import datetime

from ingestion import merge_multiple_dataframe, write_dataset, save_record
from scoring import score_model
from deployment import store_model_into_pickle
from reporting import score_model as report_score
from diagnostics import model_predictions
import logging

from load_model import load_model
from data_preprocess import data_preprocess

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This handler sends logs to the console
        logging.FileHandler('./logs/full_process.log', mode='w')  # This handler writes logs to a file
    ]
)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Read ingested files record
ingested_files_record = os.path.join(prod_deployment_path, 'ingestfiles.txt')

def check_for_new_data():
    """
    Checks for new data files in the input folder that have not been ingested yet.
    If new data is found, it will run the ingestion script to process the new data.
    """
    logging.info("Checking for new data")

    if os.path.exists(ingested_files_record):
        with open(ingested_files_record, 'r') as file:
            ingested_files = [line.strip().split(' ')[1] for line in file.readlines()]
    else:
        ingested_files = []

    current_files = os.listdir(input_folder_path)
    new_files = [file for file in current_files if file not in ingested_files]

    if new_files:
        logging.info(f"New files detected: {new_files}")
        subprocess.run(["python3", "ingestion.py"])
    else:
        logging.info("No new data files detected.")
        return False

    return True

def check_for_model_drift():
    """
    Checks for model drift by comparing the latest F1 score with the F1 score from the new data.
    """
    logging.info("Checking for model drift")

    # Read the latest score from the deployment directory
    score_file = os.path.join(prod_deployment_path, "latestscore.txt")
    if not os.path.exists(score_file):
        logging.error("Score file does not exist.")
        return False

    with open(score_file, 'r') as file:
        old_score = float(file.read().strip())

    # Load new data and compute the new score
    new_data_file = os.path.join(output_folder_path, "finaldata.csv")
    if not os.path.exists(new_data_file):
        logging.error("New data file does not exist.")
        return False
    
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    model = load_model(model_path)
    X_test, y_test = data_preprocess(new_data_file)
    new_score = score_model(X_test, y_test, model)

    logging.info(f"Old score: {old_score}, New score: {new_score}")

    # Compare scores
    if new_score < old_score:
        logging.info("Model drift detected.")
        return True
    else:
        logging.info("No model drift detected.")
        return False

def retrain_model():
    """
    Retrains the model using the most recent data.
    """
    logging.info("Retraining model")
    subprocess.run(["python3", "training.py"])

def re_deploy_model():
    """
    Re-deploys the newly trained model.
    """
    logging.info("Re-deploying model")
    store_model_into_pickle()

def run_diagnostics_and_reporting():
    """
    Runs diagnostics and reporting on the newly deployed model.
    """
    logging.info("Running diagnostics and reporting")
    report_score()

def run_apicalls():
    """
    Runs the API calls script and saves responses.
    """
    logging.info("Running API calls script")
    subprocess.run(["python3", "apicalls.py"])
    
def rename_reporting_files():
    """
    Renames the reporting files for final submission.
    """
    logging.info("Renaming reporting files")
    models_dir = os.path.join(config["output_model_path"])

    # Rename confusion matrix
    confusion_matrix_old = os.path.join(models_dir, "confusionmatrix.png")
    confusion_matrix_new = os.path.join(models_dir, "confusionmatrix2.png")
    if os.path.exists(confusion_matrix_old):
        os.rename(confusion_matrix_old, confusion_matrix_new)
    
    # Rename API returns
    api_returns_old = os.path.join(models_dir, "apireturns.txt")
    api_returns_new = os.path.join(models_dir, "apireturns2.txt")
    if os.path.exists(api_returns_old):
        os.rename(api_returns_old, api_returns_new)
    
if __name__ == '__main__':
    if check_for_new_data():
        if check_for_model_drift():
            retrain_model()
            re_deploy_model()
        run_diagnostics_and_reporting()
        run_apicalls()
        rename_reporting_files()
    else:
        logging.info("No new data to process.")
