import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import logging

from load_model import load_model
from data_preprocess import data_preprocess

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('./logs/utility_functions.log', mode='w')  # This handler writes logs to a file
    ]
)

# Load config.json and get environment variables
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

# Model path and data path
trained_model = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")

# Dataset to calculate the summary
finaldata_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")

# Dataset to test the model
test_dataset_path = os.path.join(config['test_data_path'], "testdata.csv")

# Split the data
logging.info("Starting data preprocessing")
X_test, y_test = data_preprocess(test_dataset_path)
logging.info("Data preprocessing completed")

################## Function to get model predictions
def model_predictions(X_test):
    """
    Load the deployed model and calculate predictions for the test data.

    Parameters:
    - X_test: The feature matrix for the test data.

    Returns:
    - predictions: The model predictions for the test data.
    """
    logging.info("Loading model for predictions")
    
    model = load_model(trained_model)
    
    logging.info("Making predictions with the loaded model")
    predictions = model.predict(X_test)
    
    if X_test.shape[0] == predictions.shape[0]:
        logging.info("Predictions successfully generated")
        return predictions
    else:
        logging.error("Mismatch in number of samples between X_test and predictions")
        return None

################## Function to get summary statistics
def dataframe_summary():
    """
    Calculate summary statistics (mean, median, std) for numeric columns in the dataset.

    Returns:
    - summary: A list of lists with mean, median, and std for each numeric column.
    """
    logging.info("Calculating summary statistics")
    df = pd.read_csv(finaldata_csv_path)
    
    # Select only numeric columns 
    numeric_df = df.select_dtypes(include=['number'])
    
    summary = []
    for column in numeric_df.columns:
        mean = numeric_df[column].mean()
        median = numeric_df[column].median()
        std = numeric_df[column].std()
        summary.append([mean, median, std])
    
    logging.info("Summary statistics calculated successfully")
    return summary

def missing_data_percentage():
    """
    Calculate the percentage of missing data for each column in the dataset.

    Returns:
    - na_percentage: A list of percentages of missing data for each column.
    """
    logging.info("Calculating missing data percentage")
    df = pd.read_csv(finaldata_csv_path)
    
    na_counts = df.isna().sum()
    
    total_counts = len(df)
    
    na_percentage = list((na_counts / total_counts) * 100)
    
    logging.info("Missing data percentage calculated successfully")
    return na_percentage

################## Function to get timings
def execution_time():
    """
    Calculate the execution time of the ingestion and training scripts.

    Returns:
    - exec_time: A list of execution times for ingestion and training scripts.
    """
    logging.info("Calculating execution time for ingestion and training scripts")
    
    start_time_ingestion = timeit.default_timer()
    subprocess.run(["python3", "ingestion.py"])
    ingestion_time = timeit.default_timer() - start_time_ingestion

    start_time_training = timeit.default_timer()
    subprocess.run(["python3", "training.py"])
    training_time = timeit.default_timer() - start_time_training
    
    exec_time = [ingestion_time, training_time]
    logging.info(f"Ingestion time: {ingestion_time} seconds")
    logging.info(f"Training time: {training_time} seconds")
    return exec_time

################## Function to check dependencies
def outdated_packages_list():
    """
    Run the 'pip list --outdated' command and capture the list of outdated packages.

    Returns:
    - df: A DataFrame containing the list of outdated packages and their versions.
    """
    logging.info("Checking for outdated packages")
    
    result = subprocess.run(["pip", "list", "--outdated", "--format=columns"], capture_output=True, text=True)
    
    # Parse the output
    lines = result.stdout.strip().splitlines()
    package_info = []
    
    # Skip the header
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 3:
            package_name = parts[0]
            current_version = parts[1]
            latest_version = parts[2]
            package_info.append([package_name, current_version, latest_version])
    
    df = pd.DataFrame(package_info, columns=['Package', 'Current Version', 'Latest Version'])
    logging.info("Outdated packages list generated successfully")
    return df

if __name__ == '__main__':
    logging.info("Starting utility functions execution")
    
    # Get model predictions
    predictions = model_predictions(X_test)
    logging.info(f"Model predictions: {predictions}")
    
    # Get summary statistics
    summary = dataframe_summary()
    logging.info(f"Data summary: {summary}")
    
    # Get missing data percentage
    na_percentage = missing_data_percentage()
    logging.info(f"Missing data percentage: {na_percentage}")
    
    # Get execution times
    exec_times = execution_time()
    logging.info(f"Execution times: {exec_times}")
    
    # Get outdated packages list
    outdated_packages = outdated_packages_list()
    logging.info(f"Outdated packages:\n{outdated_packages.to_string(index=False)}")
    
    logging.info("Utility functions execution completed")



    
