import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

# Configure logging to display messages and save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This handler sends logs to the console
        logging.FileHandler('./logs/data_processing.log', mode='w')  # This handler writes logs to a file
    ]
)

############# Load config.json and get input and output paths
logging.info("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)
logging.info("Configuration file loaded successfully")

input_folder_path = os.path.join(config['input_folder_path'])
input_filenames = os.listdir(input_folder_path)

output_folder_path = os.path.join(config['output_folder_path'])

############# Function for data ingestion
def merge_multiple_dataframe():
    """
    Merges multiple CSV files from a specified directory into a single DataFrame.

    This function reads all CSV files from the directory specified by `input_folder_path`,
    combines them into a single pandas DataFrame, and handles potential file reading errors.
    Duplicate rows are removed from the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all CSV files in the input directory.
    """
    logging.info("Starting to merge multiple dataframes")
    
    # Create an empty dataframe with the specified columns
    final_df = pd.DataFrame(columns=["corporation", "lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"])
    
    # Read input filenames and append data
    for file in input_filenames:
        file_path = os.path.join(input_folder_path, file)
        logging.info(f"Reading file {file_path}")
        df = pd.read_csv(file_path)
        final_df = final_df.append(df, ignore_index=True)
        
    final_df.drop_duplicates(inplace=True)
    logging.info("Finished merging dataframes.")
    
    return final_df

def write_dataset(filename):
    """
    Writes the merged data from multiple CSV files into a single CSV file.

    This function calls `merge_multiple_dataframe` to combine data from CSV files
    in the input directory and writes the resulting DataFrame to a specified CSV file.
    The output file can be saved in a specified output directory if provided.

    Args:
        filename (str): The name of the output file (without directory path).
    """
    logging.info(f"Attempting to write dataset to {output_folder_path}/{filename}.csv")
    
    # Grab the dataframe
    final_df = merge_multiple_dataframe()
    
    # Determine the full path for the output file
    output_file = os.path.join(output_folder_path, filename)
    
    # Write the DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    logging.info(f"Data successfully written to {output_file}")

def save_record(output_filename):
    """
    Saves a record of filenames with a timestamp into a CSV file.

    Args:
        output_filename (str): The name of the CSV file to be saved.
    """
    logging.info("Saving record of the ingestion")
    
    # Get current date and time
    date_time = datetime.now()
    # Format the date as MM/DD/YYYY
    formatted_date = date_time.strftime("%m/%d/%Y")
    
    # Define the path where the file will be saved
    save_path = os.path.join(output_folder_path, output_filename)
    
    # Open the file in write mode
    with open(save_path, "w") as file:
        for filename in input_filenames:
            file.write(f"{formatted_date} {filename}\n")
    
    logging.info(f"Record successfully saved to {save_path}")

if __name__ == '__main__':
    merge_multiple_dataframe()
    write_dataset("finaldata.csv")
    save_record("ingestfiles.txt")