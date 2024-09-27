# Dynamic Risk Assessment Model

This project is designed to help a company assess the attrition risk of their 10,000 corporate clients. By predicting which clients are at the highest risk of exiting their contracts, the company can prioritize outreach efforts and minimize revenue loss. The project includes processes for data ingestion, model training, deployment, diagnostics, and regular monitoring of model performance.

## Project Overview

The goal of this project is to develop a machine learning (ML) model to estimate the attrition risk of corporate clients, deploy that model, and set up automated monitoring and reporting processes. Given that the business environment is dynamic, the system is designed to regularly check for new data, retrain the model, and update risk scores. The entire pipeline—from data ingestion to model deployment and monitoring—is automated and runs on a schedule.

### Key Components

1. **Data Ingestion**
   - Automates the process of checking a database for new data related to client behavior and contract status.
   - Compiles all historical and new data into a training dataset.
   - Saves the training dataset to persistent storage.
   - Logs metrics about the data ingestion process for transparency and tracking.

2. **Model Training, Scoring, and Deployment**
   - Trains an ML model to predict client attrition risk using historical data.
   - Scores the trained model on a validation set and saves the predictions.
   - Deploys the model to make future predictions on new data.
   - Stores model artifacts and scoring metrics in persistent storage for future use.

3. **Diagnostics**
   - Summarizes key statistics of the training dataset, including mean, median, standard deviation, and more.
   - Logs performance times for model training and scoring.
   - Checks for dependency changes and package updates to ensure compatibility and stability over time.

4. **Reporting**
   - Automatically generates plots and documents to visualize and report on model performance.
   - Provides an API endpoint to serve model predictions and scoring metrics to other services or teams.

5. **Process Automation**
   - A script is set up to automate all the steps: data ingestion, model training, deployment, diagnostics, and reporting.
   - A cron job is configured to run the script at regular intervals to ensure that the model is always up-to-date with the latest data and can predict client risk accurately.

---

## Installation

To set up the project on your local machine or server:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dynamic-risk-model.git
   cd attrition-risk-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up persistent storage:**
   Ensure you have persistent storage configured for storing datasets, model artifacts, and logs (e.g., AWS S3, Google Cloud Storage, local filesystem).

---

## Project Workflow

### 1. Data Ingestion
The data ingestion script regularly checks the company database for new data, aggregates it with existing data, and stores it in persistent storage. Metrics about the data ingestion process (e.g., rows ingested, ingestion time) are saved to logs for later reference.

### 2. Model Training, Scoring, and Deployment
The model training script runs at regular intervals (via a cron job), retraining the model using the latest dataset. The model is then evaluated on a validation set, and the results (predictions, accuracy, AUC, etc.) are logged. Once training is complete, the model is deployed for future use, and the scoring metrics are saved for monitoring.

### 3. Diagnostics
Diagnostic scripts are run to gather summary statistics on the dataset and to measure the performance of the training and scoring processes. The diagnostic output includes information on data distributions, package dependencies, and model runtime, which are logged for transparency.

### 4. Reporting
Reports are automatically generated, including plots and tables that visualize key metrics such as model accuracy, precision, recall, and feature importance. These reports can be shared with stakeholders for review. The reporting module also includes an API endpoint for returning real-time predictions and model performance metrics.

### 5. Process Automation
The entire pipeline is automated using a cron job that runs the following scripts in sequence:
   1. **Data Ingestion**
   2. **Model Training and Deployment**
   3. **Diagnostics**
   4. **Reporting**



This ensures that the model is continuously updated, deployed, and monitored without manual intervention.

---

## Usage

1. **To Run the Pipeline Manually:**

   After setting up the environment, you can manually execute the scripts in order:

   ```bash
   python ingest_data.py
   python train_model.py
   python diagnostics.py
   python generate_report.py
   ```

2. **To Automate the Pipeline:**

   Set up a cron job to automatically run the pipeline at your desired intervals. For example, to run the pipeline daily at midnight, add this line to your crontab:

   ```bash
   0 0 * * * /usr/bin/python3 /path/to/your/project/run_pipeline.py
   ```

## Making API Call
Ensure the API server is running (e.g., locally at http://0.0.0.0:8000).
Run the Python script:

```bash
python apicalls.py
```
## High Level Overview of Assessment System
![High Level Overview](./fullprocess.jpg)

## Contributing

If you'd like to contribute to the project, feel free to fork the repository and submit a pull request. We welcome new ideas, bug fixes, and additional features!
