from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os

from data_preprocess import data_preprocess
from load_model import load_model
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, missing_data_percentage, execution_time

# Set up Flask app
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prediction_model = os.path.join(config["prod_deployment_path"])

# Prediction Endpoint
@app.route("/prediction", methods=['POST'])
def predict():
    model_path = os.path.join(prediction_model, "trainedmodel.pkl")
    model = load_model(model_path)

    # Get JSON payload from request
    data = request.get_json()
    test_data = pd.DataFrame(data)

    try:
        X, _ = data_preprocess(test_data)
        predictions = model.predict(X)
        predictions_list = predictions.tolist()
        return jsonify(predictions=predictions_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Scoring Endpoint
@app.route("/scoring", methods=['GET'])
def score():
    data = os.path.join(dataset_csv_path, "finaldata.csv")
    X_test, y_test = data_preprocess(data)

    model_path = os.path.join(prediction_model, "trainedmodel.pkl")
    model = load_model(model_path)

    try:
        f1 = score_model(X_test, y_test, model)
        return jsonify(f1_score=str(f1))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET'])
def stats():
    column_names = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    stats_names = ["mean", "median", "std"]

    try:
        summary = dataframe_summary()
        stats_summary = {
            column_names[i]: {stats_names[j]: summary[i][j] for j in range(len(stats_names))}
            for i in range(len(column_names))
        }
        return jsonify(stats_summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET'])
def diagnosis():
    column_names = ["corporation", "lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"]

    try:
        exec_time = execution_time()
        na_percentage = missing_data_percentage()
        diag = {
            "execution_time": {
                "ingestion_time": exec_time[0],
                "training_time": exec_time[1]
            },
            "na_percentage": dict(zip(column_names, na_percentage))
        }
        return jsonify(diag)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)