import requests
import pandas as pd
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Read test data from CSV
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
save_response_path = os.path.join(config["output_model_path"], "apireturns.txt")

try:
    df = pd.read_csv(test_data_path)
    payload = df.to_dict(orient='records')
    headers = {"Content-Type": "application/json"}

    # Call each API endpoint and store the responses
    response1 = requests.post(URL + "/prediction", json=payload, headers=headers)
    response2 = requests.get(URL + "/scoring")
    response3 = requests.get(URL + "/summarystats")
    response4 = requests.get(URL + "/diagnostics")

    responses = {
        "prediction": response1.json() if response1.status_code == 200 else response1.text,
        "scoring": response2.json() if response2.status_code == 200 else response2.text,
        "summarystats": response3.json() if response3.status_code == 200 else response3.text,
        "diagnostics": response4.json() if response4.status_code == 200 else response4.text
    }

    # Write the responses to a text file
    with open(save_response_path, 'w') as f:
        for key, value in responses.items():
            f.write(f"{key}:\n{json.dumps(value, indent=4)}\n\n")

    # Print responses to the console
    print("Prediction :", responses["prediction"])
    print("Scoring :", responses["scoring"])
    print("Summary :", responses["summarystats"])
    print("Diagnostics :", responses["diagnostics"])

except Exception as e:
    print(f"An error occurred: {e}")