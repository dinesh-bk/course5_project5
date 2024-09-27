import pandas as pd

def data_preprocess(input_data):
    if isinstance(input_data, pd.DataFrame):
        test_df = input_data
        # Extract features 
        X = test_df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values
        # Extract target
        y = test_df["exited"].values
        
        return X, y
    elif isinstance(input_data, str):
        test_df = pd.read_csv(input_data)
        # Extract features 
        X = test_df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values
        # Extract target
        y = test_df["exited"].values
        return X, y