import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost
import glob
from sklearn.metrics import mean_squared_error

def create_threshold_values():
    return [
        t/100 for t in range(40,80,1)
    ]

TEST_DIR = "/opt/ml/processing/test/*.csv"
EPS = 1e-6
THRESHOLDS = create_threshold_values()

# Calculate classification score based on precision, recall.
# We should be prioritizing high recall more
def calculate_cls_score(precision, recall):
    return precision + 1.5*recall

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))
    
    MSE = STD = TOTAL = TP = TN = FP = FN = 0
    
    CLS_DICT = {} # Dict map from cls_{threshold} to value
    
    prediction_threshold = 0.5    
    for test_path in glob.glob(TEST_DIR):
        df = pd.read_csv(test_path, header=None, skiprows=1)
        y_test = df.iloc[:, -1].to_numpy()
        df.drop(df.columns[-1], axis=1, inplace=True)
    
        X_test = xgboost.DMatrix(df.values)
    
        predictions = model.predict(X_test)
        
        TOTAL += y_test.shape[-1]
        mse = mean_squared_error(y_test, predictions)
        std = np.std(y_test - predictions)
        
        MSE += mse
        STD += std
        
        for prediction_threshold in THRESHOLDS:
            CLS_DICT[prediction_threshold] = {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                "Precision": None,
                "Recall": None,
                "F1": None
            }
            result_dict = CLS_DICT[prediction_threshold]
            result_dict["TP"] += float(np.sum((y_test == 1) & (predictions >= prediction_threshold)))
            result_dict["TN"] += float(np.sum((y_test == 0) & (predictions < prediction_threshold)))
            result_dict["FP"] += float(np.sum((y_test == 0) & (predictions >= prediction_threshold)))
            result_dict["FN"] += float(np.sum((y_test == 1) & (predictions < prediction_threshold)))
    
    for prediction_threshold in THRESHOLDS:
        result_dict = CLS_DICT[prediction_threshold]
        result_dict["Precision"] = result_dict["TP"] / (result_dict["TP"] + result_dict["FP"] + EPS)
        result_dict["Recall"] = result_dict["TP"] / (result_dict["TP"] + result_dict["FN"] + EPS)
        result_dict["F1"] = result_dict["Precision"] * result_dict["Recall"] * 2 / (result_dict["Precision"] + result_dict["Recall"] + EPS)
    
    MSE /= TOTAL
    STD /= TOTAL

    report_dict = {
        "cls_results": CLS_DICT,
        "regression_metrics": {
            "mse": {"value": MSE, "standard_deviation": STD},
        }
    }
    print(json.dumps(report_dict, indent=4))
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict, indent=4))