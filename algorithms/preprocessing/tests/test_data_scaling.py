import pandas as pd
import os
import sys
import logging
import pytest
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

logging.basicConfig(level=logging.DEBUG)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the Python module search path
sys.path.insert(0, parent_dir)

from data_scaling import scale_data, parse_scalers_from_str

PREFIX = "algorithms/preprocessing/tests"

def test_data_scaling(
    tmp_path,
    categorical_columns = [],
    numerical_columns = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
):
    input_path = os.path.join(PREFIX, "input/")
    output_path = os.path.join(tmp_path)
    scaler_str = f"StandardScaler[{','.join(numerical_columns)}]"
    
    logging.info(f"Scaling data {input_path} -> {output_path}: {parse_scalers_from_str(scaler_str)}")
    scale_data(input_path, output_path, parse_scalers_from_str(scaler_str))
    
    for input_file in glob.glob(f"{input_path}/*.csv"):
        original_data = pd.read_csv(input_file)
        
        output_file = os.path.join(output_path, os.path.basename(input_file))
        converted_data = pd.read_csv(output_file)
        logging.info(output_file)
        
        # Label encoding
        label_encoders = {}
        for column in categorical_columns:
            assert column in original_data and column in converted_data, f"Column {column} is in both original and converted data"
            le = LabelEncoder()
            feat = le.fit_transform(original_data[column])
            # TODO: Add assertion for categorical columns
        
        # Standardize numeric data
        scaler = StandardScaler()
        numeric_data = original_data[numerical_columns]
        scaled_data = scaler.fit_transform(numeric_data)
        
        for idx, column in enumerate(numerical_columns):
            assert column in original_data and column in converted_data, f"Column {column} is in both original and converted data"
            logging.info(f"Column {column}")
            np.allclose(scaled_data[:,idx], converted_data[column].values, atol=1e-6)

        logging.info(f"{original_data.shape[0]} | {converted_data.shape[0]}")
        assert original_data.shape[0] == converted_data.shape[0], "Original data shape is similar to converted data shape"
        