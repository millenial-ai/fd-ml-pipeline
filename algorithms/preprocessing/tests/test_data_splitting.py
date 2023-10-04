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

from data_splitting import split_data

PREFIX = "algorithms/preprocessing/tests"

def test_data_splitting(
    tmp_path,
    categorical_columns = [],
    split_ratio = 0.3
):
    input_path = os.path.join(PREFIX, "input/")
    output_path = os.path.join(tmp_path)
    
    split_data(input_path, output_path, split_ratio)
    
    for input_file in glob.glob(f"{input_path}/*.csv"):
        original_data = pd.read_csv(input_file)
        
        train_output_file = os.path.join(output_path, "train/" + os.path.basename(input_file))
        val_output_file = os.path.join(output_path, "val/" + os.path.basename(input_file))
        logging.info(output_path)
        logging.info(train_output_file)
        assert os.path.exists(train_output_file), f"Train file exists in output dir"
        assert os.path.exists(val_output_file), f"Val file exists in output dir"
        