import pandas as pd
import os
import sys
import logging
import pytest
import glob

logging.basicConfig(level=logging.DEBUG)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the Python module search path
sys.path.insert(0, parent_dir)

from feature_selection_by_columns import select_features_by_columns

PREFIX = "algorithms/feature_selection/tests"

def test_select_and_export_right_columns(tmp_path, columns=["category","merchant"]):
    input_path = os.path.join(PREFIX, "input/")
    output_path = os.path.join(tmp_path)
    
    select_features_by_columns(input_path, output_path, columns)
    
    for input_file in glob.glob(f"{input_path}/*.csv"):
        original_data = pd.read_csv(input_file)
        output_file = os.path.join(output_path, os.path.basename(input_file))
        converted_data = pd.read_csv(output_file)
    
        assert original_data.shape[0] == converted_data.shape[0]
        
        for column in columns:
            assert column in original_data and column in converted_data