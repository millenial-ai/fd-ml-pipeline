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

from feature_aggregator import add_features

PREFIX = "algorithms/feature_selection/tests"
FEATURES_TO_CHECK = ["age"]

def test_feature_aggregator(tmp_path):
    input_path = os.path.join(PREFIX, "input/")
    output_path = os.path.join(tmp_path)
    
    add_features(input_path, output_path)
    
    for input_file in glob.glob(f"{input_path}/*.csv"):
        original_data = pd.read_csv(input_file)
        output_file = os.path.join(output_path, os.path.basename(input_file))
        converted_data = pd.read_csv(output_file)
        
        assert "age" in converted_data
        assert "part_of_day" in converted_data