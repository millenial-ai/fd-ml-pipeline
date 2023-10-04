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

PREFIX = "algorithms/evaluation/tests"

def test_model_evaluation(
    tmp_path,
    categorical_columns = [],
    numerical_columns = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
):
    pass