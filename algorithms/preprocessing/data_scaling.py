import argparse
import pandas as pd
import glob
import os
import logging
from typing import List, Any
import re
import numpy as np
import logging
import importlib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import zipfile

logging.basicConfig(level=logging.DEBUG)

SUPPORTED_SCALERS = ["StandardScaler", "LabelEncoder"]
def custom_import(name):
    mod = importlib.import_module("sklearn.preprocessing")
    return getattr(mod, name)

"""
Convert from scaler string to mapping of scaler object and their column
Example
Input: sklearn.preprocessing.StandardScaler[col1,col2] sklearn.preprocessing.LabelEncoder[col3,col4]
Output: {
    'sklearn.preprocessing.StandardScaler': {
        'object': <class 'sklearn.preprocessing._data.StandardScaler'>
        'columns': [col1,col2]
    },
    ...
}
"""
def parse_scalers_from_str(scaler_str: str) -> dict:
    # Define a dictionary to store the mappings
    scaler_mapping = {}

    # Define a regex pattern to match scaler and column information
    pattern = r'(\S+)\[(.*?)\]'
    
    logging.info(f"Pattern {pattern} {scaler_str}")
    
    # Find all matches in the input string
    matches = re.findall(pattern, scaler_str)
    
    logging.info(f"Matches {matches}")

    # Iterate over matches and extract information
    for match in matches:
        logging.info(f"MATCH {match}")
        scaler_name, columns_str = match
        scaler_class = None
        columns = []
        
        scaler_class = custom_import(scaler_name)

        # Split columns by ',' and strip whitespace
        columns = [col.strip() for col in columns_str.split(',')]

        # Store the information in the dictionary
        scaler_mapping[scaler_name] = {
            'class': scaler_class,
            'columns': columns
        }
    logging.info(f"Parse scaler returns {scaler_mapping}")
    return scaler_mapping
    
def dump_scaler(scaler, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = open(output_path, 'wb+')
    pickle.dump(scaler, output)
    output.close()
    logging.info(f"Dumping scaler {scaler} to {output_path}")

"""
Input a list of files
Select by column names
Output a list of processed files
"""
def scale_data(
        input_path: str, 
        output_path: str, 
        scalers={}
    ):
    logging.info(f"Scale data {input_path} -> {output_path}")
    logging.info(f"Processing {glob.glob(f'{input_path}/*.csv')}")
    for input_file in glob.glob(f"{input_path}/*.csv"):
        logging.info(f"Scaling input file {input_file}")
        output_file = os.path.join(output_path, os.path.basename(input_file))
        
        logging.info(f"Processing {input_file} -> {output_file}")
        
        # Read the data from S3
        data = pd.read_csv(input_file)
        
        transformed_data = pd.DataFrame(data)

        for key in scalers:
            logging.info(key)
            scaler_item = scalers[key]
            scaler_class, columns = scaler_item["class"], scaler_item["columns"]
            logging.info(f"Scaler {scaler_class}: {columns}")
            for column in columns:
                scaler = scaler_class()
                logging.info(f"Applying scaler {key} to column {column}")
                scaled_data = scaler.fit_transform(np.array(data[column])[:, np.newaxis])
                logging.info(data[column].shape)
                logging.info(transformed_data[column].shape)
                logging.info(scaled_data.shape)
                if len(scaled_data.shape) == 1:
                    transformed_data[column] = scaled_data[:]
                else:
                    transformed_data[column] = scaled_data[:,0]
                
                dump_scaler(scaler, os.path.join(args.artifact_data, f"{key}_{column}.pkl"))
        
        with zipfile.ZipFile(os.path.join(args.artifact_data, "all-scalers.zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the directory and add each file to the ZIP archive
            for foldername, subfolders, filenames in os.walk(args.artifact_data):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    # You can specify a different name or path for the files in the ZIP archive
                    # by modifying the second argument in write().
                    zipf.write(file_path, os.path.relpath(file_path, args.artifact_data))

        logging.info("Saving transformed data")
        # Save the selected data
        transformed_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="S3 URI of the input data")
    parser.add_argument("--output-data", type=str, required=True, help="S3 URI for output data")
    parser.add_argument("--artifact-data", type=str, required=True, help="S3 URI for artifact data (scalers, encoders)")
    parser.add_argument("--scalers", type=str, help="""
        List of scalers and their respective colunns. The format is 
        Example: 'StandardScaler[col1,col2] LabelEncoder[col3,col4]'
    """)
    args = parser.parse_args()
    
    logging.info(args.scalers)

    # Process the data and save the selected features
    scale_data(
        args.input_data, 
        args.output_data, 
        parse_scalers_from_str(args.scalers)
    )
    
