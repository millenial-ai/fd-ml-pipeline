import argparse
import pandas as pd
import glob
import os
import logging

"""
Input a list of files
Select by column names
Output a list of processed files
"""
def select_features_by_columns(input_path, output_path, selected_features=[], selected_label="is_fraud"):
    for input_file in glob.glob(f"{input_path}/*.csv"):
        output_file = os.path.join(output_path, os.path.basename(input_file))
        
        logging.info(f"Processing {input_file} -> {output_file}")
        
        # Read the data from S3
        data = pd.read_csv(input_file)
        
        selected_columns = selected_features 
        selected_columns += [selected_label]
        # Select the desired features
        selected_data = data[selected_columns]
    
        # Save the selected data
        selected_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="S3 URI of the input data")
    parser.add_argument("--output-data", type=str, required=True, help="S3 URI for saving the selected data")
    parser.add_argument("--selected-features", type=str, required=True, help="Comma-separated list of selected features")
    parser.add_argument("--selected-label", type=str, required=True, help="Comma-separated list of selected features")
    
    args = parser.parse_args()

    # Process the data and save the selected features
    select_features_by_columns(args.input_data, args.output_data, args.selected_features.split(","), args.selected_label)
