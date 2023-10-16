import argparse
import pandas as pd
import glob
import os
import logging
import datetime

def age(data: pd.DataFrame):
    current_year = datetime.datetime.now().year
    
    # Calculate age by subtracting the birth year from the current year
    return current_year - pd.to_datetime(data['dob']).dt.year

def part_of_day(data: pd.DataFrame):
    def categorize_part_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    return pd.to_datetime(data['trans_date_trans_time']).dt.hour.apply(categorize_part_of_day)
    
AGGREGATED_FEATURES = {
    "age": age,
    "part_of_day": part_of_day
}

"""
Input a list of files
Add new features
Output a list of processed files
"""
def add_features(input_path, output_path, selected_features=[]):
    for input_file in glob.glob(f"{input_path}/*.csv"):
        output_file = os.path.join(output_path, os.path.basename(input_file))
        
        logging.info(f"Processing {input_file} -> {output_file}")
        
        # Read the data from S3
        data = pd.read_csv(input_file)
        logging.info(data.columns)
        
        for new_column in AGGREGATED_FEATURES:
            transform_fn = AGGREGATED_FEATURES[new_column]
            data[new_column] = transform_fn(data)
        
        data.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="S3 URI of the input data")
    parser.add_argument("--output-data", type=str, required=True, help="S3 URI for saving the selected data")

    args = parser.parse_args()

    # Process the data and save the selected features
    add_features(args.input_data, args.output_data)
