from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

import logging

# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.1, random_state=42
# )

"""
Split data to train/test subset
"""
def split_data(input_path: str, output_path: str, test_split_ratio: float = 0.2):
    # Check if the input directory exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get a list of CSV files in the input directory
    csv_files = [file for file in os.listdir(input_path) if file.endswith('.csv')]

    # Iterate over each CSV file and split it into train and validation subsets
    for file in csv_files:
        file_path = os.path.join(input_path, file)
        df = pd.read_csv(file_path)

        # Split the data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=test_split_ratio, random_state=42)

        # Define the output file paths for train and validation
        train_output_file = os.path.join(output_path, f"train_{file}")
        val_output_file = os.path.join(output_path, f"val_{file}")

        # Save the train and validation sets to CSV files
        train_df.to_csv(train_output_file, index=False)
        val_df.to_csv(val_output_file, index=False)

        print(f"Split '{file}' into train and validation sets and saved to {train_output_file} and {val_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="S3 URI of the input data")
    parser.add_argument("--output-data", type=str, required=True, help="S3 URI for output data")
    parser.add_argument("--test-split-ratio", type=float, help="Ratio of test set")
    args = parser.parse_args()

    # Process the data and save the selected features
    split_data(
        args.input_data, 
        args.output_data, 
        args.test_split_ratio
    )
    
