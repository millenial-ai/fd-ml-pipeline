from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

import logging

"""
Split data to train/test subset
"""
def split_data(
    input_path: str, 
    output_path: str, 
    label: str,
    drop_train_label: bool = False,
    drop_val_label: bool = False,
    test_split_ratio: float = 0.2,
    drop_train_headers: bool = True,
    drop_val_headers: bool = True,
    
):
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
        logging.info(f"Splitting {file_path}")
        df = pd.read_csv(file_path)

        # Split the data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=test_split_ratio, random_state=42)
        
        if str(label) != "None":
            if drop_train_label:
                train_df = train_df.drop(label, axis=1)
            if drop_val_label:
                val_df = val_df.drop(label, axis=1)

        os.makedirs(f"{output_path}/train", exist_ok=True)
        os.makedirs(f"{output_path}/val", exist_ok=True)
        # Define the output file paths for train and validation
        train_output_file = os.path.join(output_path, f"train/{file}")
        val_output_file = os.path.join(output_path, f"val/{file}")

        # Save the train and validation sets to CSV files
        train_df.to_csv(train_output_file, index=False, header=not drop_train_headers)
        val_df.to_csv(val_output_file, index=False, header=not drop_val_headers)

        print(f"Split '{file}' into train and validation sets and saved to {train_output_file} and {val_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, help="S3 URI of the input data")
    parser.add_argument("--output-data", type=str, required=True, help="S3 URI for output data")
    parser.add_argument("--test-split-ratio", type=float, help="Ratio of test set")
    parser.add_argument("--label", type=str, default=None, help="Label to drop")
    parser.add_argument("--drop-train-label", action="store_true", help="Drop label column in train data")
    parser.add_argument("--drop-val-label", action="store_true", help="Drop label column in val data")
    parser.add_argument("--drop-train-headers", action="store_true", help="Drop header row in train data")
    parser.add_argument("--drop-val-headers", action="store_true", help="Drop header row in val data")
    args = parser.parse_args()
    
    logging.info(f"SPLITTING DATA {args.input_data} {args.output_data} {args.test_split_ratio}")

    # Process the data and save the selected features
    split_data(
        args.input_data, 
        args.output_data, 
        args.label,
        args.drop_train_label,
        args.drop_val_label,
        args.test_split_ratio,
        args.drop_train_headers,
        args.drop_val_headers
    )
    
