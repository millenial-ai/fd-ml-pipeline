#!/bin/bash

# Set the S3 bucket and folder path
ARTIFACT_BUCKET=s3://fd-ml-artifacts
LOCAL_DIR=~/fd-worker/resource/bin

XGB_ARTIFACT_PATH="$ARTIFACT_BUCKET/xgb/"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Use the AWS CLI to sync the S3 bucket with the local directory
aws s3 sync "$XGB_ARTIFACT_PATH" "$LOCAL_DIR"

# Check if the sync command was successful
if [ $? -eq 0 ]; then
  echo "Download completed successfully."
else
  echo "Download failed. Check your AWS CLI configuration and permissions."
fi
