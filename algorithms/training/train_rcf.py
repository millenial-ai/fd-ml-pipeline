import argparse
import sagemaker
from sagemaker import RandomCutForest
import os
import boto3
import pandas as pd
import json

s3 = boto3.client("s3")

def get_bucket_name(s3_path):
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    return s3_path.split("/")[0]

"""
List object in s3 path
subset can be 'train', 'val'
"""
def list_s3_files(s3_path):
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    bucket_name, prefix = s3_path.split('/', 1)
    
    # List objects with the specified prefix in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    obj_keys = []
    # Iterate through the objects and print their keys (file names)
    for obj in response.get('Contents', []):
        obj_keys.append(obj['Key'])
    return obj_keys
    

def main(args):
    # Initialize SageMaker RCF estimator
    rcf = RandomCutForest(
        role=args.role,
        instance_count=1,
        instance_type=args.instance_type,
        data_location=args.input_path,
        output_path=args.artifact_path,
        num_samples_per_tree=args.num_samples_per_tree,
        num_trees=args.num_trees,
    )
    
    train_data_prefixes = list_s3_files(args.input_path)
    bucket_name = get_bucket_name(args.input_path)
    
    
    total_training_data = pd.DataFrame([])
    
    for train_prefix in train_data_prefixes:
        if not train_prefix.endswith(".csv"): continue
        train_data = pd.read_csv(f"s3://{bucket_name}/{train_prefix}")
        total_training_data = pd.concat([total_training_data, train_data])
    rcf.fit(rcf.record_set(total_training_data.values), wait=True)
    rcf.create_model()
    
    model_artifact_path = os.path.join(rcf.output_path, rcf.latest_training_job.job_name)
    # model_artifact_path = "test"
    # Print the location of the trained model artifacts
    print(f"Model artifacts are saved at: {model_artifact_path}")
    
    metadata = {
        "model_artifact_path": model_artifact_path
    }
    with open(os.path.join(args.local_metadata_path, "metadata.json"), "w") as fout:
        json.dump(metadata, fout)
        


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Random Cut Forest model using SageMaker")
    parser.add_argument("--input-path", type=str, help="S3 input data path", required=True)
    parser.add_argument("--artifact-path", type=str, help="S3 output artifact path", required=True)
    parser.add_argument("--local-metadata-path", type=str, help="Local metadata output path", required=True)
    
    parser.add_argument("--role", type=str, help="IAM role ARN for SageMaker", required=True)
    parser.add_argument("--instance-type", type=str, help="SageMaker instance type for training", required=True)
    parser.add_argument("--num-samples-per-tree", type=int, default=512, help="Number of samples per tree (default: 512)")
    parser.add_argument("--num-trees", type=int, default=50, help="Number of trees in the forest (default: 50)")
    args = parser.parse_args()
    
    main(args)


