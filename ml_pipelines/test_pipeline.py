import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, TransformStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.estimator import Estimator
from sagemaker.model import Model
import json
import sys
import os
from argparse import ArgumentParser
from datetime import datetime
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession

import boto3
from sagemaker.session import Session
from sagemaker.feature_store.feature_store import FeatureStore

from steps import get_feature_aggregation_step

PREFIX = "./ml_pipelines"

file_dir = os.path.dirname(os.path.abspath(__file__))

def create_dataset(
    feature_group_name: str,
    output_path: str
):
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)
    
    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=region
    )
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime",region_name=region
    )
    
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )
    
    feature_store = FeatureStore(feature_store_session)
    
    transactions_fg = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_store = FeatureStore(feature_store_session)
    builder = feature_store.create_dataset(
        base=transactions_fg,
        output_path=output_path,
    ).to_csv_file()

def main(args):
    parameters = json.load(open(os.path.join(file_dir, "xgb/parameters.json")))

    create_dataset(
        feature_group_name = args.feature_group_name,
        output_path = args.dataset_path
    )
    # SageMaker session and role setup
    if args.local:
        sagemaker_session = LocalPipelineSession()
    else:
        sagemaker_session = PipelineSession()

    role = ParameterString(name="ExecutionRole", default_value=parameters.get("execution_role"))
    
    feature_aggregation_step = get_feature_aggregation_step(
        parameters,
        sagemaker_session,
        step_inputs=[
            ProcessingInput(
                input_name="feature-aggregation-input",
                source=args.dataset_path,
                destination="/opt/ml/processing/input",
                s3_data_distribution_type="FullyReplicated",
            )
        ],
        step_outputs=[
            ProcessingOutput(
                output_name="feature-aggregation-output",
                source="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
                destination=args.aggregation_path,
            )
        ],
        processing_instance_type="ml.t3.medium",
        processing_instance_count=1,
        role=role,
        # cache_config=cache_config,
        step_name="Test_FeatureAggregation"
    )
    
    # Define the pipeline
    pipeline_name = "test-pipeline"
    pipeline_steps = [
        feature_aggregation_step
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            role,
        ],
        steps=pipeline_steps,
        sagemaker_session=sagemaker_session
    )
    
    # Create and execute the pipeline
    pipeline.upsert(role_arn=role.default_value)
    execution = pipeline.start()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run sagemaker local session")
    parser.add_argument("--feature-group-name", default= "transactions-dev", help="Run sagemaker local session")
    parser.add_argument("--dataset-path", default="s3://fd-ml-pipeline/test/datasets", help="Run sagemaker local session")
    parser.add_argument("--aggregation-path", default="s3://fd-ml-pipeline/test/aggregation", help="Run sagemaker local session")

    args = parser.parse_args()
    main(args)