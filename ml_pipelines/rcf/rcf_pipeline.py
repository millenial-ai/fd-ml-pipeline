import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.steps import ProcessingStep, CreateModelStep
from sagemaker.workflow.pipeline import Pipeline
from datetime import datetime
import json

parameters = json.load(open("parameters.json"))

# SageMaker and AWS session setup
sagemaker_session = sagemaker.Session()
boto_session = boto3.Session()

# Define parameters
input_data = ParameterString(name="InputData", default_value=parameters.get("input_data"))
output_data = ParameterString(name="OutputData", default_value=parameters.get("output_data"))
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.medium")
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# Define feature selection step
data_preprocess_step = ProcessingStep(
    name="DataPreprocessing",
    processor=sagemaker.processing.ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        role="arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457",  # Replace with your SageMaker role ARN
    ),
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
            input_name="input-1",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="processed-data",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=output_data,
        )
    ],
    code="feature_selection_by_columns.py",
    job_arguments=[
        '--input-data', '/opt/ml/processing/input', 
        '--output-data', '/opt/ml/processing/output/output.csv',
        '--selected-features', 'category,merchant,'
    ]
)

# Define feature selection step
feature_selection_step = ProcessingStep(
    name="DataPreprocessing",
    processor=sagemaker.processing.ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        role="arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457",  # Replace with your SageMaker role ARN
    ),
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
            input_name="input-1",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="processed-data",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=output_data,
        )
    ],
    code="feature_selection_by_columns.py",
    job_arguments=[
        '--input-data', '/opt/ml/processing/input', 
        '--output-data', '/opt/ml/processing/output/output.csv',
        '--selected-features', 'category,merchant,'
    ]
)




# Get the current date and time
current_datetime = datetime.now()

# Format it as "YYYY-MM-DD_hh-mm-ss"
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

# Define the SageMaker pipeline
pipeline_name = f"data-preprocessing-pipeline"
print(pipeline_name)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[input_data, output_data, processing_instance_type, processing_instance_count],
    steps=[feature_selection_step],
)

# Create the pipeline
pipeline.upsert(role_arn="arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457")

# Start the pipeline execution
execution = pipeline.start()
print(f"Pipeline execution started with execution ID: {execution.arn}")
