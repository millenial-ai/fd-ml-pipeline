import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, TransformStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
import json
import sys
import os

PREFIX = "./ml_pipelines/rcf"

file_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python module search path
sys.path.insert(0, file_dir)

parameters = json.load(open(os.path.join(file_dir, "./parameters.json")))

# SageMaker session and role setup
sagemaker_session = LocalPipelineSession()
role = "arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457"  # Replace with your SageMaker role ARN

# Define pipeline parameters
input_data = ParameterString(name="InputData", default_value=parameters.get("input_data"))

feature_selection_data = ParameterString(name="FeatureSelectionData", default_value=parameters.get("feature_selection_data"))

std_scaling_data = ParameterString(name="StdScalingData", default_value=parameters.get("std_scaling_data"))
splitting_data = ParameterString(name="SplittingData", default_value=parameters.get("splitting_data"))

feature_list = ParameterString(name="FeatureList", default_value=parameters.get("feature_list"))
train_instance_type = ParameterString(name="TrainInstanceType", default_value=parameters.get("train_instance_type"))
train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=parameters.get("train_instance_count"))
evaluation_metric_name = ParameterString(name="EvaluationMetricName", default_value=parameters.get("evaluation_metric_name"))
evaluation_metric_value = ParameterFloat(name="EvaluationMetricValue", default_value=parameters.get("evaluation_metric_value"))
endpoint_instance_type = ParameterString(name="EndpointInstanceType", default_value=parameters.get("endpoint_instance_type"))

execution_role = "arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457"

# Step 1: Feature Selection
feature_selection_step = ProcessingStep(
    name="FeatureSelection",
    processor=sagemaker.processing.ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        command=["python3"],
        instance_type=parameters.get("processing_instance_type"),
        instance_count=parameters.get("processing_instance_count"),
        role=execution_role,  # Replace with your SageMaker role ARN
    ),
    inputs=[
        ProcessingInput(
            input_name="feature-selection-input",
            source=input_data,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="feature-selection-output",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=feature_selection_data,
        )
    ],
    code="./algorithms/feature_selection/feature_selection_by_columns.py",
    job_arguments=[
        '--input-data', '/opt/ml/processing/input/', 
        '--output-data', '/opt/ml/processing/output/',
        '--selected-features', parameters.get("features")
    ]
)

data_scaling_processor = sagemaker.processing.ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    instance_type=parameters.get("processing_instance_type"),
    instance_count=parameters.get("processing_instance_count"),
    role=execution_role,  # Replace with your SageMaker role ARN
)

# Step 2: Standard scaling
data_scaling_step = ProcessingStep(
    name="Data Standard Scaling",
    # step_args=data_scaling_step_args
    processor=data_scaling_processor,
    inputs=[
        ProcessingInput(
            source=feature_selection_step.properties.ProcessingOutputConfig.Outputs["feature-selection-output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
            input_name="data-standard-scaling-input",
        )
    ],
    # source_dir="./",
    outputs=[
        ProcessingOutput(
            output_name="data-standard-scaling-output",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=std_scaling_data,
        )
    ],
    code="./algorithms/preprocessing/data_scaling.py",  # Replace with the path to your data preprocessing script
    job_arguments=[
        '--input-data', '/opt/ml/processing/input/', 
        '--output-data', '/opt/ml/processing/output/',
        '--scalers', parameters.get("scalers")
    ]
    # Add other relevant configuration options
)

# Step 3: Data Splitting (Training and Validation)
data_splitting_step = ProcessingStep(
    name="DataSplitting",
    processor=sagemaker.processing.ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        command=["python3"],
        instance_type=parameters.get("processing_instance_type"),
        instance_count=parameters.get("processing_instance_count"),
        role=execution_role,  # Replace with your SageMaker role ARN
    ),
    inputs=[
        ProcessingInput(
            source=data_scaling_step.properties.ProcessingOutputConfig.Outputs["data-standard-scaling-output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
            input_name="data-splitting-input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="data-splitting-output-train",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=splitting_data
        ),
        ProcessingOutput(
            output_name="data-splitting-output-val",
            source="/opt/ml/processing/output",
            s3_upload_mode="EndOfJob",
            destination=splitting_data
        ),
    ],
    code="./algorithms/preprocessing/data_splitting.py",  # Replace with the path to your data splitting script
    job_arguments=[
        "--input-data", "/opt/ml/processing/input/", 
        "--output-data", "/opt/ml/processing/output/",
        "--test-split-ratio", "0.2"
    ],  # Example: Splitting data into 80% training and 20% validation
    # Add other relevant configuration options
)

# # Step 4: Model Training
# training_step = TrainingStep(
#     name="ModelTraining",
#     estimator=sagemaker.estimator.Estimator(
#         image_uri="your-training-container-image",
#         role=role,
#         instance_count=train_instance_count,
#         instance_type=train_instance_type,
#         hyperparameters={
#             "hyperparam1": "value1",
#             "hyperparam2": "value2",
#             # Add other hyperparameters as needed
#         },
#     ),
#     inputs={
#         "train": sagemaker.inputs.TrainingInput(
#             s3_data=data_splitting_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
#             content_type="text/csv",  # Replace with your data format if different
#         ),
#         "validation": sagemaker.inputs.TrainingInput(
#             s3_data=data_splitting_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
#             content_type="text/csv",  # Replace with your data format if different
#         ),
#     },
#     # model_artifacts_path=output_data,  # Save the trained model artifacts to this location
#     # base_job_name="your-training-job-prefix",  # Customize the job name prefix
#     # enable_network_isolation=False,  # Set to True if you want network isolation
#     # Add other relevant configuration options
# )


# # Step 5: Model Evaluation
# evaluation_step = ProcessingStep(
#     name="ModelEvaluation",
#     processor=sagemaker.processing.Processor(
#         role=role,
#         instance_type="ml.t3.medium",
#         instance_count=1,
#         image_uri="your-evaluation-container-image",
#     ),
#     inputs=[
#         sagemaker.processing.ProcessingInput(
#             source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
#             destination="/opt/ml/processing/input/model",
#             input_name="input-1",
#         ),
#         sagemaker.processing.ProcessingInput(
#             source=data_splitting_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
#             destination="/opt/ml/processing/input/validation_data",
#             input_name="input-2",
#         ),
#     ],
#     outputs=[
#         sagemaker.processing.ProcessingOutput(
#             output_name="evaluation_output",
#             source="/opt/ml/processing/output",
#             s3_upload_mode="EndOfJob",
#         )
#     ],
#     code="./algorithms/evaluation/model_evaluation.py",  # Replace with the path to your model evaluation script
#     # Add other relevant configuration options
# )


# # Step 6: Model Deployment
# deployment_step = CreateModelStep(
#     name="ModelDeployment",
#     model=sagemaker.model.Model(
#         image_uri="your-deployment-container-image",
#         model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
#         role=role,
#         sagemaker_session=sagemaker_session,
#     ),
#     # endpoint_name="your-endpoint-name",
#     # tags=[
#     #     {"Key": "key1", "Value": "value1"},
#     #     {"Key": "key2", "Value": "value2"},
#     #     # Add any other tags as needed
#     # ],
#     # enable_network_isolation=False,  # Set to True if you want network isolation
#     # Add other relevant configuration options
# )


# # Step 7: Register the Model
# register_step = RegisterModel(
#     name="RegisterModel",
#     model=sagemaker.model.Model(
#         image_uri="your-registration-container-image",
#         model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
#         role=role,
#         sagemaker_session=sagemaker_session,
#     ),
#     inputs=[
#         sagemaker.processing.ProcessingInput(
#             source=evaluation_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
#             destination="/opt/ml/processing/input/validation_data",
#             input_name="input-22",
#         ),
#     ],
#     content_types=["application/x-recordio-protobuf"],
#     response_types=["application/json"],
#     inference_instances=["ml.t3.medium"],
#     transform_instances=["ml.t3.medium"],
#     model_package_group_name="your-model-package-group-name",
#     approval_status="Approved",
#     tags=[
#         {"Key": "key1", "Value": "value1"},
#         {"Key": "key2", "Value": "value2"},
#         # Add any other tags as needed
#     ],
# )
# print(evaluation_step.properties.expr)
# Step 8: Cleanup (Optional)
cleanup_step = ProcessingStep(
    name="Cleanup",
    processor=sagemaker.processing.Processor(
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        image_uri="your-cleanup-container-image",
    ),
    inputs=[
        # sagemaker.processing.ProcessingInput(
        #     source=feature_selection_step.properties.ProcessingOutputConfig.Outputs["output-1"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/feature_selection",
        #     input_name="input-1",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["output-1"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/preprocessing",
        #     input_name="input-2",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=data_splitting_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/train_data",
        #     input_name="input-3",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=data_splitting_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/validation_data",
        #     input_name="input-4",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        #     destination="/opt/ml/processing/input/trained_model",
        #     input_name="input-5",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=deployment_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/trained_model",
        #     input_name="input-5",
        # ),
        
        # sagemaker.processing.ProcessingInput(
        #     source=evaluation_step.properties.ModelArtifacts.S3ModelArtifacts,
        #     destination="/opt/ml/processing/input/trained_model",
        #     input_name="input-5",
        # ),
        # sagemaker.processing.ProcessingInput(
        #     source=register_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
        #     destination="/opt/ml/processing/input/trained_model",
        #     input_name="input-5",
        # ),
        # Add other input artifacts as needed
    ],
    code="./algorithms/cleanup/cleanup.py",  # Replace with the path to your cleanup script
    # Add other relevant configuration options
)

# Define the pipeline
pipeline_name = "sagemaker-pipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        input_data,
        feature_selection_data,
        std_scaling_data,
        splitting_data,
        feature_list,
        train_instance_type,
        train_instance_count,
        evaluation_metric_name,
        evaluation_metric_value,
        endpoint_instance_type,
    ],
    steps=[
        feature_selection_step,
        data_scaling_step,
        data_splitting_step,
        # training_step,
        # evaluation_step,
        # deployment_step,
        # cleanup_step,  # Optional
        # register_step,
    ],
    sagemaker_session=sagemaker_session
)

# Create and execute the pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()