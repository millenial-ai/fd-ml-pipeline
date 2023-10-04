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
from argparse import ArgumentParser
from datetime import datetime

from sagemaker.workflow.pipeline_context import PipelineSession

PREFIX = "./ml_pipelines/rcf"

file_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python module search path
sys.path.insert(0, file_dir)

def calculate_parameters(parameters):
    prefix = os.path.join(parameters.get("bucket_name"), parameters.get("pipeline_name"))
    parameters["feature_selection_data"] = os.path.join(prefix, "preprocessed_data", "feature_selection")
    parameters["std_scaling_data"] = os.path.join(prefix, "preprocessed_data", "std_scaling")
    parameters["splitting_data_train"] = os.path.join(prefix, "preprocessed_data", "splitting/train")
    parameters["splitting_data_val"] = os.path.join(prefix, "preprocessed_data", "splitting/val")
    parameters["train_artifact_path"] = os.path.join(prefix, "training", "artifact")


def main(args):
    parameters = json.load(open(os.path.join(file_dir, "./parameters.json")))

    # SageMaker session and role setup
    if args.local:
        sagemaker_session = LocalPipelineSession()
    else:
        sagemaker_session = PipelineSession()
    
    if args.pipeline_name is not None:
        parameters["pipeline_name"] = args.pipeline_name
    else:
        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        parameters["pipeline_name"] = f"pipeline-{now}"
    
    calculate_parameters(parameters)
    
    role = parameters.get("execution_role")  # Replace with your SageMaker role ARN
    
    # Define pipeline parameters
    input_data = ParameterString(name="InputData", default_value=parameters.get("input_data"))
    
    feature_selection_data = ParameterString(name="FeatureSelectionData", default_value=parameters.get("feature_selection_data"))
    
    std_scaling_data = ParameterString(name="StdScalingData", default_value=parameters.get("std_scaling_data"))
    splitting_data_train = ParameterString(name="SplittingDataTrain", default_value=parameters.get("splitting_data_train"))
    splitting_data_val = ParameterString(name="SplittingDataVal", default_value=parameters.get("splitting_data_val"))
    
    feature_list = ParameterString(name="FeatureList", default_value=parameters.get("feature_list"))
    train_instance_type = ParameterString(name="TrainInstanceType", default_value=parameters.get("train_instance_type"))
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=parameters.get("train_instance_count"))
    evaluation_metric_name = ParameterString(name="EvaluationMetricName", default_value=parameters.get("evaluation_metric_name"))
    evaluation_metric_value = ParameterFloat(name="EvaluationMetricValue", default_value=parameters.get("evaluation_metric_value"))
    endpoint_instance_type = ParameterString(name="EndpointInstanceType", default_value=parameters.get("endpoint_instance_type"))
    
    from steps import get_feature_selection_step, get_data_scaling_step, get_rcf_data_splitting_step, get_rcf_training_step, get_rcf_register_step
    
    feature_selection_step = get_feature_selection_step(
        parameters,
        sagemaker_session,
        step_inputs=[
            ProcessingInput(
                input_name="feature-selection-input",
                source=input_data,
                destination="/opt/ml/processing/input",
                s3_data_distribution_type="FullyReplicated",
            )
        ],
        step_outputs=[
            ProcessingOutput(
                output_name="feature-selection-output",
                source="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
                destination=feature_selection_data,
            )
        ],
    )
    
    data_scaling_step = get_data_scaling_step(
        parameters,
        sagemaker_session,
        step_inputs=[
            ProcessingInput(
                source=feature_selection_step.properties.ProcessingOutputConfig.Outputs["feature-selection-output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
                s3_data_distribution_type="FullyReplicated",
                input_name="data-standard-scaling-input",
            )
        ],
        step_outputs=[
            ProcessingOutput(
                output_name="data-standard-scaling-output",
                source="/opt/ml/processing/output",
                s3_upload_mode="EndOfJob",
                destination=std_scaling_data,
            )
        ],
    )
    
    rcf_data_splitting_step = get_rcf_data_splitting_step(
        parameters,
        sagemaker_session,
        step_inputs=[
            ProcessingInput(
                source=data_scaling_step.properties.ProcessingOutputConfig.Outputs["data-standard-scaling-output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
                input_name="data-splitting-input",
            )
        ],
        step_outputs=[
            ProcessingOutput(
                output_name="data-splitting-output-train",
                source="/opt/ml/processing/output/train",
                s3_upload_mode="EndOfJob",
                destination=splitting_data_train
            ),
            ProcessingOutput(
                output_name="data-splitting-output-val",
                source="/opt/ml/processing/output/val",
                s3_upload_mode="EndOfJob",
                destination=splitting_data_val
            ),
        ]
    )
    
    rcf_training_step = get_rcf_training_step(
        parameters, 
        sagemaker_session,
        s3_data=rcf_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-train"].S3Output.S3Uri,
    )
    
    
    from sagemaker.model import Model
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker import image_uris

    model = Model(
        image_uri=image_uris.retrieve(region=sagemaker_session.boto_region_name, 
                                framework="randomcutforest", 
                                version="1"),
        model_data=rcf_training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    model_package_group_name = parameters.get("rcf_model_package_group_name")
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        approval_status=model_approval_status,
        model_package_group_name=model_package_group_name,
    )
    
    rcf_register_step = ModelStep(
        name="RCF_ModelRegistration",
        step_args=register_args
    )
    rcf_register_step = get_rcf_register_step(
        parameters,
        sagemaker_session,
        model_data=rcf_training_step.properties.ModelArtifacts.S3ModelArtifacts
    )
    
    # Define the pipeline
    pipeline_name = "sagemaker-pipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            feature_selection_data,
            std_scaling_data,
            splitting_data_train,
            splitting_data_val,
            feature_list,
            train_instance_type,
            train_instance_count,
            evaluation_metric_name,
            evaluation_metric_value,
            model_approval_status,
            endpoint_instance_type,
        ],
        steps=[
            feature_selection_step,
            data_scaling_step,
            rcf_data_splitting_step,
            rcf_training_step,
            # evaluation_step,
            rcf_register_step,
            # deployment_step,
            # cleanup_step,  # Optional
            # register_step,
        ],
        sagemaker_session=sagemaker_session
    )
    
    # Create and execute the pipeline
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run sagemaker local session")
    parser.add_argument("--pipeline-name", type=str, default=None, help="Name for the whole pipeline. This name would be used to specify s3 location")
    args = parser.parse_args()
    main(args)