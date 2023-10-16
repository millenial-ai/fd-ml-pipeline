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

from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession

PREFIX = "./ml_pipelines"

file_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python module search path
sys.path.insert(0, file_dir)
sys.path.insert(0, PREFIX)

def calculate_parameters(parameters):
    prefix = os.path.join(parameters.get("bucket_name"), "xgb", parameters.get("pipeline_name"))
    parameters["feature_selection_data"] = os.path.join(prefix, "preprocessed_data", "feature_selection")
    parameters["std_scaling_data"] = os.path.join(prefix, "preprocessed_data", "std_scaling")
    parameters["xgb_splitting_data_train"] = os.path.join(prefix, "preprocessed_data/splitting", "train")
    parameters["xgb_splitting_data_val"] = os.path.join(prefix, "preprocessed_data/splitting", "val")
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
    
    if args.use_cache is None:
        args.use_cache = False
    cache_config = CacheConfig(enable_caching=args.use_cache, expire_after="1y")
    
    calculate_parameters(parameters)
    
    role = ParameterString(name="ExecutionRole", default_value=parameters.get("execution_role"))
    
    # Define pipeline parameters
    input_data = ParameterString(name="InputData", default_value=parameters.get("input_data"))
    
    feature_selection_data = ParameterString(name="FeatureSelectionData", default_value=parameters.get("feature_selection_data"))
    
    std_scaling_data = ParameterString(name="StdScalingData", default_value=parameters.get("std_scaling_data"))
    scalers = ParameterString(name="Scalers", default_value=parameters.get("scalers"))
    
    xgb_splitting_data_train = ParameterString(name="XGBSplittingDataTrain", default_value=parameters.get("xgb_splitting_data_train"))
    xgb_splitting_data_val = ParameterString(name="XGBSplittingDataVal", default_value=parameters.get("xgb_splitting_data_val"))
    
    feature_list = ParameterString(name="FeatureList", default_value=parameters.get("feature_list"))
    label = ParameterString(name="Label", default_value=parameters.get("label"))
    splitting_ratio = ParameterFloat(name="SplittingRatio", default_value=parameters.get("splitting_ratio"))
    
    train_instance_type = ParameterString(name="TrainInstanceType", default_value=parameters.get("train_instance_type"))
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=parameters.get("train_instance_count"))
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=parameters.get("processing_instance_type"))
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=parameters.get("processing_instance_count"))
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    
    from steps import get_feature_selection_step, get_data_scaling_step, get_rcf_data_splitting_step, get_xgb_data_splitting_step, get_rcf_training_step, get_xgb_evaluation_step, get_rcf_register_step
    
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
                destination=parameters.get("feature_selection_data"),
            )
        ],
        feature_list=feature_list,
        label=label,
        processing_instance_type=processing_instance_type,
        processing_instance_count=processing_instance_count,
        role=role,
        cache_config=cache_config,
        step_name="XGB_FeatureSelection"
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
                destination=parameters.get("std_scaling_data"),
            )
        ],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        execution_role=role,
        scalers=scalers,
        cache_config=cache_config,
        step_name="XGB_FeatureScaling"
    )
    
    xgb_data_splitting_step = get_xgb_data_splitting_step(
        parameters,
        sagemaker_session,
        role=role,
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
                destination=parameters.get("xgb_splitting_data_train")
            ),
            ProcessingOutput(
                output_name="data-splitting-output-val",
                source="/opt/ml/processing/output/val",
                s3_upload_mode="EndOfJob",
                destination=parameters.get("xgb_splitting_data_val")
            ),
        ],
        processing_instance_type=processing_instance_type,
        processing_instance_count=processing_instance_count,
        splitting_ratio=splitting_ratio,
        label_to_drop=None,
        cache_config=cache_config,
        step_name="XGB_DataSplitting"
    )
    
    from sagemaker import image_uris
    # Define an XGBoost estimator
    xgboost_estimator = Estimator(
        role=role,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        image_uri=image_uris.retrieve(region=sagemaker_session.boto_region_name, 
                                framework="xgboost", 
                                version="1.0-1"),
        output_path=parameters.get("train_artifact_path"),
        hyperparameters={"objective": "binary:logistic", "num_round": 100},
        sagemaker_session=sagemaker_session,
    )
    
    # Define the training step
    xgb_training_step = TrainingStep(
        name="XGB_TrainingStep",
        estimator=xgboost_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-train"].S3Output.S3Uri, 
                content_type="text/csv",
                distribution="ShardedByS3Key",
            ),
            "validation": sagemaker.inputs.TrainingInput(
                xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-val"].S3Output.S3Uri, 
                content_type="text/csv",
            )
        },
        cache_config=cache_config
    )
    
    xgb_evaluation_step = get_xgb_evaluation_step(
        parameters,
        sagemaker_session,
        role=role,
        step_inputs=[
            ProcessingInput(
                source=xgb_training_step.properties.ModelArtifacts.S3ModelArtifacts,
                # source="s3://fd-ml-pipeline/xgb/pipeline-13-10-2023-14-28-53/training/artifact/pipelines-svyx17quwkt2-XGBoostTrainingStep-mTTBJLCYjd/output/model.tar.gz",
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-val"].S3Output.S3Uri,
                # source="s3://fd-ml-pipeline/xgb/pipeline-09-10-2023-14-11-37/preprocessed_data/splitting/val",
                destination="/opt/ml/processing/test",
            ),
        ],
        step_outputs=[
            ProcessingOutput(
                output_name="evaluation-result", 
                source="/opt/ml/processing/evaluation",
                s3_upload_mode="EndOfJob",
                destination="s3://fd-ml-pipeline/xgb/pipeline-13-10-2023-14-28-53/evaluation"
            ),
        ],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        cache_config=cache_config,
        step_name="XGB_Evaluation"
    )

    xgb_register_step = RegisterModel(
        name="XGBoostModelStep",
        content_types=["text/csv"],
        response_types=["text/csv"],
        estimator=xgboost_estimator,
        model=Model(
            image_uri=xgboost_estimator.image_uri,
            role=role,
            model_data=xgb_training_step.properties.ModelArtifacts.S3ModelArtifacts,
        ),
        approval_status="PendingManualApproval",
        model_package_group_name="xgb",
        cache_config=cache_config,
    )
    
    # Define the pipeline
    pipeline_name = "fd-pipeline-xgb"
    pipeline_steps = [
        feature_selection_step,
        data_scaling_step,
        xgb_data_splitting_step
    ] if args.local else [
        feature_selection_step,
        data_scaling_step,
        xgb_data_splitting_step,
        xgb_training_step,
        xgb_evaluation_step,
        xgb_register_step,
    ]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            role,
            input_data,
            scalers,
            feature_list,
            label,
            splitting_ratio,
            train_instance_type,
            train_instance_count,
            processing_instance_type,
            processing_instance_count,
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
    parser.add_argument("--pipeline-name", type=str, default=None, help="Name for the whole pipeline. This name would be used to specify s3 location")
    parser.add_argument("--use-cache", action="store_true", help="Cache sagemaker pipeline steps")

    args = parser.parse_args()
    main(args)