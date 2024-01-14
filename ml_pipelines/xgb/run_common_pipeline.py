import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, TransformStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.workflow.model_step import ModelStep
from sagemaker.estimator import Estimator
from sagemaker.model import Model
import json
import sys
import os
from argparse import ArgumentParser
from datetime import datetime

from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.properties import PropertyFile

PREFIX = "./ml_pipelines"

file_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python module search path
sys.path.insert(0, file_dir)
sys.path.insert(0, PREFIX)

from utils import create_dataset

# Get rid of parameters Done x
# Get the steps out of functions x
# Merge xgb and rcf

def main(args):
    parameters = json.load(open(os.path.join(file_dir, "./parameters.json")))

    # SageMaker session and role setup
    if args.local:
        sagemaker_session = LocalPipelineSession()
    else:
        sagemaker_session = PipelineSession()
    
    if args.use_cache is None:
        args.use_cache = False
    cache_config = CacheConfig(enable_caching=args.use_cache, expire_after="1y")
    
    if args.pipeline_execution_name is not None:
        pipeline_execution_name_str = args.pipeline_execution_name
    else:
        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        pipeline_execution_name_str = f"pipeline-{now}"
    role = ParameterString(name="ExecutionRole", default_value=parameters.get("execution_role"))
    pipeline_execution_name = ParameterString(name="PipelineExecutionName", default_value=pipeline_execution_name_str)
    feature_group_name = ParameterString(name="FeatureGroupName", default_value=args.feature_group_name)
    
    # Define pipeline parameters
    input_data = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/dataset"
    
    scaling_data = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/std_scaling"
    
    xgb_splitting_data_train = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/splitting/train"
    xgb_splitting_data_val = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/splitting/val"
    
    rcf_feature_selection_data = f"s3://fd-ml-pipeline/rcf/{pipeline_execution_name_str}/preprocessed_data/feature_selection"
    xgb_feature_selection_data = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/feature_selection"
    
    scaling_artifact = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/scaling_artifact"
    feature_aggregation_data = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/preprocessed_data/feature_aggregation"

    scalers = ParameterString(name="Scalers", default_value=parameters.get("scalers"))
    rcf_feature_list = ParameterString(name="RCFFeatureList", default_value=parameters.get("rcf_feature_list"))
    xgb_feature_list = ParameterString(name="XGBFeatureList", default_value=parameters.get("xgb_feature_list"))
    label = ParameterString(name="Label", default_value=parameters.get("label"))
    splitting_ratio = ParameterFloat(name="SplittingRatio", default_value=parameters.get("splitting_ratio"))
    
    rcf_train_artifact_path = f"s3://fd-ml-pipeline/rcf/{pipeline_execution_name_str}/training/artifact"
    xgb_train_artifact_path = f"s3://fd-ml-pipeline/xgb/{pipeline_execution_name_str}/training/artifact"

    train_instance_type = ParameterString(name="TrainInstanceType", default_value=parameters.get("train_instance_type"))
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=parameters.get("train_instance_count"))
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=parameters.get("processing_instance_type"))
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=parameters.get("processing_instance_count"))
    

    sklearn_image_url = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    
    def get_dataset_creation_step():
        return ProcessingStep(
            name="Dataset_Creation",
            processor=sagemaker.processing.ScriptProcessor(
                # image_uri=sklearn_image_url,
                image_uri="348490654799.dkr.ecr.us-east-1.amazonaws.com/fdml-common:0.0.1",
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role, 
                sagemaker_session=sagemaker_session
            ),
            code="./algorithms/preprocessing/create_dataset.py",
            job_arguments=[
                '--feature-group-name', args.feature_group_name, 
                '--dataset-path', input_data,
                '--region', sagemaker_session.boto_region_name,
            ],
            cache_config=cache_config
        )
        
    def get_feature_aggregation_step():
        return ProcessingStep(
            name="FeatureAggregation",
            processor=sagemaker.processing.ScriptProcessor(
                image_uri=sklearn_image_url,
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role, 
                sagemaker_session=sagemaker_session
            ),
            inputs=[
                ProcessingInput(
                    input_name="feature-aggregation-input",
                    source=input_data,
                    destination="/opt/ml/processing/input",
                    s3_data_distribution_type="FullyReplicated",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="feature-aggregation-output",
                    source="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                    destination=feature_aggregation_data,
                )
            ],
            code="./algorithms/feature_selection/feature_aggregator.py",
            job_arguments=[
                '--input-data', '/opt/ml/processing/input/', 
                '--output-data', '/opt/ml/processing/output/'
            ],
            cache_config=cache_config
        )
    
    def get_data_scaling_step(feature_aggregation_step):
        return ProcessingStep(
            name="Data_Scaling",
            # step_args=data_scaling_step_args
            processor=sagemaker.processing.ScriptProcessor(
                image_uri=sklearn_image_url,
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role,
                sagemaker_session=sagemaker_session
            ),
            inputs=[
                ProcessingInput(
                    input_name="data-standard-scaling-input",
                    source=feature_aggregation_step.properties.ProcessingOutputConfig.Outputs["feature-aggregation-output"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input",
                    s3_data_distribution_type="FullyReplicated",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="data-scaling-output",
                    source="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                    destination=scaling_data,
                ),
                ProcessingOutput(
                    output_name="data-scaling-artifact",
                    source="/opt/ml/processing/artifact",
                    s3_upload_mode="EndOfJob",
                    destination=scaling_artifact,
                )
            ],
            code="./algorithms/preprocessing/data_scaling.py",  # Replace with the path to your data preprocessing script
            job_arguments=[
                '--input-data', '/opt/ml/processing/input/', 
                '--output-data', '/opt/ml/processing/output/',
                '--artifact-data', '/opt/ml/processing/artifact/',
                '--scalers', scalers
            ],
            cache_config=cache_config
        )
        
    def get_rcf_feature_selection_step(data_scaling_step):
        return ProcessingStep(
            name="RCF_FeatureSelection",
            processor=sagemaker.processing.ScriptProcessor(
                image_uri=sklearn_image_url,
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role, 
                sagemaker_session=sagemaker_session
            ),
            inputs=[
                ProcessingInput(
                    input_name="rcf-feature-selection-input",
                    source=data_scaling_step.properties.ProcessingOutputConfig.Outputs["data-scaling-output"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input",
                    s3_data_distribution_type="FullyReplicated",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="rcf-feature-selection-output",
                    source="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                    destination=rcf_feature_selection_data,
                )
            ],
            code="./algorithms/feature_selection/feature_selection_by_columns.py",
            job_arguments=[
                '--input-data', '/opt/ml/processing/input/', 
                '--output-data', '/opt/ml/processing/output/',
                '--selected-features', parameters.get("rcf_feature_list"),
                '--selected-label', "None",
                '--no-header',
            ],
            cache_config=cache_config
        )
        
    def get_xgb_feature_selection_step(data_scaling_step):
        return ProcessingStep(
            name="XGB_FeatureSelection",
            processor=sagemaker.processing.ScriptProcessor(
                image_uri=sklearn_image_url,
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role, 
                sagemaker_session=sagemaker_session
            ),
            inputs=[
                ProcessingInput(
                    input_name="xgb-feature-selection-input",
                    source=data_scaling_step.properties.ProcessingOutputConfig.Outputs["data-scaling-output"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input",
                    s3_data_distribution_type="FullyReplicated",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="xgb-feature-selection-output",
                    source="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                    destination=xgb_feature_selection_data,
                )
            ],
            code="./algorithms/feature_selection/feature_selection_by_columns.py",
            job_arguments=[
                '--input-data', '/opt/ml/processing/input/', 
                '--output-data', '/opt/ml/processing/output/',
                '--selected-features', parameters.get("xgb_feature_list"),
                '--selected-label', label
            ],
            cache_config=cache_config
        )
        
    def get_xgb_data_splitting_step(xgb_feature_selection_step):
        return ProcessingStep(
            name="XGB_DataSplitting",
            processor=sagemaker.processing.ScriptProcessor(
                image_uri=sklearn_image_url,
                command=["python3"],
                instance_type=processing_instance_type,
                instance_count=processing_instance_count,
                role=role,
                sagemaker_session=sagemaker_session
            ),
            inputs=[
                ProcessingInput(
                    source=xgb_feature_selection_step.properties.ProcessingOutputConfig.Outputs["xgb-feature-selection-output"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input",
                    s3_data_distribution_type="FullyReplicated",
                    input_name="xgb-data-splitting-input",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="xgb-data-splitting-output-train",
                    source="/opt/ml/processing/output/train",
                    s3_upload_mode="EndOfJob",
                    destination=xgb_splitting_data_train
                ),
                ProcessingOutput(
                    output_name="xgb-data-splitting-output-val",
                    source="/opt/ml/processing/output/val",
                    s3_upload_mode="EndOfJob",
                    destination=xgb_splitting_data_val
                ),
            ],
            code="./algorithms/preprocessing/data_splitting.py",  # Replace with the path to your data splitting script
            job_arguments=[
                "--input-data", "/opt/ml/processing/input/", 
                "--output-data", "/opt/ml/processing/output/",
                "--test-split-ratio", splitting_ratio.to_string(),
                "--drop-train-headers",
                "--drop-val-headers"
            ],
            cache_config=cache_config
        )
        
    from sagemaker import image_uris
    xgb_image_uri = image_uris.retrieve(
        region=sagemaker_session.boto_region_name, 
        framework="xgboost", 
        version="1.0-1"
    )
    # Define an XGBoost estimator
    xgboost_estimator = Estimator(
        role=role,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        image_uri=xgb_image_uri,
        output_path=xgb_train_artifact_path,
        # hyperparameters={"objective": "binary:logistic", "num_round": 100},
        sagemaker_session=sagemaker_session,
    )
    scale_pos_weight = 50 # set to fixed for now. Can change to output from previous step
    # Setting params similar to original notebook
    xgboost_estimator.set_hyperparameters(
        max_depth=5,
        subsample=0.8,
        num_round=100,
        eta=0.9,
        gamma=10,
        min_child_weight=16,
        silent=0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight
    )
    
    def get_xgb_training_step(xgb_data_splitting_step):
        # Define the training step
        return TrainingStep(
            name="XGB_TrainingStep",
            estimator=xgboost_estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["xgb-data-splitting-output-train"].S3Output.S3Uri, 
                    content_type="text/csv",
                    distribution="ShardedByS3Key",
                ),
                "validation": sagemaker.inputs.TrainingInput(
                    xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["xgb-data-splitting-output-val"].S3Output.S3Uri, 
                    content_type="text/csv",
                )
            },
            # cache_config=CacheConfig(enable_caching=False, expire_after="1y")
            cache_config=cache_config
        )
        
    rcf_image_uri = image_uris.retrieve(
        region=sagemaker_session.boto_region_name, 
        framework="randomcutforest", 
        version="1"
    )
    def get_rcf_training_step(rcf_feature_selection_step):
        rcf = sagemaker.estimator.Estimator(
            rcf_image_uri,
            role,
            output_path=rcf_train_artifact_path,
            instance_count=train_instance_count,
            instance_type=train_instance_type,
            sagemaker_session=sagemaker_session,
        )
        
        rcf.set_hyperparameters(num_samples_per_tree=200, num_trees=50, feature_dim=6)
    
        train_step_args = rcf.fit(
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=rcf_feature_selection_step.properties.ProcessingOutputConfig.Outputs["rcf-feature-selection-output"].S3Output.S3Uri,
                    content_type="text/csv;label_size=0",
                    distribution="ShardedByS3Key",
                )
            },
        )
        return TrainingStep(
            name="RCF_Training",
            step_args=train_step_args,
            cache_config=cache_config
        )
        
    def get_xgb_evaluation_step(xgb_training_step, xgb_data_splitting_step):
        script_eval = ScriptProcessor(
            image_uri=xgb_image_uri,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=1,
            # base_job_name="script-abalone-eval",
            role=role,
            sagemaker_session=sagemaker_session,
        )
        
        evaluation_report = PropertyFile(
            name="EvaluationReport", output_name="evaluation-result", path="evaluation.json"
        )
        
        return ProcessingStep(
            name="XGB_Evaluation",
            step_args=script_eval.run(
                inputs=[
                    ProcessingInput(
                        source=xgb_training_step.properties.ModelArtifacts.S3ModelArtifacts,
                        destination="/opt/ml/processing/model",
                    ),
                    ProcessingInput(
                        source=xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["xgb-data-splitting-output-val"].S3Output.S3Uri,
                        destination="/opt/ml/processing/test",
                    ),
                ],
                outputs=[
                    ProcessingOutput(
                        output_name="evaluation-result", 
                        source="/opt/ml/processing/evaluation",
                        s3_upload_mode="EndOfJob",
                        destination="s3://fd-ml-pipeline/xgb/pipeline-13-10-2023-14-28-53/evaluation"
                    ),
                ],
                code="./algorithms/evaluation/xgb_model_evaluation.py",
            ),
            property_files=[evaluation_report],
            cache_config=cache_config
        )
    
    def get_xgb_register_step(xgb_training_step):
        return RegisterModel(
            name="XGB_ModelRegister",
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
        
    def get_rcf_register_step(rcf_training_step):
        model = Model(
            image_uri=rcf_image_uri,
            model_data=rcf_training_step.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=sagemaker_session,
            role=role
        )
        
        register_args = model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            approval_status="PendingManualApproval",
            model_package_group_name="rcf",
        )
        
        return ModelStep(
            name="RCF_ModelRegister",
            step_args=register_args,
        )
    
    dataset_creation_step = get_dataset_creation_step()
    
    feature_aggregation_step = get_feature_aggregation_step()
    feature_aggregation_step.add_depends_on([dataset_creation_step])
    
    data_scaling_step = get_data_scaling_step(feature_aggregation_step)
    
    rcf_feature_selection_step = get_rcf_feature_selection_step(data_scaling_step)
    
    xgb_feature_selection_step = get_xgb_feature_selection_step(data_scaling_step)
    
    xgb_data_splitting_step = get_xgb_data_splitting_step(xgb_feature_selection_step)
    # xgb_data_splitting_step = ProcessingStep(
    #     name="XGBSPLIT",
    #     processor=sagemaker.processing.ScriptProcessor(
    #         image_uri=sklearn_image_url,
    #         command=["python3"],
    #         instance_type=processing_instance_type,
    #         instance_count=processing_instance_count,
    #         role=role,  # Replace with your SageMaker role ARN
    #         sagemaker_session=sagemaker_session
    #     ),
    #     inputs=[
    #         ProcessingInput(
    #             source=xgb_feature_selection_step.properties.ProcessingOutputConfig.Outputs["xgb-feature-selection-output"].S3Output.S3Uri,
    #             destination="/opt/ml/processing/input",
    #             input_name="data-splitting-input",
    #         )
    #     ],
    #     outputs=[
    #         ProcessingOutput(
    #             output_name="data-splitting-output-train",
    #             source="/opt/ml/processing/output/train",
    #             s3_upload_mode="EndOfJob",
    #             destination=xgb_splitting_data_train
    #         ),
    #         ProcessingOutput(
    #             output_name="data-splitting-output-val",
    #             source="/opt/ml/processing/output/val",
    #             s3_upload_mode="EndOfJob",
    #             destination=xgb_splitting_data_val
    #         ),
    #     ],
    #     code="./algorithms/preprocessing/data_splitting.py",  # Replace with the path to your data splitting script
    #     job_arguments=[
    #         "--input-data", "/opt/ml/processing/input/", 
    #         "--output-data", "/opt/ml/processing/output/",
    #         "--test-split-ratio", splitting_ratio.to_string(),
    #         "--drop-train-headers",
    #         "--drop-val-headers"
    #     ],
    #     cache_config=cache_config
    # )
    
    xgb_training_step = get_xgb_training_step(xgb_data_splitting_step)
    
    # xgb_training_step = TrainingStep(
    #     name="XGB_TrainingStep",
    #     estimator=xgboost_estimator,
    #     inputs={
    #         "train": sagemaker.inputs.TrainingInput(
    #             xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-train"].S3Output.S3Uri, 
    #             content_type="text/csv",
    #             distribution="ShardedByS3Key",
    #         ),
    #         "validation": sagemaker.inputs.TrainingInput(
    #             xgb_data_splitting_step.properties.ProcessingOutputConfig.Outputs["data-splitting-output-val"].S3Output.S3Uri, 
    #             content_type="text/csv",
    #         )
    #     },
    #     cache_config=CacheConfig(enable_caching=False, expire_after="1y")
    # )

    xgb_evaluation_step = get_xgb_evaluation_step(xgb_training_step, xgb_data_splitting_step)
    
    xgb_register_step = get_xgb_register_step(xgb_training_step)
    
    rcf_training_step = get_rcf_training_step(rcf_feature_selection_step)
    
    rcf_register_step = get_rcf_register_step(rcf_training_step)
    
    # Define the pipeline
    pipeline_steps = [
        dataset_creation_step,
        feature_aggregation_step,
        data_scaling_step,
        xgb_feature_selection_step,
        rcf_feature_selection_step,
        xgb_data_splitting_step
    ] if args.local else [
        dataset_creation_step,
        feature_aggregation_step,
        data_scaling_step,
        xgb_feature_selection_step,
        xgb_data_splitting_step,
        xgb_training_step,
        xgb_evaluation_step,
        xgb_register_step,
        rcf_feature_selection_step,
        rcf_training_step,
        rcf_register_step
    ]
    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[
            role,
            scalers,
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
    parser.add_argument("--pipeline-name", type=str, default="fd-pipeline-xgb", help="Name for the whole pipeline. This name would be used to specify s3 location")
    parser.add_argument("--pipeline-execution-name", type=str, default=None, help="Name for the whole pipeline. This name would be used to specify s3 location")
    parser.add_argument("--feature-group-name", type=str, default="transactions-dev", help="Name for the whole pipeline. This name would be used to specify s3 location")
    parser.add_argument("--use-cache", action="store_true", help="Cache sagemaker pipeline steps")

    args = parser.parse_args()
    main(args)