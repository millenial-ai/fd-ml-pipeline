import sagemaker
import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker import image_uris
import sagemaker
from sagemaker.workflow.steps import ProcessingStep

def get_rcf_training_step(
    parameters, 
    sagemaker_session,
    role,
    s3_data,
    train_instance_count,
    train_instance_type,
    step_name="RCF_TrainModel"
):
    container = image_uris.retrieve(region=sagemaker_session.boto_region_name, 
                                framework="randomcutforest", 
                                version="1")
    rcf = sagemaker.estimator.Estimator(
        container,
        role,
        output_path=parameters.get("train_artifact_path"),
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        sagemaker_session=sagemaker_session,
    )
    
    rcf.set_hyperparameters(num_samples_per_tree=200, num_trees=50, feature_dim=6)

    train_step_args = rcf.fit(
        inputs={
            "train": TrainingInput(
                # s3_data = Output of the previous call back step
                s3_data=s3_data,
                content_type="text/csv;label_size=0",
                distribution="ShardedByS3Key",
            )
        },
    )
    return TrainingStep(
        name=step_name,
        step_args=train_step_args
    )