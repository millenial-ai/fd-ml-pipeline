import sagemaker
from sagemaker.workflow.steps import ProcessingStep

def get_data_scaling_step(
    parameters,
    sagemaker_session,
    step_inputs,
    step_outputs,
    instance_type,
    instance_count,
    execution_role,
    scalers,
    cache_config=None,
    step_name="DataStandardScaling",
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
):
    return ProcessingStep(
        name=step_name,
        # step_args=data_scaling_step_args
        processor=sagemaker.processing.ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type,
            instance_count=instance_count,
            role=execution_role,
            sagemaker_session=sagemaker_session
        ),
        inputs=step_inputs,
        outputs=step_outputs,
        code="./algorithms/preprocessing/data_scaling.py",  # Replace with the path to your data preprocessing script
        job_arguments=[
            '--input-data', '/opt/ml/processing/input/', 
            '--output-data', '/opt/ml/processing/output/',
            '--artifact-data', '/opt/ml/processing/artifact/',
            '--scalers', scalers
        ],
        cache_config=cache_config
    )