import sagemaker
from sagemaker.workflow.steps import ProcessingStep

def get_rcf_data_splitting_step(
    parameters,
    sagemaker_session,
    step_inputs,
    step_outputs,
    step_name="RCF_DataSplitting",
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
):
    data_splitting_step = ProcessingStep(
        name=step_name,
        processor=sagemaker.processing.ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=parameters.get("processing_instance_type"),
            instance_count=parameters.get("processing_instance_count"),
            role=parameters.get("execution_role"),  # Replace with your SageMaker role ARN
            sagemaker_session=sagemaker_session
        ),
        inputs=step_inputs,
        outputs=step_outputs,
        code="./algorithms/preprocessing/data_splitting.py",  # Replace with the path to your data splitting script
        job_arguments=[
            "--input-data", "/opt/ml/processing/input/", 
            "--output-data", "/opt/ml/processing/output/",
            "--test-split-ratio", str(parameters.get("splitting_ratio")),
            "--drop-label", parameters.get("label")
        ]
    )
    return data_splitting_step