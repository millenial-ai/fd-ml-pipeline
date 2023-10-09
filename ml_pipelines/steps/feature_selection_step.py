import sagemaker
from sagemaker.workflow.steps import ProcessingStep

def get_feature_selection_step(
    parameters,
    sagemaker_session,
    step_inputs,
    step_outputs,
    feature_list,
    label,
    processing_instance_type,
    processing_instance_count,
    role,
    step_name="FeatureSelection",
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
):
    return ProcessingStep(
        name=step_name,
        processor=sagemaker.processing.ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=processing_instance_count,
            role=role, 
            sagemaker_session=sagemaker_session
        ),
        inputs=step_inputs,
        outputs=step_outputs,
        code="./algorithms/feature_selection/feature_selection_by_columns.py",
        job_arguments=[
            '--input-data', '/opt/ml/processing/input/', 
            '--output-data', '/opt/ml/processing/output/',
            '--selected-features', feature_list,
            '--selected-label', label
        ]
    )