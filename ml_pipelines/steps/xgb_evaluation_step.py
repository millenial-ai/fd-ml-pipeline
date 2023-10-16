from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker import image_uris
from sagemaker.workflow.parameters import ParameterString
import sagemaker
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.properties import PropertyFile

def get_xgb_evaluation_step(
    parameters,
    sagemaker_session,
    role,
    step_inputs,
    step_outputs,
    instance_count,
    instance_type,
    cache_config=None,
    # image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    step_name="XGB_Evaluation",
):
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=sagemaker_session.boto_region_name,
        version="1.0-1",
        py_version="py3",
        instance_type=instance_type,
    )
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        base_job_name="script-abalone-eval",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    eval_args = script_eval.run(
        inputs=step_inputs,
        outputs=step_outputs,
        code="./algorithms/evaluation/xgb_model_evaluation.py",
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation-result", path="evaluation.json"
    )
    
    return ProcessingStep(
        name=step_name,
        step_args=eval_args,
        property_files=[evaluation_report],
        cache_config=cache_config
    )