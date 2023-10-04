from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker import image_uris
from sagemaker.workflow.parameters import ParameterString

def get_rcf_register_step(
    parameters, 
    sagemaker_session,
    model_data,
    step_name="RCF_ModelRegistration",
):
    model = Model(
        image_uri=image_uris.retrieve(region=sagemaker_session.boto_region_name, 
                                framework="randomcutforest", 
                                version="1"),
        model_data=model_data,
        sagemaker_session=sagemaker_session,
        role=parameters.get("execution_role"),
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
    
    return ModelStep(
        name=step_name,
        step_args=register_args
    )