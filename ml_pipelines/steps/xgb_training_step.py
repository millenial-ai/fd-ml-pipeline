# import sagemaker
# import boto3
# from sagemaker.workflow.pipeline_context import PipelineSession
# from sagemaker.inputs import TrainingInput
# from sagemaker.workflow.steps import TrainingStep
# from sagemaker import image_uris
# from sagemaker.debugger import Rule, rule_configs

# def get_xgb_training_step(
#     parameters, 
#     sagemaker_session,
#     role,
#     s3_data,
#     train_instance_count,
#     train_instance_type,
#     cache_config=None,
#     step_name="RCF_TrainModel"
# ):
#     container = image_uris.retrieve(region=sagemaker_session.boto_region_name, 
#                                 framework="randomcutforest", 
#                                 version="1")
#     xgb = sagemaker.estimator.Estimator(
#         container,
#         role,
#         output_path=parameters.get("train_artifact_path"),
#         instance_count=train_instance_count,
#         instance_type=train_instance_type,
#         sagemaker_session=sagemaker_session,
#         rules=[
#             Rule.sagemaker(rule_configs.create_xgboost_report())
#         ]
#     )
    
#     # xgb.set_hyperparameters(num_samples_per_tree=200, num_trees=50, feature_dim=6)
#     scale_pos_weight = .01 # set to fixed for now. Can change to output from previous step
#     xgb.set_hyperparameters(
#         max_depth=5,
#         subsample=0.8,
#         num_round=100,
#         eta=0.9,
#         gamma=10,
#         min_child_weight=16,
#         silent=0,
#         objective="binary:logistic",
#         eval_metric="auc",
#         scale_pos_weight=scale_pos_weight
#     )

#     train_step_args = xgb.fit(
#         inputs={
#             "train": TrainingInput(
#                 # s3_data = Output of the previous call back step
#                 s3_data=s3_data,
#                 content_type="text/csv",
#                 distribution="ShardedByS3Key",
#             )
#         },
#     )
#     return TrainingStep(
#         name=step_name,
#         step_args=train_step_args,
#         cache_config=cache_config
#     )