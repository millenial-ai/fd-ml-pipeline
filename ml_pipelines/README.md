 After you develop the source code and tests of each step, the next step is to define the SageMaker pipelines in another root folder. 
 
 Each ML pipeline definition is placed in subfolder that contains the .py file and a JSON or .yaml file for input parameters, such as hyperparameter ranges. 
 
 A readme file to describe the ML pipelines is necessary.
 
 python train_rcf.py --input-path s3://fd-ml-pipeline/preprocessed_data/splitting/train --artifact-path s3://fd-ml-pipeline/training/artifact --role arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457 --instance-type ml.m5.large --local-metadata-path /tmp/artifact/