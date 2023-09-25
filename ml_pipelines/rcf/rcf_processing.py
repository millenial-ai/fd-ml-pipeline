import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Define your SageMaker execution role
# role = get_execution_role()
role = "arn:aws:iam::348490654799:role/service-role/AmazonSageMaker-ExecutionRole-20230705T105457"
# Initialize SageMaker session and S3 bucket
sagemaker_session = sagemaker.Session()
bucket_name = 'fd-ml-pipeline'
prefix = 'sagemaker-pipeline-rcf'

# Define data processing script
data_preprocessing_script = 'data_preprocessing.py'

# Step 1: Data Processing
processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type='ml.t3.medium',
    instance_count=1,
)

# Define processing inputs and outputs
input_data = 's3://fd-ml-pipeline/input_data/datacamp-creditcardfraud-100.csv'
output_data = 's3://{}/{}/processed'.format(bucket_name, prefix)

processing_input = ProcessingInput(
    source=input_data,
    destination='/opt/ml/processing/input',
    s3_data_distribution_type='ShardedByS3Key'
)

processing_output = ProcessingOutput(
    output_name='processed_data',
    source='/opt/ml/processing/output',
    s3_upload_mode='EndOfJob'
)

# Create and run the processing job
processor.run(
    code=data_preprocessing_script,
    inputs=[processing_input],
    outputs=[processing_output],
    arguments=[
        '--input-data', '/opt/ml/processing/input', 
        '--output-data', '/opt/ml/processing/output/output.csv'
    ]
)

# Wait for the processing job to complete
processor_job_description = processor.jobs[-1].describe()
job_status = processor_job_description['ProcessingJobStatus']
if job_status == 'Completed':
    print("Data preprocessing job completed successfully.")
else:
    print("Data preprocessing job failed.")
