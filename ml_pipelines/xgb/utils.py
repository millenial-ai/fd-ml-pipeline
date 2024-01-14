import boto3
from sagemaker.session import Session
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.feature_group import FeatureGroup

def create_dataset(
    feature_group_name: str,
    output_path: str
):
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)
    
    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=region
    )
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime",region_name=region
    )
    
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )
    
    feature_store = FeatureStore(feature_store_session)
    
    transactions_fg = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    feature_store = FeatureStore(feature_store_session)
    builder = feature_store.create_dataset(
        base=transactions_fg,
        output_path=output_path,
    ).to_csv_file()