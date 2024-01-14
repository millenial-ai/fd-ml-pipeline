zipname="scaler_deployment.zip"
zip -r $zipname *
# aws s3 rm s3://fd-ml-artifacts/$zipname
aws s3 cp $zipname s3://fd-ml-artifacts/$zipname