export RCF_ENDPOINT=fd-endpoint-rcf
export XGB_ENDPOINT=fd-endpoint-xgb
# export RCF_ENDPOINT=custom-nopca-datacamp-rcf
# export XGB_ENDPOINT=custom-nopca-datacamp-xgbsmote
export SQS_NAME=FraudRequestQueue-Staging
aws configure set region $(aws configure list | grep region | awk '{print $2}')

cd ~/fd-worker

source ~/.bashrc
echo "Starting app ($env)"

nohup bash "run-${env:-staging}.sh" > /dev/null 2>&1 &
