### Files Used to Create Sagemaker Instance and API

#### Container
Used to create a Docker container which was then uploaded to AWS and used to create a Sagemaker endpoint. The only things missing here are the model artifacts created from model development. See the "NOTES" below on how to get the model artifacts.

#### Local Test
After the container is created this builds a local copy of the container and allows for local testing of the model.

#### Autoscale API Endpoint
Contains two of the files needed to create an API endpoint using chalice. In order to create a new chalice project, make sure chalice and boto3 are installed on your environment (can be done using pip). After, type "chalice new-project" and it will set up a new chalice directory where you can name your project. The app.py and the config.json file are the edited versions for this project. You will need to change these files to match up with your environment. Once those files are created/configured, the AWS Lambda and the REST API are automatically created after typing "chalice deploy".


### NOTES
#### Model Artifacts
In order to deploy the model/container in AWS or deploy the model/container locally, you will need the model artifacts. These can be downloaded from our S3 bucket using the AWS CLI. Once the AWS CLI is installed and set up for your account (aws configure), run the following command:

```bash
aws s3 cp s3://openalex-institution-tagger-model-artifacts/ . --recursive
```

This will download multiple files, one of them being a institution_tagger_artifacts.tgz file. The .tgz file will need to be unzipped in the appropriate location in either the local test folder (local_test/test_dir/model/) or the container folder (container/model_files/).
