### V1 of the OpenAlex Institution Tagger Model

This page serves as a guide for how to use this part of the repository. You should only have to make minor changes to the code (adding API key files, changing file paths, updating configuration files, etc.) in order to make this code functional. The python package requirements file can be used for all python notebooks in this directory.

#### 001 Exploration

Use the notebook in this directory to look into the queries used to explore both the institution data in OpenAlex as well as the ROR data that was used to augment the training data.

#### 002 Model

Use the notebooks in this directory if you would like to train a model from scratch using the same methods as OpenAlex. These notebooks using both Spark (Databricks) and Jupyter notebooks so make sure you are in the right environment. The notebooks progress sequentially so start at notebook 001 and go in order from there.

#### 003 Deploy

Use the notebooks in this directory if you would like to deploy the model locally or in AWS. The model artifacts will need to be downloaded into the appropriate folder before the model can be deployed, so make sure to follow the instructions in the "NOTES" section below.


### NOTES
#### Model Artifacts
In order to deploy the model/container in AWS or deploy the model/container locally, you will need the model artifacts. These can be downloaded from our S3 bucket using the AWS CLI. Once the AWS CLI is installed and set up for your account (aws configure), run the following command:

```bash
aws s3 cp s3://openalex-institution-tagger-model-artifacts/ . --recursive
```

This will download multiple files, one of them being a institution_tagger_artifacts.tgz file. The .tgz file will need to be unzipped in the appropriate location in either the local test folder (local_test/test_dir/model/) or the container folder (container/model_files/).

#### Gold Test Data
In order to effectively test the models, two separate "gold" datasets were created. Both of these datasets will be located in the same S3 bucket as the model artifacts (institution_gold_datasets.tgz) so please see the section above for how to download them. The gold 1000 dataset took a random sample of 1000 affiliation strings that had an institution tagged in MAG or the original OpenAlex process. Each affiliation string and institution were checked to make sure there was a match so that true performance could be obtained for the old tagging process as well as the new model. The gold 500 dataset took a random sample of 500 affiliation strings that did not have an institution tagged in MAG or the original OpenAlex process. These affiliation strings are harder to parse and so the performance on this dataset is significantly lower than that of the gold 1000.
