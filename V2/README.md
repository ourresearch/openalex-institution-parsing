### V2 of the OpenAlex Institution Tagger Model

The V2 model improves over the V1 model in both performance and functionality. We are now able to accurately predict on strings that have multiple institutions and return multiple institution predictions. There was also a decision made to only predict on institutions that have a ROR ID, as opposed to the V1 model which tried to predict on all institutions we inherited from MAG and other data sources. Lastly, we are releasing more "gold" test data that can be used to quickly test model performance. Please see below for the data that is available and look in the notebooks to see the code for using those datasets. 

This page serves as a guide for how to use this part of the repository. You should only have to make minor changes to the code (adding API key files, changing file paths, updating configuration files, etc.) in order to make this code functional. The python package requirements file can be used for all python notebooks in this directory.

#### 001 Exploration

Use the notebook in this directory to look into the way the institution data in OpenAlex as well as the ROR data were combined to create artificial data. A notebook was also added that creates some of the model artifact files.

#### 002 Model

Use the notebooks in this directory if you would like to train a model from scratch using the same methods as OpenAlex. These notebooks using both Spark (Databricks) and Jupyter notebooks so make sure you are in the right environment. The notebooks progress sequentially so start at notebook 001 and go in order from there. The last notebook was used to test the model.

#### 003 Deploy

Use the notebooks in this directory if you would like to deploy the model locally or in AWS. The model artifacts will need to be downloaded into the appropriate folder before the model can be deployed, so make sure to follow the instructions in the "NOTES" section below.


### NOTES
#### Model Artifacts
In order to deploy the model/container in AWS or deploy the model/container locally, you will need the model artifacts. These can be downloaded from our S3 bucket using the AWS CLI. Once the AWS CLI is installed and set up for your account (aws configure), run the following command:

```bash
aws s3 cp s3://openalex-institution-tagger-model-artifacts/ . --recursive
```

This will download multiple files (including the V1 model artifacts), one of them being a institution_tagger_v2_artifacts.tar.gz file. The .tgz file will need to be unzipped in the appropriate location in either the local test folder (local_test/test_dir/model/) or the container folder (container/model_files/).

#### Test Data
In order to effectively test the model, multiple different datasets were created. All of these datasets will be contained within the files that can be downloaded from our institution parser model artifacts bucket (institution_gold_datasets_v2.tgz) so please see the section above for how to download them. The following datasets will be found within the .tgz file:

* Gold Data: Mixture of 1000 "easy" samples (labeled gold_1000 in dataset) and 500 "hard" samples (labeled gold_500 in dataset)
* CWTS Multi-String Institution No Relation: Mostly multi-string institutions where institutions are not related. Data was curated by CWTS.
* CWTS Multi-String Institution Related: Mostly multi-string institutions where institutions are related (think university and university hospital). Data was curated by CWTS.
* Sampled 200: Randomly sampled strings. Gives true idea of our model performance.
* Multi-String OpenAlex: Small multi-institution string dataset created to help with initial model testing.
