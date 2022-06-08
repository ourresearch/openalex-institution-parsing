### Most files needed for deployment to AWS
* "model_to_api" folder contains all of the files for either deploying a container locally or setting up an API through AWS
* "mag_functions.py" is a python module with code to call the API
* use "testing_API_institution_tagger.ipynb" notebook to actually call the API, however this code needs the mag_functions.py as well as a json file with the API key and URL
