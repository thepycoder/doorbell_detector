# doorbell_detector
An over engineered doorbell detector based on machine learning


This repository is structured as follows:

raw_data: folder to keep track of original, uncut recordings for reference. It's always a good idea to keep the original data around.
split_data: folder to put the manually extracted bell sounds in and the automatically extracted non-bell and non-silence parts.
    bell: bell sounds of varying lengths
    noise: ambient noise sounds of varying lengths


Quickstart on your own data:
    Extract the bell sounds and put each audio file under `split_data/bell`
    Throw the ambient sounds file in `raw_data` as `ambient.wav`, it will be split as part of the notebook

    Run the notebook to generate a model file

    Add a `creds.py` file under `deploy` and add an `app_token` and `client_token` variable of your pushover api keys
    Copy the deploy folder (also containing the trained model file now) to the raspberry pi

    You can use `screen` to run the python script until it is deployed with docker in the next version



TODO: add retraining code for corrected windows