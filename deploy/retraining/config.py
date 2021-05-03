import os

if os.environ.get('CONTAINER'):
    ORIGINAL_DATA_PATH='/app/original_data'
    LABELED_DATA_PATH='/app/labeled_data'
    MODEL_PATH='/app/models'
else:
    ORIGINAL_DATA_PATH='/home/pi/doorbell_detector/deploy/data/original_data'
    LABELED_DATA_PATH='/home/pi/doorbell_detector/deploy/data/labeled_data'
    MODEL_PATH='/home/pi/doorbell_detector/deploy/models'