import os

if os.environ.get('CONTAINER'):
    ORIGINAL_DATA_PATH='/app/original_data'
    LABELED_DATA_PATH='/app/labeled_data'
    MODEL_PATH='/app/models'
else:
    ORIGINAL_DATA_PATH='deploy/data/original_data'
    LABELED_DATA_PATH='deploy/data/labeled_data'
    MODEL_PATH='models'