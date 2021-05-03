import os

if os.environ.get('CONTAINER'):
    ORIGINAL_DATA_PATH='/app/data/original'
    LABELED_DATA_PATH='/app/data/labeled_data'
    MODEL_PATH='/app/models'
else:
    ORIGINAL_DATA_PATH='deploy/data/original'
    LABELED_DATA_PATH='deploy/data/labeled_data'
    MODEL_PATH='deploy/models'