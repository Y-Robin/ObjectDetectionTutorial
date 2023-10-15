# Install the TensorFlow Object Detection API
#pip install tf_slim
#pip install tf-models-official

import urllib.request
import tarfile

# URL for the pre-trained ssd_mobilenet_v2_coco
MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
MODEL_NAME = 'ssd_mobilenet'

# Download and extract the model
urllib.request.urlretrieve(MODEL_URL, f"{MODEL_NAME}.tar.gz")

with tarfile.open(f"{MODEL_NAME}.tar.gz", 'r:gz') as tar:
    tar.extractall()