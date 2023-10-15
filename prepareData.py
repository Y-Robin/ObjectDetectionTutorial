import tensorflow as tf
import json
import cv2
import os

def create_tf_example(image_path, annotations_list, gesture):
    # Read the image using OpenCV and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Get image dimensions
    height, width, _ = image.shape

    # Prepare the data for the TFRecord fields
    filename = image_path.encode('utf8')
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = b'JPEG'

    # Class mapping for our 4 gestures
    class_mapping = {'thumbs_up': 1, 'thumbs_down': 2, 'fist': 3, 'index': 4}

    # Lists to store multiple annotations
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    class_texts = [gesture.encode('utf8')] * len(annotations_list)
    class_labels = [class_mapping[gesture]] * len(annotations_list)

    for annotations in annotations_list:
        # Normalize bounding box coordinates and append to list
        x_min = annotations['top_left'][0] / width
        y_min = annotations['top_left'][1] / height
        x_max = annotations['bottom_right'][0] / width
        y_max = annotations['bottom_right'][1] / height

        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)

    # Create a TFRecord example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=x_mins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=x_maxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=y_mins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=y_maxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_texts)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_labels)),
    }))
    return tf_example

# Load annotations from JSON file
with open('annotations.json', 'r') as f:
    annotations_data = json.load(f)

# Create a TFRecord writer
writer = tf.io.TFRecordWriter('gestures.tfrecord')

# Loop through all gestures
for gesture in os.listdir('samples'):
    # Path to the folder containing images for this gesture
    gesture_folder = os.path.join('samples', gesture)
    
    # Skip if not a directory
    if not os.path.isdir(gesture_folder):
        continue

    # Loop through all image files in this gesture folder
    for image_file in os.listdir(gesture_folder):
        # Skip if not an image file
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Build the full path to the image file
        image_path = os.path.join(gesture_folder, image_file)

        # Annotations for this image
        coords_list = annotations_data.get(gesture, {}).get(image_file, None)

        # Skip if no annotations found
        if coords_list is None:
            continue

        # Wrap in a list to align with create_tf_example function
        coords_list = [coords_list]


        # Add class name to each annotation
        for annotation in coords_list:
            annotation['class'] = gesture
            
        print(coords_list)
        # Create a TFRecord example and write it to the TFRecord file
        tf_example = create_tf_example(image_path, coords_list, gesture)
        writer.write(tf_example.SerializeToString())


# Close the TFRecord writer
writer.close()