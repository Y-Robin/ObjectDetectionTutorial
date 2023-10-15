import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model_dir = "C:/Users/ObjectDetectionTutorial/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/saved_model"  # Replace with your path
loaded_model = tf.saved_model.load(model_dir)
infer = loaded_model.signatures["serving_default"]
target_size = (224, 224)
labelNames = ["Thumbsup","thumbsdown","fist","index"]
# Function to run inference on a frame
def run_inference(frame, infer):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # Normalize image pixel values
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = infer(input_tensor)
    return detections

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, target_size)
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    output_dict = run_inference(frame_rgb, infer)
    
    # Extract detection boxes, classes, and scores
    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy().astype(int)
    
    # Draw bounding boxes on the frame
    height, width, _ = frame.shape
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # Only consider detections with confidence > 0.5
            box = boxes[i]
            class_id = classes[i]
            score = scores[i]
            
            # Convert to pixel coordinates
            height, width, _ = frame.shape
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Display class label and score
            label = f"Class: {labelNames[class_id-1]}, Score: {score:.2f}"
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Gesture Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()