import cv2
import os
import glob

# Define target image size
target_size = (320, 320)

# Define folders and corresponding keys for each gesture
gestures = {
    'thumbs_up': 'u',
    'thumbs_down': 'd',
    'fist': 'f',
    'index': 'i'
}

# Create folders to save the images if they don't exist and initialize counters
img_counters = {}
for gesture in gestures.keys():
    folder_path = f'samples/{gesture}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Find existing images and set counter to the next available index
    existing_files = glob.glob(f"{folder_path}/{gesture}_*.png")
    if existing_files:
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        img_counters[gesture] = max(existing_indices) + 1
    else:
        img_counters[gesture] = 0

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame
    resized_frame = cv2.resize(frame, target_size)

    # Display the resulting frame
    cv2.imshow('Gesture Capture', frame)

    # Check for user input
    key = cv2.waitKey(1)

    # Loop through gestures to see if any corresponding key was pressed
    for gesture, gesture_key in gestures.items():
        if key == ord(gesture_key):
            img_name = f"samples/{gesture}/{gesture}_{img_counters[gesture]}.png"
            cv2.imwrite(img_name, resized_frame)
            print(f"{img_name} saved!")
            img_counters[gesture] += 1

    # If 'q' is pressed, quit the application
    if key == ord('q'):
        break

# When everything is done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()