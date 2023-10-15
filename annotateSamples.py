import cv2
import os
import glob
import json

# Initialize variables for drawing rectangles
drawing = False
top_left_pt, bottom_right_pt = None, None

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow(f'Labeling {gesture}', img)

# Load existing annotations if available
try:
    with open('annotations.json', 'r') as f:
        annotations = json.load(f)
except FileNotFoundError:
    annotations = {}

# Define folders for each gesture
gestures = ['thumbs_up', 'thumbs_down', 'fist', 'index']

# Loop through each gesture folder
for gesture in gestures:
    folder_path = f'samples/{gesture}'
    
    # Get list of image files
    image_files = glob.glob(f"{folder_path}/{gesture}_*.png")

    if gesture not in annotations:
        annotations[gesture] = {}

    for image_file in image_files:
        image_name = os.path.basename(image_file)

        # Skip if this image is already annotated
        if image_name in annotations[gesture]:
            continue

        # Read and display the image
        img = cv2.imread(image_file)
        cv2.namedWindow(f'Labeling {gesture}')
        cv2.setMouseCallback(f'Labeling {gesture}', draw_rectangle)

        while True:
            cv2.imshow(f'Labeling {gesture}', img)
            key = cv2.waitKey(1)

            # If 's' is pressed, save the coordinates of the bounding box
            if key == ord('s'):
                if top_left_pt and bottom_right_pt:
                    annotations[gesture][image_name] = {
                        'top_left': top_left_pt,
                        'bottom_right': bottom_right_pt
                    }
                break
            
            # If 'd' is pressed, delete the last annotation
            elif key == ord('d'):
                if image_name in annotations[gesture]:
                    del annotations[gesture][image_name]
                img = cv2.imread(image_file)  # Reload the original image
                cv2.imshow(f'Labeling {gesture}', img)
            
            # If 'q' is pressed, quit the application
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

# Save annotations to a JSON file
with open('annotations.json', 'w') as f:
    json.dump(annotations, f)

print("Annotations saved.")