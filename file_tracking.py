import cv2
import torch.cuda
from ultralytics import YOLO
from collections import deque
import numpy as np
import logging

#Removes Yolo logs from the terminal - logs only errores(optional)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print("Press q - quit, p - pause frame, n - go to next frame(when paussed), b - go to previous frame(when paussed)")

# Load YOLOv8 model and move to GPU (CUDA)
model = YOLO("yolo11x.pt")  # Ensure you have the correct model file

if torch.cuda.is_available():
    print(f"CUDA Available")
    model.to("cuda")  # Move model to GPU (if available)

# Define animal classes for detection
animal_classes = ["cat", "dog", "horse", "cow", "sheep", "elephant", "bear", "zebra"]

# Open video file (can be any format OpenCV supports .mp4 .avi .mov .mkv .wmv .flv .webm) 
video_path = "videos\single_horse.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the vertical line for counting crossings
line_x = 300
crossing_count = 0

# Object tracking storage
object_tracks = {}  # Dictionary storing object positions and trails
object_trackers = {}  # Dictionary storing individual object trackers
track_length = 30  # Maximum tracking trail length
inactive_frames_threshold = 10  # Threshold for removing inactive objects

# Function to calculate Euclidean distance
def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


# Set up the video writer to save the processed video
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'output/detection_{timestamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for .mp4 files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Initialize frame navigation state
paused = False
current_frame = 0

# Process video frames
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video file readched.")
            break  # Exit loop if video ends or error occurs
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Update current frame

    # Perform object detection using YOLO
    results = model(frame, iou=0.5)  # Adjust IoU threshold to reduce duplicate detections

    # Get detected objects
    boxes = results[0].boxes
    names = results[0].names

    new_tracks = {}  # Stores updated object positions
    inactive_tracks = {}  # Tracks inactive objects for removal

    for box in boxes:
        # Extract bounding box details
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = names[class_id]

        # Process only selected animal classes with high confidence
        if label in animal_classes and confidence > 0.4:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Assign an ID based on proximity to existing tracked objects
            obj_id = None
            min_distance = float('inf')

            for existing_id, trail in object_tracks.items():
                last_position = trail[-1]  # Last known position
                distance = calculate_distance((center_x, center_y), last_position)

                if distance < 75:  # Increased threshold to prevent multiple detections of the same object
                    obj_id = existing_id  # Assign same ID
                    min_distance = distance
                    break

            # Create a new ID if no close object is found
            if obj_id is None:
                obj_id = f"{label}_{int(x1)}_{int(y1)}"
                object_trackers[obj_id] = cv2.TrackerCSRT_create()
                object_trackers[obj_id].init(frame, bbox)

            # Update tracking history
            if obj_id not in object_tracks:
                object_tracks[obj_id] = deque(maxlen=track_length)
            object_tracks[obj_id].append((center_x, center_y))
            new_tracks[obj_id] = (center_x, center_y)

    # Remove inactive objects that haven't been detected for too long
    for obj_id in list(object_tracks.keys()):
        if obj_id not in new_tracks:
            if obj_id not in inactive_tracks:
                inactive_tracks[obj_id] = 0
            inactive_tracks[obj_id] += 1

            if inactive_tracks[obj_id] >= inactive_frames_threshold:
                print(f"Object {obj_id} removed due to inactivity.")
                del object_tracks[obj_id]
                del object_trackers[obj_id]
                del inactive_tracks[obj_id]

    # Draw tracking trails
    for obj_id, trail in object_tracks.items():
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

    # Draw counting line
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    # Check if objects crossed the vertical line
    for obj_id, trail in object_tracks.items():
        prev_x, prev_y = trail[-2] if len(trail) > 1 else trail[-1]
        center_x, center_y = trail[-1]

        if prev_x < line_x <= center_x:  # Left to right
            crossing_count += 1
            print(f"{obj_id} crossed LEFT to RIGHT. Total: {crossing_count}")
        elif prev_x > line_x >= center_x:  # Right to left
            crossing_count -= 1
            print(f"{obj_id} crossed RIGHT to LEFT. Total: {crossing_count}")

    # Display crossing count on video
    cv2.putText(frame, f"Crossings: {crossing_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to output video file
    out.write(frame)

    # Show video with detections (Optional)
    cv2.imshow("Animal Detection & Tracking", frame)

    # Handle key events for pause, resume, and frame navigation
    key = cv2.waitKey(0 if paused else 1) & 0xFF
    if key == ord('q'):  # Exit if 'q' is pressed
        print("Exiting")
        break
    elif key == ord('p'):  # Toggle pause
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('n') and paused:  # Go to next frame when paused
        if current_frame < total_frames - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
            ret, frame = cap.read()
            current_frame += 1
        print("Next Frame:", current_frame)
    elif key == ord('b') and paused:  # Go to previous frame when paused
        if current_frame > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
            ret, frame = cap.read()
            current_frame -= 1
        print("Previous Frame:", current_frame)
# Cleanup
cap.release()
cv2.destroyAllWindows()