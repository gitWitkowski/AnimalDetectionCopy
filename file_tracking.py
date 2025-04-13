import cv2
import torch.cuda
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import numpy as np
import logging
import PySimpleGUI as sg

#Removes Yolo logs from the terminal - logs only errores(optional)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load YOLOv8 model and move to GPU (CUDA)
model = YOLO("yolo11x.pt") # Ensure you have the correct model file

if torch.cuda.is_available():
    model.to("cuda") # Move model to GPU (if available)

# Define animal classes for detection
animal_classes = ["cat", "dog", "horse", "cow", "sheep", "elephant", "bear", "zebra"]

# Open video file (can be any format OpenCV supports .mp4 .avi .mov .mkv .wmv .flv .webm) 
video_path = "videos/single_horse.mp4"# Replace with the path to your video file animals3_compressed_cropped_30
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    sg.popup_error(f"Error: Could not open video file: {video_path}.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Object tracking storage
object_tracks = {}  # Dictionary storing object positions and trails
object_trackers = {}  # Dictionary storing individual object trackers
track_length = 30  # Maximum tracking trail length
inactive_frames_threshold = 10  # Threshold for removing inactive objects
line_x = 300
crossing_count = 0

# Initialize frame navigation state
paused = False
current_frame = 0
frame_changed = False

# Set up the video writer to save the processed video
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'output/detection_{timestamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for .mp4 files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Define GUI layout
layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Text('Crossings: 0', key='-CROSSINGS-')],
    [sg.Text('Line Position:'), sg.Slider(range=(0, frame_width), default_value=line_x,
                                          orientation='h', size=(40, 15), key='-LINE_SLIDER-', enable_events=True)],
    [sg.Button('Play/Pause'), sg.Button('Next Frame'), sg.Button('Previous Frame'), sg.Button('Exit')]
]


# Create the window
window = sg.Window('Animal Detection & Tracking', layout, location=(100, 100))

# Function to calculate Euclidean distance
def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

# Main event loop - Process video frames
while True:
    event, values = window.read(timeout=0 if paused else 10)
    if event in (sg.WIN_CLOSED, 'Exit'):
        print("Exiting")
        break
    elif event == 'Play/Pause':
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif event == 'Next Frame' and paused:
        if current_frame < total_frames - 1:
            current_frame += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            paused_frame = frame.copy()
            print("Next Frame:", current_frame)
            frame_changed = True
        else:
            print("You are watching last frame of the  video - you can't go to next frame")
    elif event == 'Previous Frame' and paused:
        if current_frame > 1:
            current_frame -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            paused_frame = frame.copy()
            print("Previous Frame:", current_frame)
            frame_changed = True
        else:
            print("You are watching first frame of the  video - you can't go to previous frame")    
    elif event == '-LINE_SLIDER-':
        line_x = int(values['-LINE_SLIDER-'])
        print(f"new position of the line: {line_x}")

    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

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

    # Preserve drawn lines
    if paused and   event == '-LINE_SLIDER-':
        frame = paused_frame.copy()

    # Draw tracking trails
    for obj_id, trail in object_tracks.items():
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

    if not paused:
        paused_frame = frame.copy()

    # Draw counting line
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    # Check if objects crossed the vertical line
    if  event != '-LINE_SLIDER-':
        if not paused or  (paused and frame_changed): 
            for obj_id, trail in object_tracks.items():
                prev_x, prev_y = trail[-2] if len(trail) > 1 else trail[-1]
                center_x, center_y = trail[-1]

                if prev_x < line_x <= center_x:  # Left to right
                    crossing_count += 1
                    print(f"{obj_id} crossed LEFT to RIGHT. Total: {crossing_count}")
                elif prev_x > line_x >= center_x:  # Right to left
                    crossing_count -= 1
                    print(f"{obj_id} crossed RIGHT to LEFT. Total: {crossing_count}")

    # Display crossing count - optional
    # cv2.putText(frame, f"Crossings: {crossing_count}", (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to output video - optional
    out.write(frame)

    # Convert frame to RGB and update GUI image
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)
    window['-CROSSINGS-'].update(f'Crossings: {crossing_count}')

    frame_changed = False

# Cleanup
cap.release()
out.release()
window.close()