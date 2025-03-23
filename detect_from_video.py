from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolo11x.pt")  # Using YOLOv8 Nano for speed

# Define animal classes from the COCO dataset
animal_classes = {"cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# Open video file
video_path = "videos/single_horse.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter("output/output_video.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 on the frame
    results = model(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            label = model.names[cls_id]  # Get class label

            if label in animal_classes and conf > 0.5:  # Filter only animals with confidence > 50%
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Annotate label & confidence score
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show processed frame
    cv2.imshow("Animal Detection", frame)

    # Save processed frame to output video
    output.write(frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
