from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolo11x.pt")  # Using YOLOv8 Nano for speed

# Define animal classes from the COCO dataset
animal_classes = {"cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# Read input image
image_path = "images/animals.png"  # Replace with your image path
image = cv2.imread(image_path)

# Run YOLO on the image
results = model(image)

# Process results
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence score
        label = model.names[cls_id]  # Get class label

        if label in animal_classes and conf > 0.5:  # Filter only animals with confidence > 50%
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Annotate label & confidence score
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the processed image
cv2.imshow("Animal Detection", image)

# Save the output image
output_path = "output/processed_image.jpg"
cv2.imwrite(output_path, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
