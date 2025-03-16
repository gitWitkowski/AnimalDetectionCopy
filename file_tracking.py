# **Animal Detection & Tracking with YOLOv8**

## **Overview**
This project utilizes **YOLOv8** for real-time animal detection and tracking. The system detects predefined animal classes and tracks their movement, counting crossings over a defined vertical line.

## **Installation**
### **1. Install Dependencies**
Ensure you have Python installed, then install the required dependencies:
```sh
pip install -r requirements.txt
```

### **2. Check CUDA Availability**
For optimal performance, ensure that CUDA is available on your system. See [CUDA Installation Guide](CUDA.md) for setup instructions.

To check CUDA availability, run:
```sh
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is `True`, CUDA is available.

## **Running the Program**
There are two versions of the program available:

### **1. Live Webcam Detection**
To run the program with a live camera feed:
```sh
python live_tracking.py
```
Ensure you have a connected webcam before running this version.

### **2. Video File Detection**
To process an existing video file:
```sh
python file_tracking.py
```
Modify `video_path` in `file_tracking.py` to specify the input video file.

## **Features**
- **Real-time object detection** using YOLOv8
- **Animal tracking** with unique ID assignment
- **Crossing count detection** for monitoring movement
- **Supports both live webcam and video file processing**
- **CUDA acceleration** for improved performance (if supported)

## **Supported Animal Classes**
The system detects and tracks the following animals:
- Cat
- Dog
- Horse
- Cow
- Sheep
- Elephant
- Bear
- Zebra

## **Additional Notes**
- The `yolo11x.pt` model file must be placed in the working directory.
- Ensure your camera is accessible before running the live tracking script.
- The tracking algorithm may be adjusted for different use cases.