# **Animal Detection & Tracking with YOLOv11**

## **Overview**
This project utilizes **YOLOv11** for real-time animal detection and tracking using a webcam feed. The system detects predefined animal classes and tracks their movement, counting crossings over a defined vertical line.

## **Installation**
### **0. Create Virtual Environment (optional)**
Create a virtual environment:
```sh
python -m venv myenv
```
Activate it
On Windows:
```sh
myenv\Scripts\activate
```
On Mac/Linux:
```sh
source myenv/bin/activate
```
### **1. Install Dependencies**
Ensure you have Python installed, then install the required dependencies:
```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### **2. Check CUDA Availability**
For optimal performance, ensure that CUDA is available on your system. See [CUDA Installation Guide](CUDA.md) for setup instructions.

To check CUDA availability, run:
```sh
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is `True`, CUDA is available.

## **Running the Program**
To start the detection and tracking program for live camera, run:
```sh
python live_tracking.py
```

To start the detection and tracking program for live camera, run:
```sh
python file_tracking.py
```

## **Features**
- **Real-time object detection** using YOLOv11
- **Animal tracking** with unique ID assignment
- **Crossing count detection** for monitoring movement
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
- Ensure your camera is accessible before running the script.
- The tracking algorithm may be adjusted for different use cases.