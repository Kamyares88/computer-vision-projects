# Vehicle Detection with YOLOv8

## Project Overview
This project implements a real-time vehicle detection system using the latest YOLOv8 model. The system can detect and classify different types of vehicles (cars, trucks, buses, etc.) in real-time video streams or static images.

## Features
- Real-time vehicle detection using YOLOv8
- Vehicle classification (car, truck, bus, motorcycle)
- Speed estimation
- Traffic analysis
- Visualization of detection results

## Dataset
The project uses the following datasets:
- COCO dataset (for general object detection)
- KITTI dataset (for vehicle-specific detection)
- BDD100K (for diverse driving scenarios)

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python detect.py --source 0  # webcam
python detect.py --source path/to/video.mp4  # video file
python detect.py --source path/to/image.jpg  # image file
```

## Project Structure
```
vehicle_detection_yolov8/
├── data/               # Dataset and configuration files
├── models/            # Model weights and architecture
├── utils/             # Utility functions
├── detect.py          # Main detection script
├── train.py           # Training script
└── requirements.txt   # Project dependencies
```

## Future Improvements
- Implement vehicle tracking
- Add license plate detection
- Integrate with traffic management systems
- Develop web interface for remote monitoring 