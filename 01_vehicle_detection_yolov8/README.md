# Vehicle Detection with YOLOv8

## Project Overview
This project implements a real-time vehicle detection system using the latest YOLOv8 model. The system can detect and classify different types of vehicles (cars, trucks, buses) in real-time video streams or static images.

## Features
- Real-time vehicle detection using YOLOv8
- Vehicle classification (car, truck, bus)
- Speed estimation
- Traffic analysis
- Visualization of detection results
- Support for webcam, video files, and images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kamyares88/computer-vision-projects.git
cd computer-vision-projects/01_vehicle_detection_yolov8
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-time Detection
For webcam:
```bash
python detect.py --source 0
```

For video file:
```bash
python detect.py --source path/to/video.mp4
```

For image file:
```bash
python detect.py --source path/to/image.jpg
```

### Training
To train the model on your own dataset:
1. Prepare your dataset in YOLO format
2. Update the dataset configuration in `data/vehicles.yaml`
3. Run training:
```bash
python train.py --data data/vehicles.yaml --epochs 100 --batch 16
```

### Command Line Arguments
- `--source`: Input source (0 for webcam, path for video/image)
- `--model`: Model to use for detection (default: yolov8n.pt)
- `--conf`: Confidence threshold (default: 0.5)
- `--save`: Save output video (default: False)

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

## Dataset
The project uses the following datasets:
- COCO dataset (for general object detection)
- KITTI dataset (for vehicle-specific detection)
- BDD100K (for diverse driving scenarios)

## Performance
- Real-time processing on modern GPUs
- High accuracy for vehicle detection
- Support for multiple vehicle classes
- Efficient memory usage

## Future Improvements
- Implement vehicle tracking
- Add license plate detection
- Integrate with traffic management systems
- Develop web interface for remote monitoring
- Add support for more vehicle types
- Implement traffic flow analysis

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 