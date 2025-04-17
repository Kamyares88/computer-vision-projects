# Action Recognition in Videos

## Project Overview
This project implements a deep learning-based system for recognizing human actions in videos using 3D CNNs and transformer architectures. The system can classify various human actions in real-time or from pre-recorded videos.

## Features
- Real-time action recognition
- Multi-person action detection
- Temporal action localization
- Attention mechanisms for improved accuracy
- Real-time visualization of recognized actions

## Dataset
The project uses the following datasets:
- UCF101 (Action Recognition Dataset)
- HMDB51 (Human Motion Database)
- Kinetics-400 (Large-scale action recognition dataset)

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- MoviePy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python train.py --dataset path/to/dataset --model i3d
python predict.py --model path/to/model --video path/to/video
```

## Project Structure
```
action_recognition_videos/
├── data/               # Dataset and preprocessing scripts
├── models/            # Model architectures (I3D, SlowFast, etc.)
├── training/          # Training scripts and utilities
├── evaluation/        # Evaluation metrics and visualization
├── visualization/     # Visualization tools
└── requirements.txt   # Project dependencies
```

## Future Improvements
- Implement real-time multi-person action recognition
- Add action detection in streaming video
- Develop web interface for video analysis
- Integrate with surveillance systems 