import cv2
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path
import time

def draw_vehicle_count(frame: np.ndarray, count: int) -> np.ndarray:
    """Draw vehicle count on the frame."""
    cv2.putText(frame, f'Vehicles: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def calculate_speed(bbox1: List[float], bbox2: List[float], 
                   fps: float, pixel_to_meter: float = 0.1) -> float:
    """
    Calculate speed of a vehicle between two frames.
    
    Args:
        bbox1: Bounding box in first frame [x1, y1, x2, y2]
        bbox2: Bounding box in second frame [x1, y1, x2, y2]
        fps: Frames per second
        pixel_to_meter: Conversion factor from pixels to meters
    
    Returns:
        Speed in km/h
    """
    # Calculate center points
    center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
    center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
    
    # Calculate distance in pixels
    distance_pixels = np.sqrt((center2[0] - center1[0])**2 + 
                            (center2[1] - center1[1])**2)
    
    # Convert to meters and calculate speed
    distance_meters = distance_pixels * pixel_to_meter
    speed_mps = distance_meters * fps
    speed_kmh = speed_mps * 3.6
    
    return speed_kmh

def create_output_dir(base_dir: Union[str, Path] = 'output') -> Path:
    """Create output directory with timestamp."""
    base_dir = Path(base_dir)
    timestamp = base_dir / time.strftime("%Y%m%d-%H%M%S")
    timestamp.mkdir(parents=True, exist_ok=True)
    return timestamp

def get_video_writer(output_path: Union[str, Path], 
                    width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Create video writer for saving output."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

def preprocess_frame(frame: np.ndarray, 
                    target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess frame for model input."""
    # Resize
    frame = cv2.resize(frame, target_size)
    # Normalize
    frame = frame.astype(np.float32) / 255.0
    return frame 