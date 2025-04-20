import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import time

from models.yolov8 import YOLOv8
from utils.postprocess import non_max_suppression, scale_boxes

def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle Detection using Custom YOLOv8')
    parser.add_argument('--source', type=str, default='0', help='Source for detection (0 for webcam, path for video/image)')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--device', type=str, default='0', help='Device to use (cpu, 0, 1, ...)')
    return parser.parse_args()

def preprocess(img: np.ndarray, img_size: int = 640) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for model input."""
    # Resize
    h, w = img.shape[:2]
    r = min(img_size / h, img_size / w)
    new_h, new_w = int(h * r), int(w * r)
    img = cv2.resize(img, (new_w, new_h))
    
    # Pad
    top = (img_size - new_h) // 2
    bottom = img_size - new_h - top
    left = (img_size - new_w) // 2
    right = img_size - new_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Convert to tensor
    img = img.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0  # Normalize
    if len(img.shape) == 3:
        img = img[None]  # Add batch dimension
    
    return img, (h, w)

def draw_boxes(img: np.ndarray, boxes: torch.Tensor, names: List[str]) -> np.ndarray:
    """Draw bounding boxes on image."""
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f'{names[cls]} {conf:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else 'cpu')
    
    # Initialize model
    model = YOLOv8(num_classes=80).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    # Class names
    names = ['car', 'truck', 'bus']  # Vehicle classes
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if save is enabled
    if args.save:
        output_path = Path('output')
        output_path.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_path / f'vehicle_detection_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        img, (h, w) = preprocess(frame)
        img = img.to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(img)
        
        # Post-process
        pred = non_max_suppression(pred, args.conf, args.iou, classes=[2, 5, 7])[0]  # Filter vehicle classes
        if len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], (h, w))
        
        # Draw results
        frame = draw_boxes(frame, pred, names)
        
        # Display vehicle count
        vehicle_count = len(pred)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Vehicle Detection', frame)
        
        # Save frame if enabled
        if args.save:
            out.write(frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 