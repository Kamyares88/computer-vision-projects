import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import time
import cv2
import numpy as np

from models.yolov8 import YOLOv8
from utils.loss import YOLOv8Loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train Custom YOLOv8')
    parser.add_argument('--data', type=str, default='data/vehicles.yaml', help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (cpu, 0, 1, ...)')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use (SGD, Adam)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    return parser.parse_args()

def create_dataset_config():
    """Create dataset configuration file."""
    config = {
        'path': '../datasets/vehicles',  # Path to dataset
        'train': 'images/train',  # Path to train images
        'val': 'images/val',      # Path to validation images
        'names': {
            0: 'car',
            1: 'bus',
            2: 'truck'
        }
    }
    
    config_path = Path('data/vehicles.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

class VehicleDataset(torch.utils.data.Dataset):
    """Custom dataset for vehicle detection."""
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.img_files = list(self.img_dir.glob('*.jpg'))
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / f'{img_path.stem}.txt'
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    labels.append([cls, x, y, w, h])
        
        # Preprocess
        img, labels = self.preprocess(img, labels)
        
        return img, torch.tensor(labels)
    
    def preprocess(self, img, labels):
        """Preprocess image and labels."""
        # Resize
        h, w = img.shape[:2]
        r = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * r), int(w * r)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to tensor
        img = img.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # Normalize
        
        # Adjust labels
        if labels:
            labels = torch.tensor(labels)
            labels[:, 1:] *= r  # Scale coordinates
            labels[:, 1] += left / self.img_size  # Adjust x
            labels[:, 2] += top / self.img_size   # Adjust y
        
        return img, labels

def train():
    args = parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else 'cpu')
    
    # Create dataset config if it doesn't exist
    if not Path(args.data).exists():
        print("Creating dataset configuration file...")
        args.data = create_dataset_config()
    
    # Load dataset config
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = VehicleDataset(
        Path(data_config['path']) / data_config['train'],
        Path(data_config['path']) / 'labels/train',
        args.img_size
    )
    val_dataset = VehicleDataset(
        Path(data_config['path']) / data_config['val'],
        Path(data_config['path']) / 'labels/val',
        args.img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = YOLOv8(num_classes=len(data_config['names'])).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    # Initialize loss function
    criterion = YOLOv8Loss(num_classes=len(data_config['names']))
    
    # Initialize optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(imgs)
            loss, loss_items = criterion(preds, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Validation loss: {val_loss:.4f}')
        
        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'weights/yolov8_epoch{epoch + 1}.pt')

if __name__ == '__main__':
    train() 