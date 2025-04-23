import torch
import torch.nn as nn
from utils.loss import YOLOv8Loss

def test_yolov8_loss():
    # Test parameters
    batch_size = 2
    num_classes = 80
    img_size = 640
    scales = [80, 40, 20]  # Feature map sizes for 3 scales
    
    # Initialize loss function
    loss_fn = YOLOv8Loss(num_classes=num_classes)
    
    # Create dummy predictions and targets for each scale
    preds = []
    targets = []
    
    for scale in scales:
        # Create predictions [B, 3*(5+num_classes), H, W]
        pred = torch.randn(batch_size, 3*(5+num_classes), scale, scale)
        preds.append(pred)
        
        # Create targets [B, 3*6, H, W]
        target = torch.zeros(batch_size, 3*6, scale, scale)
        # Add some random objects
        num_objects = 5
        for b in range(batch_size):
            for _ in range(num_objects):
                # Random position
                x = torch.randint(0, scale, (1,))
                y = torch.randint(0, scale, (1,))
                # Random anchor
                a = torch.randint(0, 3, (1,))
                # Set object presence
                target[b, a*6+4, y, x] = 1.0
                # Set random box coordinates
                target[b, a*6:a*6+4, y, x] = torch.rand(4,1)
                # Set random class
                target[b, a*6+5, y, x] = torch.randint(0, num_classes, (1,)).float()
        
        targets.append(target)
    
    # Convert to tuples
    preds = tuple(preds)
    targets = tuple(targets)
    
    # Calculate loss
    total_loss, loss_components = loss_fn(preds, targets)
    
    # Print results
    print(f"Total Loss: {total_loss.item():.4f}")
    print("\nLoss Components per Scale:")
    for i, (box_loss, cls_loss, dfl_loss, obj_loss) in enumerate(loss_components):
        print(f"Scale {i+1}:")
        print(f"  Box Loss: {box_loss.item():.4f}")
        print(f"  Class Loss: {cls_loss.item():.4f}")
        print(f"  DFL Loss: {dfl_loss.item():.4f}")
        print(f"  Object Loss: {obj_loss.item():.4f}")
    
    # Verify shapes and values
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0  # Scalar
    assert total_loss > 0  # Loss should be positive
    
    assert isinstance(loss_components, torch.Tensor)
    assert loss_components.shape == (3, 4)  # 3 scales, 4 loss components
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_yolov8_loss() 