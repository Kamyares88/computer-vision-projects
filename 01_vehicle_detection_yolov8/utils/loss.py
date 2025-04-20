import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes: int, box_gain: float = 7.5,
                 cls_gain: float = 0.5, dfl_gain: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl = DistributionFocalLoss()

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate loss between predictions and targets."""
        # Split predictions
        pred_distri, pred_scores = torch.split(pred, [4, self.num_classes], dim=1)
        
        # Calculate losses
        box_loss = self.box_loss(pred_distri, targets)
        cls_loss = self.cls_loss(pred_scores, targets)
        dfl_loss = self.dfl(pred_distri, targets)
        
        # Combine losses
        loss = self.box_gain * box_loss + self.cls_gain * cls_loss + self.dfl_gain * dfl_loss
        return loss, torch.stack((box_loss, cls_loss, dfl_loss))

    def box_loss(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate box regression loss."""
        # Calculate IoU
        iou = self.calculate_iou(pred, targets)
        return 1.0 - iou.mean()

    def cls_loss(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss."""
        return self.bce(pred, targets).mean()

    @staticmethod
    def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=1)
        
        # Calculate intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        # Calculate IoU
        return inter_area / (union_area + 1e-7)

class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) for box regression."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate DFL loss."""
        # Split predictions into left and right
        pred_left = pred[:, :-1]
        pred_right = pred[:, 1:]
        
        # Calculate target distribution
        target_left = targets.floor()
        target_right = target_left + 1
        weight_left = target_right - targets
        weight_right = targets - target_left
        
        # Calculate losses
        loss_left = self.bce(pred_left, target_left) * weight_left
        loss_right = self.bce(pred_right, target_right) * weight_right
        
        return (loss_left + loss_right).mean() 