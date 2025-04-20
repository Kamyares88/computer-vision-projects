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

    def forward(self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate loss between predictions and targets for all three scales."""
        total_loss = 0
        loss_components = []
        
        for pred, target in zip(preds, targets):
            # Reshape predictions and targets
            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes, H, W)
            target = target.view(B, 3, 6, H, W)
            
            # Split predictions
            pred_bbox = pred[:, :, :4]  # [B, 3, 4, H, W]
            pred_obj = pred[:, :, 4:5]  # [B, 3, 1, H, W]
            pred_cls = pred[:, :, 5:]   # [B, 3, num_classes, H, W]
            
            # Split targets
            target_bbox = target[:, :, :4]  # [B, 3, 4, H, W]
            target_obj = target[:, :, 4:5]  # [B, 3, 1, H, W]
            target_cls = target[:, :, 5:]   # [B, 3, 1, H, W]
            
            # Calculate losses
            box_loss = self.box_loss(pred_bbox, target_bbox, target_obj)
            obj_loss = self.obj_loss(pred_obj, target_obj)
            cls_loss = self.cls_loss(pred_cls, target_cls, target_obj)
            dfl_loss = self.dfl(pred_bbox, target_bbox, target_obj)
            
            # Combine losses
            scale_loss = (self.box_gain * box_loss + 
                         self.cls_gain * cls_loss + 
                         self.dfl_gain * dfl_loss +
                         obj_loss)
            
            total_loss += scale_loss
            loss_components.append(torch.stack((box_loss, cls_loss, dfl_loss, obj_loss)))
        
        return total_loss, torch.stack(loss_components)

    def box_loss(self, pred: torch.Tensor, target: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
        """Calculate box regression loss."""
        # Calculate IoU
        iou = self.calculate_iou(pred, target)
        # Apply object mask
        iou = iou * obj_mask.squeeze(-1)
        return 1.0 - iou.mean()

    def obj_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate objectness loss."""
        return self.bce(pred, target).mean()

    def cls_loss(self, pred: torch.Tensor, target: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss."""
        # Only calculate loss for cells with objects
        loss = self.bce(pred, target) * obj_mask
        return loss.mean()

    @staticmethod
    def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=2)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=2)
        
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
    """
    Distribution Focal Loss (DFL) for bounding box regression.
    
    DFL treats box coordinates as discrete probability distributions rather than
    direct regression values. This helps in learning more accurate and stable
    bounding box predictions.
    
    The loss encourages the model to predict a smooth distribution around the
    target value, with higher probabilities for values closer to the target.
    
    Args:
        reg_max (int): Maximum value in the regression range
        use_sigmoid (bool): Whether to apply sigmoid to predictions
    """
    def __init__(self, reg_max: int = 16, use_sigmoid: bool = True):
        super().__init__()
        self.reg_max = reg_max
        self.use_sigmoid = use_sigmoid
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                obj_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate DFL loss.
        
        Args:
            pred: Predicted distribution [B, 3, 4, reg_max, H, W]
            target: Target values [B, 3, 4, H, W]
            obj_mask: Object presence mask [B, 3, 1, H, W]
            
        Returns:
            torch.Tensor: DFL loss
        """
        # Create target distribution
        target = target.unsqueeze(-2)  # [B, 3, 4, 1, H, W]
        target_left = target.floor()
        target_right = target_left + 1
        
        # Calculate weights for left and right values
        weight_right = target - target_left
        weight_left = 1 - weight_right
        
        # Create one-hot targets
        target_left = target_left.long()
        target_right = target_right.long()
        
        # Create target distributions
        target_dist_left = F.one_hot(target_left, self.reg_max)
        target_dist_right = F.one_hot(target_right, self.reg_max)
        
        # Calculate losses
        if self.use_sigmoid:
            pred = pred.sigmoid()
        
        loss_left = self.bce(pred, target_dist_left) * weight_left
        loss_right = self.bce(pred, target_dist_right) * weight_right
        
        # Combine losses and apply object mask
        loss = (loss_left + loss_right).mean(dim=-1)  # [B, 3, 4, H, W]
        loss = loss * obj_mask.squeeze(-2)
        
        return loss.mean()

    

        