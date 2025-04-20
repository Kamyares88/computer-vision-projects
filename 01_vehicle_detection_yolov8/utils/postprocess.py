import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """
    Perform non-maximum suppression on predictions.
    
    Args:
        prediction: Raw predictions from the model
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: List of classes to keep
        agnostic: Whether to use class-agnostic NMS
        max_det: Maximum number of detections to keep
    
    Returns:
        List of detections, each detection is a tensor of shape [n, 6]
        where n is the number of detections and each row is [x1, y1, x2, y2, conf, class]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[..., 4:].max(1)[0] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # maximum box width and height
    max_nms = 30000  # maximum number of boxes into NMS
    redundant = True  # require redundant detections

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if nc == 1:
            x[:, 5:] = x[:, 4:5]  # for models with one class
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        j = torch.argmax(x[:, 5:], 1)
        x = torch.cat((box, x[:, 5+j:6+j], j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2) format.
    """
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape: Tuple[int, int], boxes: torch.Tensor,
                img0_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Rescale boxes from img1_shape to img0_shape.
    """
    # Calculate gain
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, \
          (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    # Rescale boxes
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain

    # Clip boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    return boxes 