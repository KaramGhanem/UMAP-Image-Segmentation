# Contains evaluation and post-processing tools.
# Post-processing uses morphological operations to clean masks.
# Citation: Morphological ops adapted from OpenCV examples - https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

import torch
import numpy as np
import cv2
import torch.nn as nn

# --- Loss Functions ---
# Dice loss formulation adapted from commonly used segmentation practice
# https://github.com/milesial/Pytorch-UNet/tree/master/utils

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    
    Dice Loss measures the overlap between predicted and ground truth masks.
    It's particularly effective for segmentation tasks where the target objects
    may occupy a small portion of the image (class imbalance).
    
    Formula: Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    Loss = 1 - Dice
    
    Where:
    - X is the predicted mask
    - Y is the ground truth mask
    - ∩ represents intersection
    - |X| represents the sum of all pixels in X
    
    Attributes:
        smooth (float): Smoothing factor to prevent division by zero. Default: 1e-6
    
    Example:
        >>> criterion = DiceLoss()
        >>> loss = criterion(predictions, targets)
        >>> # predictions: [B, 1, H, W] or [B, H, W] tensor of logits
        >>> # targets: [B, H, W] tensor of binary masks (0 or 1)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to logits if not already applied
        if pred.max() > 1.0:
            pred = torch.sigmoid(pred)
        
        # Handle tensor shape mismatches
        if target.dim() == 3 and pred.dim() == 4:
            # Add channel dimension to target to match pred [B, 1, H, W]
            target = target.unsqueeze(1)
        elif target.dim() == 4 and pred.dim() == 4:
            if target.size(1) == 1 and pred.size(1) == 1:
                # Both have channel dimension, keep as is
                pass
            elif target.size(1) == 1:
                # Remove channel dimension from target
                target = target.squeeze(1)
            elif pred.size(1) == 1:
                # Remove channel dimension from pred
                pred = pred.squeeze(1)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute intersection and Dice score
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for binary segmentation.
    
    This loss function combines the benefits of both BCE and Dice losses:
    - BCE: Provides pixel-wise classification loss, good for overall learning
    - Dice: Focuses on overlap between predicted and ground truth masks, 
            handles class imbalance well
    
    The combined loss is particularly effective for segmentation tasks because:
    1. BCE ensures each pixel is classified correctly
    2. Dice ensures the overall shape and overlap are optimized
    3. Together they provide better convergence and final performance
    
    Formula: Loss = BCE(pred, target) + DiceLoss(pred, target)
    
    Attributes:
        bce: Binary Cross Entropy with Logits Loss
        dice: Dice Loss
    
    Example:
        >>> criterion = BCEDiceLoss()
        >>> loss = criterion(predictions, targets)
        >>> # predictions: [B, 1, H, W] or [B, H, W] tensor of logits
        >>> # targets: [B, H, W] tensor of binary masks (0 or 1)
    
    Note:
        - BCE expects logits (raw model output)
        - Dice handles sigmoid internally if needed
        - The combination often leads to better segmentation results than either loss alone
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        # Ensure targets have correct shape for BCE
        if targets.dim() == 3 and logits.dim() == 4:
            # Add channel dimension to targets to match logits [B, 1, H, W]
            targets = targets.unsqueeze(1)
        elif targets.dim() == 4 and logits.dim() == 4:
            if targets.size(1) == 1 and logits.size(1) == 1:
                # Both have channel dimension, keep as is
                pass
            elif targets.size(1) == 1:
                # Remove channel dimension from targets
                targets = targets.squeeze(1)
        
        # Combine BCE and Dice losses
        return self.bce(logits, targets) + self.dice(logits, targets)

def iou_score(output, target):
    """
    Computes IoU (Jaccard Index) for binary mask prediction.
    """
    # Handle tensor shape mismatches
    if target.dim() == 3 and output.dim() == 4:
        # Add channel dimension to target to match output [B, 1, H, W]
        target = target.unsqueeze(1)
    elif target.dim() == 4 and output.dim() == 4:
        if target.size(1) == 1 and output.size(1) == 1:
            # Both have channel dimension, keep as is
            pass
        elif target.size(1) == 1:
            # Remove channel dimension from target
            target = target.squeeze(1)
        elif output.size(1) == 1:
            # Remove channel dimension from output
            output = output.squeeze(1)
    
    # Binarize output
    output = (output > 0.5).float()
    
    # Flatten tensors
    output_flat = output.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and union
    intersection = (output_flat * target_flat).sum()
    union = output_flat.sum() + target_flat.sum() - intersection
    
    # Handle edge case where both are empty
    if union == 0:
        return 1.0
    
    # Ensure IoU is between 0 and 1
    iou = intersection / union
    return torch.clamp(iou, 0.0, 1.0)

def post_process(mask):
    """
    Applies morphological opening and closing to smooth out mask edges.
    """
    mask = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
