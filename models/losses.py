# Path: models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [N, C, H, W]
            targets: Ground truth [N, H, W]
        """
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [N, C, H, W]
            targets: Ground truth [N, H, W]
        """
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes-1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Create mask to ignore certain indices
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss
    """
    
    def __init__(
        self, 
        ce_weight: float = 0.5, 
        dice_weight: float = 0.5,
        ignore_index: int = -100
    ):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice

class BoundaryLoss(nn.Module):
    """
    Specialized loss for boundary detection that emphasizes edge pixels
    """
    
    def __init__(self, boundary_weight: float = 2.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def _get_boundary_weight_map(self, targets: torch.Tensor) -> torch.Tensor:
        """Create weight map that emphasizes boundary pixels"""
        # Create boundary detection kernel
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1], 
                              [-1, -1, -1]], dtype=torch.float32, device=targets.device)
        kernel = kernel.view(1, 1, 3, 3)
        
        # Apply to each target
        targets_float = targets.float().unsqueeze(1)
        boundaries = F.conv2d(targets_float, kernel, padding=1).abs()
        boundaries = (boundaries > 0).float().squeeze(1)
        
        # Create weight map: higher weights for boundary pixels
        weight_map = torch.ones_like(targets, dtype=torch.float32)
        weight_map[boundaries > 0] = self.boundary_weight
        
        return weight_map
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross entropy
        ce_loss = self.ce_loss(inputs, targets)
        
        # Get boundary weights
        weight_map = self._get_boundary_weight_map(targets)
        
        # Apply weights
        weighted_loss = ce_loss * weight_map
        
        return weighted_loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for room segmentation and boundary detection
    """
    
    def __init__(
        self,
        room_weight: float = 1.0,
        boundary_weight: float = 1.0,
        loss_type: str = 'focal',  # 'focal', 'ce', 'dice', 'combined'
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_boundary_enhancement: bool = True
    ):
        super(MultiTaskLoss, self).__init__()
        
        self.room_weight = room_weight
        self.boundary_weight = boundary_weight
        self.use_boundary_enhancement = use_boundary_enhancement
        
        # Initialize loss functions based on type
        if loss_type == 'focal':
            self.room_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.boundary_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'ce':
            self.room_loss_fn = nn.CrossEntropyLoss()
            self.boundary_loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            self.room_loss_fn = DiceLoss()
            self.boundary_loss_fn = DiceLoss()
        elif loss_type == 'combined':
            self.room_loss_fn = CombinedLoss()
            self.boundary_loss_fn = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Add boundary enhancement for room loss
        if use_boundary_enhancement:
            self.boundary_enhanced_loss = BoundaryLoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss
        
        Args:
            predictions: Dict with 'room_logits' and 'boundary_logits'
            targets: Dict with 'room_mask' and 'boundary_mask'
        """
        room_logits = predictions['room_logits']
        boundary_logits = predictions['boundary_logits']
        room_targets = targets['room_mask']
        boundary_targets = targets['boundary_mask']
        
        # Calculate individual losses
        room_loss = self.room_loss_fn(room_logits, room_targets)
        boundary_loss = self.boundary_loss_fn(boundary_logits, boundary_targets)
        
        # Add boundary enhancement to room loss
        if self.use_boundary_enhancement:
            boundary_enhanced_room_loss = self.boundary_enhanced_loss(room_logits, room_targets)
            room_loss = 0.7 * room_loss + 0.3 * boundary_enhanced_room_loss
        
        # Compute total weighted loss
        total_loss = (
            self.room_weight * room_loss + 
            self.boundary_weight * boundary_loss
        )
        
        return {
            'total_loss': total_loss,
            'room_loss': room_loss,
            'boundary_loss': boundary_loss
        }

class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive loss weighting that adjusts weights during training
    """
    
    def __init__(
        self,
        num_tasks: int = 2,
        loss_type: str = 'focal',
        temperature: float = 2.0
    ):
        super(AdaptiveWeightedLoss, self).__init__()
        
        # Learnable weights for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.temperature = temperature
        
        # Base loss functions
        if loss_type == 'focal':
            self.room_loss_fn = FocalLoss()
            self.boundary_loss_fn = FocalLoss()
        else:
            self.room_loss_fn = nn.CrossEntropyLoss()
            self.boundary_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate adaptive weighted multi-task loss
        """
        room_loss = self.room_loss_fn(predictions['room_logits'], targets['room_mask'])
        boundary_loss = self.boundary_loss_fn(predictions['boundary_logits'], targets['boundary_mask'])
        
        # Adaptive weighting using learnable parameters
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        
        total_loss = (
            precision1 * room_loss + self.log_vars[0] +
            precision2 * boundary_loss + self.log_vars[1]
        )
        
        return {
            'total_loss': total_loss,
            'room_loss': room_loss,
            'boundary_loss': boundary_loss,
            'room_weight': precision1.item(),
            'boundary_weight': precision2.item()
        }

class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced Cross Entropy Loss for handling class imbalance
    """
    
    def __init__(self, ignore_index: int = -100):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate balanced cross entropy loss based on class frequencies
        """
        # Calculate class weights based on frequency
        unique_classes, counts = torch.unique(targets[targets != self.ignore_index], return_counts=True)
        total_pixels = counts.sum().float()
        
        # Compute inverse frequency weights
        weights = torch.ones(inputs.size(1), device=inputs.device)
        for i, class_id in enumerate(unique_classes):
            if class_id != self.ignore_index:
                weights[class_id] = total_pixels / (len(unique_classes) * counts[i].float())
        
        # Apply weighted cross entropy
        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        return loss_fn(inputs, targets)