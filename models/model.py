# Path: models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Tuple, Dict, Any

class AttentionBlock(nn.Module):
    """Room-boundary guided attention mechanism"""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, room_features: torch.Tensor, boundary_features: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism between room and boundary features
        """
        # Concatenate features
        combined = torch.cat([room_features, boundary_features], dim=1)
        
        # Generate attention weights
        attention_weights = self.conv_attention(combined)
        
        # Apply attention to room features
        attended_features = room_features * attention_weights
        
        return attended_features + room_features  # Residual connection

class CrossTaskFusion(nn.Module):
    """Cross-task feature fusion module"""
    
    def __init__(self, channels: int):
        super(CrossTaskFusion, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Fuse features from two tasks"""
        combined = torch.cat([x1, x2], dim=1)
        fused = self.fusion_conv(combined)
        return fused + x1  # Residual connection

class ModernFloorPlanNet(nn.Module):
    """
    Modern Floor Plan Recognition Network using EfficientNet backbone
    with multi-task learning for room segmentation and boundary detection
    """
    
    def __init__(
        self, 
        encoder_name: str = "efficientnet-b4",
        num_room_classes: int = 9,
        num_boundary_classes: int = 3,
        dropout: float = 0.1,
        use_attention: bool = False  # Start with False for stability
    ):
        super(ModernFloorPlanNet, self).__init__()
        
        self.num_room_classes = num_room_classes
        self.num_boundary_classes = num_boundary_classes
        self.use_attention = use_attention
        
        # Room segmentation network
        self.room_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_room_classes,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        
        # Boundary detection network
        self.boundary_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_boundary_classes,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        
        # Cross-task attention and fusion modules
        if use_attention:
            self.attention_room = AttentionBlock(num_room_classes)
            self.attention_boundary = AttentionBlock(num_boundary_classes)
        
        # Optional dropout layers
        if dropout > 0:
            self.room_dropout = nn.Dropout2d(dropout)
            self.boundary_dropout = nn.Dropout2d(dropout)
        else:
            self.room_dropout = nn.Identity()
            self.boundary_dropout = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary containing room and boundary predictions
        """
        # Get predictions from both branches
        room_logits = self.room_net(x)
        boundary_logits = self.boundary_net(x)
        
        if self.use_attention:
            # Apply cross-task attention
            room_logits = self.attention_room(room_logits, boundary_logits)
            boundary_logits = self.attention_boundary(boundary_logits, room_logits)
        
        # Apply dropout
        room_logits = self.room_dropout(room_logits)
        boundary_logits = self.boundary_dropout(boundary_logits)
        
        return {
            'room_logits': room_logits,
            'boundary_logits': boundary_logits
        }

class SimpleFloorPlanNet(nn.Module):
    """
    Simplified Floor Plan Network - most reliable option
    Uses separate Unet models for each task
    """
    
    def __init__(
        self, 
        encoder_name: str = "efficientnet-b4",
        num_room_classes: int = 9,
        num_boundary_classes: int = 3,
        dropout: float = 0.1
    ):
        super(SimpleFloorPlanNet, self).__init__()
        
        self.num_room_classes = num_room_classes
        self.num_boundary_classes = num_boundary_classes
        
        # Room segmentation network
        self.room_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_room_classes,
        )
        
        # Boundary detection network
        self.boundary_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_boundary_classes,
        )
        
        # Optional dropout layers
        if dropout > 0:
            self.room_dropout = nn.Dropout2d(dropout)
            self.boundary_dropout = nn.Dropout2d(dropout)
        else:
            self.room_dropout = nn.Identity()
            self.boundary_dropout = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        room_logits = self.room_dropout(self.room_net(x))
        boundary_logits = self.boundary_dropout(self.boundary_net(x))
        
        return {
            'room_logits': room_logits,
            'boundary_logits': boundary_logits
        }

class LightweightFloorPlanNet(nn.Module):
    """
    Lightweight version using ResNet backbone - faster training
    """
    
    def __init__(
        self, 
        encoder_name: str = "resnet34",
        num_room_classes: int = 9,
        num_boundary_classes: int = 3,
        dropout: float = 0.1
    ):
        super(LightweightFloorPlanNet, self).__init__()
        
        self.num_room_classes = num_room_classes
        self.num_boundary_classes = num_boundary_classes
        
        # Room segmentation network
        self.room_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_room_classes,
        )
        
        # Boundary detection network
        self.boundary_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_boundary_classes,
        )
        
        # Dropout layers
        if dropout > 0:
            self.room_dropout = nn.Dropout2d(dropout)
            self.boundary_dropout = nn.Dropout2d(dropout)
        else:
            self.room_dropout = nn.Identity()
            self.boundary_dropout = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        room_logits = self.room_dropout(self.room_net(x))
        boundary_logits = self.boundary_dropout(self.boundary_net(x))
        
        return {
            'room_logits': room_logits,
            'boundary_logits': boundary_logits
        }

# Multi-task loss function
class MultiTaskLoss(nn.Module):
    """Multi-task loss combining room and boundary losses"""
    
    def __init__(
        self, 
        room_weight: float = 1.0, 
        boundary_weight: float = 1.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0
    ):
        super(MultiTaskLoss, self).__init__()
        self.room_weight = room_weight
        self.boundary_weight = boundary_weight
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            # Focal loss for handling class imbalance
            self.room_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.boundary_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            # Standard cross-entropy loss
            self.room_loss = nn.CrossEntropyLoss()
            self.boundary_loss = nn.CrossEntropyLoss()
            
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary with 'room_logits' and 'boundary_logits'
            targets: Dictionary with 'room' and 'boundary' ground truth
        """
        room_loss = self.room_loss(predictions['room_logits'], targets['room'])
        boundary_loss = self.boundary_loss(predictions['boundary_logits'], targets['boundary'])
        
        total_loss = self.room_weight * room_loss + self.boundary_weight * boundary_loss
        
        return {
            'total_loss': total_loss,
            'room_loss': room_loss,
            'boundary_loss': boundary_loss
        }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Backward compatibility alias
FloorPlanLoss = MultiTaskLoss

# Factory function to create models
def create_model(model_type: str = "simple", **kwargs) -> nn.Module:
    """
    Factory function to create different model variants
    
    Args:
        model_type: One of ['simple', 'modern', 'lightweight']
        **kwargs: Model-specific arguments
    """
    if model_type == "simple":
        return SimpleFloorPlanNet(**kwargs)
    elif model_type == "modern":
        return ModernFloorPlanNet(**kwargs)
    elif model_type == "lightweight":
        return LightweightFloorPlanNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: simple, modern, lightweight")

# For backward compatibility with existing training code
def get_model(encoder_name="efficientnet-b4", num_room_classes=9, num_boundary_classes=3, **kwargs):
    """Backward compatibility function"""
    return SimpleFloorPlanNet(
        encoder_name=encoder_name,
        num_room_classes=num_room_classes,
        num_boundary_classes=num_boundary_classes,
        **kwargs
    )

# Export all important classes and functions
__all__ = [
    'ModernFloorPlanNet', 
    'SimpleFloorPlanNet', 
    'LightweightFloorPlanNet',
    'MultiTaskLoss', 
    'FloorPlanLoss',
    'FocalLoss',
    'create_model',
    'get_model'
]