"""
Loss Functions for OCTA Vessel Segmentation

This module provides various loss functions optimized for vessel segmentation:
- DiceLoss: Region-based overlap loss
- FocalLoss: Handles class imbalance
- TverskyLoss: Weighted FP/FN for thin vessels
- CombinedLoss: Weighted combination of the above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Ground truth masks (0 or 1)
            
        Returns:
            Dice loss value
        """
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Ground truth masks (0 or 1)
            
        Returns:
            Focal loss value
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting: alpha if y=1, else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for controlling FP/FN trade-off.
    
    Tversky Index = TP / (TP + α*FP + β*FN)
    
    For thin vessel segmentation, β > α emphasizes reducing false negatives.
    
    Args:
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing factor (default: 1e-6)
    """
    
    def __init__(
        self, 
        alpha: float = 0.5, 
        beta: float = 0.5, 
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Ground truth masks (0 or 1)
            
        Returns:
            Tversky loss value
        """
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CombinedLoss(nn.Module):
    """
    Combined loss function for vessel segmentation.
    
    Combines Dice, Focal, and Tversky losses with configurable weights.
    Default configuration is optimized for OCTA vessel segmentation.
    
    Args:
        dice_weight: Weight for Dice loss (default: 0.4)
        focal_weight: Weight for Focal loss (default: 0.3)
        tversky_weight: Weight for Tversky loss (default: 0.3)
        tversky_alpha: Tversky FP weight (default: 0.3)
        tversky_beta: Tversky FN weight (default: 0.7)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.4,
        focal_weight: float = 0.3,
        tversky_weight: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Ground truth masks (0 or 1)
            
        Returns:
            Weighted combined loss value
        """
        loss = (
            self.dice_weight * self.dice(logits, targets) +
            self.focal_weight * self.focal(logits, targets) +
            self.tversky_weight * self.tversky(logits, targets)
        )
        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for precise vessel edge segmentation.
    
    Computes loss with higher weight near vessel boundaries.
    
    Args:
        theta: Temperature for boundary weight computation
        smooth: Smoothing factor
    """
    
    def __init__(self, theta: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.theta = theta
        self.smooth = smooth
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        boundary_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute boundary-weighted loss.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth masks
            boundary_weights: Optional pre-computed boundary weights
            
        Returns:
            Boundary-weighted loss value
        """
        probs = torch.sigmoid(logits)
        
        if boundary_weights is None:
            # Compute boundary weights using gradient magnitude
            # This approximates distance to boundary
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32, device=targets.device)
            sobel_y = sobel_x.t()
            
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            grad_x = F.conv2d(targets, sobel_x, padding=1)
            grad_y = F.conv2d(targets, sobel_y, padding=1)
            
            boundary = torch.sqrt(grad_x**2 + grad_y**2)
            boundary_weights = 1 + self.theta * boundary
        
        # Weighted BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = boundary_weights * bce
        
        return weighted_bce.mean()


class clDiceLoss(nn.Module):
    """
    Centerline Dice Loss for topology-aware vessel segmentation.
    
    Uses skeletonization to compute topology-preserving loss.
    Note: Requires skimage, may be slow for training.
    
    Args:
        smooth: Smoothing factor (default: 1e-6)
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute clDice loss (soft approximation).
        
        For training efficiency, this uses a soft approximation.
        For exact clDice, use the metrics module.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth masks
            
        Returns:
            Approximate clDice loss
        """
        probs = torch.sigmoid(logits)
        
        # Soft skeletonization using morphological operations
        # This is an approximation for differentiability
        kernel_size = 3
        
        # Erosion approximation
        pred_eroded = -F.max_pool2d(-probs, kernel_size, stride=1, padding=kernel_size//2)
        target_eroded = -F.max_pool2d(-targets, kernel_size, stride=1, padding=kernel_size//2)
        
        # Soft skeleton = original - eroded
        pred_skel = probs - pred_eroded
        target_skel = targets - target_eroded
        
        # clDice components
        t_prec = (pred_skel * targets).sum() / (pred_skel.sum() + self.smooth)
        t_sens = (target_skel * probs).sum() / (target_skel.sum() + self.smooth)
        
        cldice = 2 * t_prec * t_sens / (t_prec + t_sens + self.smooth)
        
        return 1.0 - cldice
