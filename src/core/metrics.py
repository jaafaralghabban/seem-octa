"""
Evaluation Metrics for OCTA Vessel Segmentation

This module provides comprehensive metrics for evaluating segmentation quality:
- Dice Score (F1)
- IoU (Jaccard Index)
- Sensitivity (Recall)
- Specificity
- clDice (Centerline Dice for topology)
"""

import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional
from skimage.morphology import skeletonize


def compute_metrics(
    pred: torch.Tensor, 
    gt: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Compute basic segmentation metrics.
    
    Args:
        pred: Predicted mask (binary or probability)
        gt: Ground truth mask
        threshold: Threshold for binarization if pred is probability
        
    Returns:
        Tuple of (dice, iou, sensitivity, specificity)
    """
    # Ensure binary tensors
    if pred.dtype != torch.bool:
        pred_bool = (pred > threshold).cpu().bool()
    else:
        pred_bool = pred.cpu().bool()
        
    if gt.dtype != torch.bool:
        gt_bool = (gt > threshold).cpu().bool()
    else:
        gt_bool = gt.cpu().bool()
    
    # Confusion matrix elements
    tp = (pred_bool & gt_bool).sum().float()
    tn = (~pred_bool & ~gt_bool).sum().float()
    fp = (pred_bool & ~gt_bool).sum().float()
    fn = (~pred_bool & gt_bool).sum().float()
    
    epsilon = 1e-6
    
    # Dice = 2*TP / (2*TP + FP + FN)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + epsilon)
    
    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn + epsilon)
    
    # Sensitivity = TP / (TP + FN)
    sensitivity = tp / (tp + fn + epsilon)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp + epsilon)
    
    return dice.item(), iou.item(), sensitivity.item(), specificity.item()


def compute_cldice(
    pred: np.ndarray, 
    gt: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Compute Centerline Dice Score for topology evaluation.
    
    clDice measures how well the predicted segmentation preserves
    the topology (connectivity) of vessel structures.
    
    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
    where:
        Tprec = |S(P) ∩ G| / |S(P)|  (skeleton precision)
        Tsens = |S(G) ∩ P| / |S(G)|  (skeleton sensitivity)
    
    Args:
        pred: Predicted binary mask (numpy array)
        gt: Ground truth binary mask (numpy array)
        smooth: Smoothing factor
        
    Returns:
        clDice score
    """
    # Ensure binary
    pred_b = pred > 0.5 if pred.max() <= 1 else pred > 127
    gt_b = gt > 0.5 if gt.max() <= 1 else gt > 127
    
    # Handle edge cases
    if gt_b.sum() == 0:
        return 1.0 if pred_b.sum() == 0 else 0.0
    if pred_b.sum() == 0:
        return 0.0
    
    # Compute skeletons
    pred_skel = skeletonize(pred_b)
    gt_skel = skeletonize(gt_b)
    
    # Topology precision: skeleton of prediction inside GT
    t_prec = np.sum(pred_skel * gt_b) / (np.sum(pred_skel) + smooth)
    
    # Topology sensitivity: skeleton of GT inside prediction
    t_sens = np.sum(gt_skel * pred_b) / (np.sum(gt_skel) + smooth)
    
    # clDice
    cldice = 2 * t_prec * t_sens / (t_prec + t_sens + smooth)
    
    return float(cldice)


def calculate_metrics(
    pred: Union[np.ndarray, torch.Tensor], 
    gt: Union[np.ndarray, torch.Tensor],
    threshold: float = 127.0
) -> Dict[str, float]:
    """
    Calculate all segmentation metrics.
    
    Args:
        pred: Predicted mask (0-255 scale)
        gt: Ground truth mask (0-255 scale)
        threshold: Binarization threshold
        
    Returns:
        Dictionary with all metrics
    """
    # Convert to numpy if tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # Ensure 2D
    pred = pred.squeeze()
    gt = gt.squeeze()
    
    # Binarize
    pred_b = pred > threshold
    gt_b = gt > threshold
    
    # Confusion matrix
    tp = np.logical_and(pred_b, gt_b).sum()
    tn = np.logical_and(~pred_b, ~gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    
    smooth = 1e-6
    
    # Basic metrics
    dice = (2.0 * tp) / (pred_b.sum() + gt_b.sum() + smooth)
    iou = tp / (tp + fp + fn + smooth)
    se = tp / (tp + fn + smooth)  # Sensitivity
    sp = tn / (tn + fp + smooth)  # Specificity
    
    # Centerline Dice
    cldice = compute_cldice(pred, gt)
    
    return {
        'Dice': float(dice),
        'IoU': float(iou),
        'clDice': float(cldice),
        'SE': float(se),
        'SP': float(sp)
    }


def calculate_batch_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    gts: Union[np.ndarray, torch.Tensor],
    threshold: float = 127.0
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate metrics for a batch of predictions.
    
    Args:
        preds: Batch of predicted masks [B, H, W]
        gts: Batch of ground truth masks [B, H, W]
        threshold: Binarization threshold
        
    Returns:
        Dictionary with (mean, std) for each metric
    """
    # Convert to numpy if tensor
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().numpy()
    
    metrics_list = []
    for pred, gt in zip(preds, gts):
        metrics_list.append(calculate_metrics(pred, gt, threshold))
    
    # Aggregate
    result = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        result[key] = (np.mean(values), np.std(values))
    
    return result


class MetricsAccumulator:
    """
    Accumulator for tracking metrics over multiple batches/epochs.
    
    Example:
        metrics = MetricsAccumulator()
        for batch in dataloader:
            pred = model(batch)
            metrics.update(pred, batch['gt'])
        print(metrics.compute())
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all accumulated values."""
        self.metrics = {
            'Dice': [],
            'IoU': [],
            'clDice': [],
            'SE': [],
            'SP': []
        }
        self.count = 0
        
    def update(
        self, 
        pred: Union[np.ndarray, torch.Tensor],
        gt: Union[np.ndarray, torch.Tensor],
        threshold: float = 127.0
    ):
        """
        Update accumulator with new prediction.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            threshold: Binarization threshold
        """
        batch_metrics = calculate_metrics(pred, gt, threshold)
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
        self.count += 1
        
    def compute(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean and std of accumulated metrics.
        
        Returns:
            Dictionary with (mean, std) for each metric
        """
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = (np.mean(values), np.std(values))
            else:
                result[key] = (0.0, 0.0)
        return result
    
    def compute_mean(self) -> Dict[str, float]:
        """
        Compute only mean of accumulated metrics.
        
        Returns:
            Dictionary with mean for each metric
        """
        result = {}
        for key, values in self.metrics.items():
            result[key] = np.mean(values) if values else 0.0
        return result


def print_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Metrics Comparison"
) -> str:
    """
    Format metrics as a printable table.
    
    Args:
        metrics_dict: Dictionary of {method_name: {metric_name: value}}
        title: Table title
        
    Returns:
        Formatted table string
    """
    methods = list(metrics_dict.keys())
    metric_names = ['Dice', 'IoU', 'clDice', 'SE', 'SP']
    
    # Header
    lines = [
        "=" * 80,
        f"  {title}",
        "=" * 80,
        f"{'Method':<20} " + " ".join([f"{m:<10}" for m in metric_names]),
        "-" * 80
    ]
    
    # Data rows
    for method in methods:
        values = metrics_dict[method]
        row = f"{method:<20} "
        for metric in metric_names:
            val = values.get(metric, 0.0)
            if isinstance(val, tuple):
                row += f"{val[0]:.4f}±{val[1]:.4f} "
            else:
                row += f"{val:.4f}     "
        lines.append(row)
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
