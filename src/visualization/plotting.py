"""
Visualization Utilities for SEEM-OCTA

This module provides plotting functions for:
- Training curves
- Test predictions
- Comparison results
- Error maps
- Click progression visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Optional, Tuple
import random


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', 
                 label='Training Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s',
                 label='Validation Loss', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Dice curve
    axes[1].plot(epochs, history['val_dice'], 'g-^',
                 label='Validation Dice', linewidth=2, markersize=4)
    best_dice = max(history['val_dice'])
    best_epoch = history['val_dice'].index(best_dice) + 1
    axes[1].axhline(y=best_dice, color='red', linestyle='--', alpha=0.5,
                    label=f'Best: {best_dice:.4f} (Epoch {best_epoch})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Validation Dice Score', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


def plot_test_predictions(
    dataset: Any,
    model: Any,
    device: str,
    num_samples: int,
    save_path: str
) -> Tuple[float, float, float, float]:
    """
    Plot test predictions with metrics.
    
    Args:
        dataset: Test dataset
        model: Trained model
        device: Device string
        num_samples: Number of samples to plot
        save_path: Path to save the figure
        
    Returns:
        Tuple of (avg_dice, avg_iou, avg_se, avg_sp)
    """
    from ..core.metrics import compute_metrics
    import torch.nn.functional as F
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Prediction', fontsize=12, fontweight='bold')
    
    model.eval()
    total_dice, total_iou, total_se, total_sp = 0, 0, 0, 0
    
    with torch.no_grad():
        for i in range(num_samples):
            sample_idx = random.randint(0, len(dataset) - 1)
            sample_data = dataset[sample_idx]
            
            image = sample_data['image'].to(device).unsqueeze(0)
            gt_mask = sample_data['mask']
            valid_mask = sample_data['labels'] != -1
            points = sample_data['points'][valid_mask].to(device).unsqueeze(0)
            labels = sample_data['labels'][valid_mask].to(device).unsqueeze(0)
            
            if points.size(1) == 0:
                pred_binary = torch.zeros_like(gt_mask).squeeze(0)
            else:
                prompts = {"point_coords": points, "point_labels": labels}
                features = model.backbone(image)
                mask_feat, _, ms_feat = model.sem_seg_head.pixel_decoder.forward_features(features)
                preds = model.sem_seg_head.predictor(ms_feat, mask_feat, extra=prompts, task='panoptic')
                logits = F.interpolate(
                    preds['pred_masks'][:, 0, :, :].unsqueeze(1),
                    size=gt_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                pred_binary = (torch.sigmoid(logits) > 0.5)[0, 0]
            
            dice, iou, se, sp = compute_metrics(pred_binary, gt_mask.to(device))
            total_dice += dice
            total_iou += iou
            total_se += se
            total_sp += sp
            
            # Prepare for visualization
            img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            mean = np.array(dataset.norm_mean)
            std = np.array(dataset.norm_std)
            img_np = (img_np * std + mean).clip(0, 1)
            gt_np = gt_mask.squeeze(0).numpy()
            pred_np = pred_binary.cpu().numpy()
            
            axes[i, 0].imshow(img_np)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(gt_np, cmap='gray')
            axes[i, 1].axis('off')
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].axis('off')
            axes[i, 2].set_xlabel(
                f'Dice: {dice:.3f} | IoU: {iou:.3f}\nSE: {se:.3f} | SP: {sp:.3f}',
                fontsize=11
            )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_se = total_se / num_samples
    avg_sp = total_sp / num_samples
    
    print(f"✓ Test predictions saved to {save_path}")
    print(f"  Avg Metrics: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, "
          f"SE={avg_se:.4f}, SP={avg_sp:.4f}")
    
    return avg_dice, avg_iou, avg_se, avg_sp


def create_error_map(
    pred: np.ndarray,
    gt: np.ndarray
) -> np.ndarray:
    """
    Create RGB error visualization.
    
    Colors:
    - White: True Positive
    - Red: False Positive
    - Green: False Negative
    - Black: True Negative
    
    Args:
        pred: Predicted mask (0-255)
        gt: Ground truth mask (0-255)
        
    Returns:
        RGB error map
    """
    pred_b = pred > 127
    gt_b = gt > 127
    
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # TP -> White
    tp = np.logical_and(pred_b, gt_b)
    rgb[tp] = [255, 255, 255]
    
    # FP -> Red
    fp = np.logical_and(pred_b, ~gt_b)
    rgb[fp] = [255, 0, 0]
    
    # FN -> Green
    fn = np.logical_and(~pred_b, gt_b)
    rgb[fn] = [0, 255, 0]
    
    return rgb


def plot_comparison_results(
    all_results: List[Dict[str, Any]],
    output_dir: str,
    num_clicks: int = 10
) -> None:
    """
    Create comprehensive comparison plots for GIIR vs SEEM.
    
    Args:
        all_results: List of result dictionaries from evaluation
        output_dir: Directory to save plots
        num_clicks: Maximum number of clicks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    num_samples = len(all_results)
    sample_indices = list(range(num_samples))
    
    initial_dices = [r['initial']['Dice'] for r in all_results]
    giir_best_dices = [r['giir']['best_metrics']['Dice'] for r in all_results]
    giir_best_clicks = [r['giir']['best_click'] for r in all_results]
    
    # SEEM averages across seeds
    seem_best_dices = [
        np.mean([sr['best_metrics']['Dice'] for sr in r['seem_seed_results']])
        for r in all_results
    ]
    seem_best_clicks = [
        np.mean([sr['best_click'] for sr in r['seem_seed_results']])
        for r in all_results
    ]
    
    # Figure 1: 4-Panel Summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Dice across samples
    ax1 = axes[0, 0]
    ax1.plot(sample_indices, initial_dices, 'gray', linestyle='--', 
             marker='o', ms=6, label='Initial')
    ax1.plot(sample_indices, giir_best_dices, 'b-o', lw=2, ms=8, label='GIIR (Ours)')
    ax1.plot(sample_indices, seem_best_dices, 'g-^', lw=2, ms=8, label='SEEM Baseline')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Dice Score Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Panel 2: Click count comparison
    ax2 = axes[0, 1]
    x = np.arange(num_samples)
    width = 0.35
    ax2.bar(x - width/2, giir_best_clicks, width, label='GIIR', color='blue', alpha=0.7)
    ax2.bar(x + width/2, seem_best_clicks, width, label='SEEM', color='green', alpha=0.7)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Number of Clicks', fontsize=12)
    ax2.set_title('Click Count Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim([0, num_clicks + 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Box plot distribution
    ax3 = axes[1, 0]
    all_seem_dices = []
    for r in all_results:
        for sr in r['seem_seed_results']:
            all_seem_dices.append(sr['best_metrics']['Dice'])
    
    box_data = [initial_dices, giir_best_dices, all_seem_dices]
    box_labels = ['Initial', 'GIIR', 'SEEM']
    box_colors = ['gray', 'blue', 'green']
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Dice Score', fontsize=12)
    ax3.set_title('Distribution of Dice Scores', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Mean metrics bar chart
    ax4 = axes[1, 1]
    metrics_to_plot = ['Dice', 'IoU', 'clDice', 'SE', 'SP']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    giir_means = []
    seem_means = []
    for metric in metrics_to_plot:
        giir_vals = [r['giir']['best_metrics'][metric] for r in all_results]
        giir_means.append(np.mean(giir_vals))
        
        seem_vals = []
        for r in all_results:
            for sr in r['seem_seed_results']:
                seem_vals.append(sr['best_metrics'][metric])
        seem_means.append(np.mean(seem_vals))
    
    bars1 = ax4.bar(x - width/2, giir_means, width, label='GIIR', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, seem_means, width, label='SEEM', color='green', alpha=0.7)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Mean Metrics Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_to_plot)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.set_ylim([0.5, 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('GIIR vs SEEM: Batch Comparison Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'batch_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'batch_comparison_summary.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to {output_dir}")


def visualize_click_progression(
    image: np.ndarray,
    gt: np.ndarray,
    masks_history: List[np.ndarray],
    points_history: List[List[Tuple[int, int]]],
    labels_history: List[List[int]],
    save_path: str,
    method_name: str = "Method"
) -> None:
    """
    Visualize the progression of clicks and masks.
    
    Args:
        image: Original image (H, W, 3)
        gt: Ground truth mask
        masks_history: List of masks at each click
        points_history: List of accumulated points at each click
        labels_history: List of accumulated labels at each click
        save_path: Path to save visualization
        method_name: Name of the method for title
    """
    num_clicks = len(masks_history)
    fig, axes = plt.subplots(2, min(num_clicks, 10), figsize=(3 * min(num_clicks, 10), 6))
    
    if num_clicks == 1:
        axes = axes.reshape(2, 1)
    
    from ..core.metrics import calculate_metrics
    
    for i, (mask, pts, lbls) in enumerate(zip(masks_history, points_history, labels_history)):
        if i >= 10:
            break
            
        # Top row: Image with clicks
        axes[0, i].imshow(image)
        if pts:
            pts_np = np.array(pts)
            pos_idx = [j for j, l in enumerate(lbls) if l == 1]
            neg_idx = [j for j, l in enumerate(lbls) if l == 0]
            if pos_idx:
                axes[0, i].scatter(pts_np[pos_idx, 0], pts_np[pos_idx, 1],
                                  c='lime', s=60, marker='o', edgecolors='black')
            if neg_idx:
                axes[0, i].scatter(pts_np[neg_idx, 0], pts_np[neg_idx, 1],
                                  c='red', s=60, marker='X', edgecolors='black')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Click {i+1}', fontsize=10)
        
        # Bottom row: Error map
        error_map = create_error_map(mask, gt)
        axes[1, i].imshow(error_map)
        axes[1, i].axis('off')
        
        metrics = calculate_metrics(mask, gt)
        axes[1, i].set_xlabel(f'Dice: {metrics["Dice"]:.3f}', fontsize=9)
    
    plt.suptitle(f'{method_name}: Click Progression', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
