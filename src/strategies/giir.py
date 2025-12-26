"""
GIIR (Geometry-Informed Interactive Refinement) Click Strategy

This module implements the GIIR click selection strategy, which uses
distance transforms to select optimal click locations in error regions.

Key Features:
- Deterministic click selection (reproducible)
- Distance transform for finding optimal points
- Robust topology-aware selection for thin vessels
- Skeletonization fallback for thin structures
"""

import numpy as np
import cv2
import torch
from typing import Optional, Dict, Any, List, Tuple
from scipy.ndimage import distance_transform_edt, label as ndimage_label, find_objects
from skimage.morphology import skeletonize

from .base import DeterministicStrategy


class GIIRClickStrategy(DeterministicStrategy):
    """
    Geometry-Informed Interactive Refinement click strategy.
    
    Selects click points using distance transform to find the point
    furthest from boundaries in error regions.
    
    Args:
        mask_radius: Radius around previous clicks to exclude
        min_distance_threshold: Minimum distance for fallback to skeleton
    """
    
    def __init__(
        self,
        mask_radius: int = 2,
        min_distance_threshold: float = 1.5
    ):
        super().__init__()
        self.mask_radius = mask_radius
        self.min_distance_threshold = min_distance_threshold
    
    @property
    def name(self) -> str:
        return "GIIR"
    
    def get_next_click(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        history_points: List[Tuple[int, int]],
        debug: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Select next click using distance transform.
        
        Args:
            pred_mask: Current prediction (0-255)
            gt_mask: Ground truth (0-255)
            history_points: Previous click locations
            debug: Print debug information
            
        Returns:
            Click dictionary or None
        """
        return get_giir_click(
            pred_mask, gt_mask, history_points,
            mask_radius=self.mask_radius,
            min_distance_threshold=self.min_distance_threshold,
            debug=debug
        )


def get_giir_click(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    history_points: List[Tuple[int, int]],
    mask_radius: int = 2,
    min_distance_threshold: float = 1.5,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    GIIR Click Selection: Deterministic using distance transform.
    
    Selects the point FURTHEST from boundaries in error regions.
    This is the PROPOSED method for optimal click placement.
    
    Args:
        pred_mask: Predicted mask (0-255)
        gt_mask: Ground truth mask (0-255)
        history_points: List of (x, y) points already clicked
        mask_radius: Radius to mask around history points
        min_distance_threshold: If max distance < this, use skeleton
        debug: Print debug info
        
    Returns:
        Click dictionary or None if no errors
    """
    # Binarize
    pred_b = pred_mask > 127
    gt_b = gt_mask > 127
    
    # Compute error masks
    fn_mask = np.logical_and(gt_b, ~pred_b).astype(np.uint8)
    fp_mask = np.logical_and(pred_b, ~gt_b).astype(np.uint8)
    
    fn_pixels = fn_mask.sum()
    fp_pixels = fp_mask.sum()
    
    # Early exit if no errors
    if fn_pixels == 0 and fp_pixels == 0:
        if debug:
            print(f"    [DEBUG] No errors: FN=0, FP=0")
        return None
    
    if debug:
        print(f"    [DEBUG] Error pixels: FN={fn_pixels}, FP={fp_pixels}")
    
    # Compute distance transforms
    fn_dist = distance_transform_edt(fn_mask) if fn_pixels > 0 else np.zeros_like(fn_mask, dtype=np.float32)
    fp_dist = distance_transform_edt(fp_mask) if fp_pixels > 0 else np.zeros_like(fp_mask, dtype=np.float32)
    
    fn_max_before = fn_dist.max()
    fp_max_before = fp_dist.max()
    
    # Mask out history points
    for pt in history_points:
        px, py = int(pt[0]), int(pt[1])
        y1 = max(0, py - mask_radius)
        y2 = min(fn_dist.shape[0], py + mask_radius + 1)
        x1 = max(0, px - mask_radius)
        x2 = min(fn_dist.shape[1], px + mask_radius + 1)
        fn_dist[y1:y2, x1:x2] = 0
        fp_dist[y1:y2, x1:x2] = 0
    
    fn_max = fn_dist.max()
    fp_max = fp_dist.max()
    
    if debug:
        print(f"    [DEBUG] Max dist before masking: FN={fn_max_before:.2f}, FP={fp_max_before:.2f}")
        print(f"    [DEBUG] Max dist after masking:  FN={fn_max:.2f}, FP={fp_max:.2f}")
        print(f"    [DEBUG] History points: {len(history_points)}")
    
    # Fallback to random sampling if distance transform is zero
    if fn_max == 0 and fp_max == 0:
        if debug:
            print(f"    [DEBUG] Distance transform zero, trying random sampling")
        
        # Create exclusion mask
        exclusion_mask = np.zeros_like(pred_mask, dtype=bool)
        exclusion_radius = 5
        for pt in history_points:
            px, py = int(pt[0]), int(pt[1])
            y1 = max(0, py - exclusion_radius)
            y2 = min(pred_mask.shape[0], py + exclusion_radius + 1)
            x1 = max(0, px - exclusion_radius)
            x2 = min(pred_mask.shape[1], px + exclusion_radius + 1)
            exclusion_mask[y1:y2, x1:x2] = True
        
        # Find valid points
        valid_fn = fn_mask.astype(bool) & ~exclusion_mask
        valid_fp = fp_mask.astype(bool) & ~exclusion_mask
        
        if valid_fn.sum() > 0:
            coords = np.argwhere(valid_fn)
            idx = np.random.randint(len(coords))
            y, x = coords[idx]
            if debug:
                print(f"    [DEBUG] Random FN point: ({x}, {y})")
            return {'pt': (int(x), int(y)), 'lbl': 1, 'type': 'FN'}
        elif valid_fp.sum() > 0:
            coords = np.argwhere(valid_fp)
            idx = np.random.randint(len(coords))
            y, x = coords[idx]
            if debug:
                print(f"    [DEBUG] Random FP point: ({x}, {y})")
            return {'pt': (int(x), int(y)), 'lbl': 0, 'type': 'FP'}
        else:
            if debug:
                print(f"    [DEBUG] No valid points")
            return None
    
    # Select point with maximum distance
    if fn_max >= fp_max:
        idx = np.unravel_index(np.argmax(fn_dist), fn_dist.shape)
        return {'pt': (int(idx[1]), int(idx[0])), 'lbl': 1, 'type': 'FN'}
    else:
        idx = np.unravel_index(np.argmax(fp_dist), fp_dist.shape)
        return {'pt': (int(idx[1]), int(idx[0])), 'lbl': 0, 'type': 'FP'}


def get_giir_click_train_robust(
    pred_mask_tensor: torch.Tensor,
    gt_mask_tensor: torch.Tensor,
    min_blob_area: int = 10,
    min_distance_threshold: float = 1.5
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    ROBUST TOPOLOGY-AWARE GIIR for Training.
    
    This version is optimized for training with:
    1. Connected component analysis (finds largest error blob)
    2. Skeletonization fallback for thin vessels
    3. Returns tensors for direct use in training
    
    Args:
        pred_mask_tensor: Predicted mask tensor
        gt_mask_tensor: Ground truth mask tensor
        min_blob_area: Minimum blob size to consider
        min_distance_threshold: Threshold for using skeleton
        
    Returns:
        Tuple of (point_tensor, label_tensor) or (None, None)
    """
    # Convert to numpy
    pred_mask = pred_mask_tensor.detach().cpu().numpy().squeeze() > 0.5
    gt_mask = gt_mask_tensor.detach().cpu().numpy().squeeze() > 0.5
    
    # Identify errors
    fn_mask = np.logical_and(gt_mask, ~pred_mask)
    fp_mask = np.logical_and(~gt_mask, pred_mask)
    
    candidates = []
    
    def process_blobs(mask, lbl_type):
        """Process connected components in error mask."""
        if not mask.any():
            return
        labeled, num_features = ndimage_label(mask)
        objects = find_objects(labeled)
        
        for idx, slice_obj in enumerate(objects):
            if slice_obj is None:
                continue
            blob_crop = (labeled[slice_obj] == (idx + 1))
            area = blob_crop.sum()
            
            if area < min_blob_area:
                continue
            
            candidates.append({
                'lbl': lbl_type,
                'area': area,
                'mask': blob_crop,
                'slice': slice_obj
            })
    
    process_blobs(fn_mask, 1)  # FN = positive click
    process_blobs(fp_mask, 0)  # FP = negative click
    
    if not candidates:
        return None, None
    
    # Sort by area (largest first)
    candidates.sort(key=lambda x: x['area'], reverse=True)
    target = candidates[0]
    
    # Find optimal point in the blob
    blob = target['mask'].astype(np.uint8)
    dt = distance_transform_edt(blob)
    max_dist = dt.max()
    
    # Strategy selection
    if max_dist < min_distance_threshold:
        # Thin structure: use skeletonization
        skel = skeletonize(blob)
        if skel.sum() > 0:
            y_ind, x_ind = np.nonzero(skel)
            centroid_y, centroid_x = np.mean(y_ind), np.mean(x_ind)
            dists = (y_ind - centroid_y)**2 + (x_ind - centroid_x)**2
            idx = np.argmin(dists)
            final_y, final_x = y_ind[idx], x_ind[idx]
        else:
            # Fallback: center of mass
            y_ind, x_ind = np.nonzero(blob)
            idx = len(y_ind) // 2
            final_y, final_x = y_ind[idx], x_ind[idx]
    else:
        # Normal: use distance transform max
        idx = np.unravel_index(np.argmax(dt), dt.shape)
        final_y, final_x = idx[0], idx[1]
    
    # Convert local coords to global
    y_global = final_y + target['slice'][0].start
    x_global = final_x + target['slice'][1].start
    
    # Return as tensors (X, Y format)
    pt_tensor = torch.tensor([[x_global, y_global]], dtype=torch.float32)
    lbl_tensor = torch.tensor([target['lbl']], dtype=torch.int64)
    
    return pt_tensor, lbl_tensor


def get_best_click_index_with_threshold(
    metrics_list: List[Dict[str, float]],
    metric_name: str = 'Dice',
    threshold: float = 0.0005
) -> int:
    """
    Find the index of best result with improvement threshold.
    
    Prevents selecting later clicks over earlier ones due to
    microscopic random fluctuations.
    
    Args:
        metrics_list: List of metrics dictionaries
        metric_name: Metric to optimize
        threshold: Minimum improvement required
        
    Returns:
        Index of best click
    """
    values = [m[metric_name] for m in metrics_list]
    
    best_idx = 0
    best_val = values[0]
    
    for i in range(1, len(values)):
        current_val = values[i]
        # Only switch if significant improvement
        if current_val > (best_val + threshold):
            best_val = current_val
            best_idx = i
    
    return best_idx
