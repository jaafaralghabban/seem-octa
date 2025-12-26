"""
Point Generation Strategies for Interactive Segmentation

This module provides different strategies for generating initial click points
for interactive segmentation training and validation.
"""

import numpy as np
import cv2
from typing import Tuple
from scipy.ndimage import distance_transform_edt


def generate_random_points(
    mask: np.ndarray,
    max_points: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random points from foreground and background.
    
    SEEM baseline style: simple random sampling.
    
    Args:
        mask: Binary mask (values 0-1 or 0-255)
        max_points: Maximum number of points to generate
        
    Returns:
        Tuple of (points [N, 2], labels [N])
    """
    # Ensure binary
    mask_np = (mask.squeeze() > 0.5).astype(np.uint8)
    
    points, labels = [], []
    
    # Get foreground and background coordinates
    fg_coords = np.argwhere(mask_np == 1)[:, [1, 0]]  # [x, y] format
    bg_coords = np.argwhere(mask_np == 0)[:, [1, 0]]
    
    n_pos = max_points // 2
    n_neg = max_points - n_pos
    
    # Sample positive points
    if len(fg_coords) > 0:
        replace = len(fg_coords) < n_pos
        indices = np.random.choice(len(fg_coords), n_pos, replace=replace)
        for idx in indices:
            points.append(fg_coords[idx])
            labels.append(1)
    
    # Sample negative points
    if len(bg_coords) > 0:
        replace = len(bg_coords) < n_neg
        indices = np.random.choice(len(bg_coords), n_neg, replace=replace)
        for idx in indices:
            points.append(bg_coords[idx])
            labels.append(0)
    
    # Format output
    final_points = np.full((max_points, 2), -1.0, dtype=np.float32)
    final_labels = np.full((max_points,), -1, dtype=np.int64)
    
    num_generated = len(points)
    if num_generated > 0:
        shuffled_indices = np.random.permutation(num_generated)
        shuffled_points = np.array(points)[shuffled_indices]
        shuffled_labels = np.array(labels)[shuffled_indices]
        final_points[:num_generated] = shuffled_points
        final_labels[:num_generated] = shuffled_labels
    
    return final_points, final_labels


def generate_hybrid_points(
    mask: np.ndarray,
    max_points: int = 10,
    dilation_kernel_size: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine geometric (smart) sampling with random sampling.
    
    Strategy:
    - Positive: 50% center points (using distance transform), 50% random
    - Negative: 50% boundary points (dilated region), 50% random
    
    This provides more informative initial points for training/validation.
    
    Args:
        mask: Binary mask
        max_points: Maximum points to generate
        dilation_kernel_size: Kernel size for boundary dilation
        
    Returns:
        Tuple of (points [N, 2], labels [N])
    """
    mask_np = (mask.squeeze() > 0.5).astype(np.uint8)
    points, labels = [], []
    
    n_pos = max_points // 2
    n_neg = max_points - n_pos
    n_pos_center = n_pos // 2
    n_neg_boundary = n_neg // 2
    
    # === POSITIVE POINTS ===
    fg_coords = np.argwhere(mask_np == 1)[:, [1, 0]]  # [x, y]
    
    # A. Center points using distance transform
    if len(fg_coords) > 0 and n_pos_center > 0:
        dist_transform = distance_transform_edt(mask_np)
        flat_dists = dist_transform[dist_transform > 0]
        
        if len(flat_dists) > 0:
            threshold = np.percentile(flat_dists, 50)
            core_candidates = np.argwhere(dist_transform >= threshold)[:, [1, 0]]
            
            if len(core_candidates) > 0:
                replace = len(core_candidates) < n_pos_center
                idxs = np.random.choice(len(core_candidates), n_pos_center, replace=replace)
                for i in idxs:
                    points.append(core_candidates[i])
                    labels.append(1)
    
    # B. Random positive points
    needed_random_pos = n_pos - len(points)
    if len(fg_coords) > 0 and needed_random_pos > 0:
        replace = len(fg_coords) < needed_random_pos
        idxs = np.random.choice(len(fg_coords), needed_random_pos, replace=replace)
        for i in idxs:
            points.append(fg_coords[i])
            labels.append(1)
    
    # === NEGATIVE POINTS ===
    bg_coords = np.argwhere(mask_np == 0)[:, [1, 0]]
    
    # A. Boundary points (in dilated region but outside mask)
    if n_neg_boundary > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        boundary_region = dilated - mask_np
        boundary_candidates = np.argwhere(boundary_region > 0)[:, [1, 0]]
        
        if len(boundary_candidates) > 0:
            replace = len(boundary_candidates) < n_neg_boundary
            idxs = np.random.choice(len(boundary_candidates), n_neg_boundary, replace=replace)
            for i in idxs:
                points.append(boundary_candidates[i])
                labels.append(0)
    
    # B. Random negative points
    needed_random_neg = max_points - len(points)
    if len(bg_coords) > 0 and needed_random_neg > 0:
        replace = len(bg_coords) < needed_random_neg
        idxs = np.random.choice(len(bg_coords), needed_random_neg, replace=replace)
        for i in idxs:
            points.append(bg_coords[i])
            labels.append(0)
    
    # === FORMAT OUTPUT ===
    final_points = np.full((max_points, 2), -1.0, dtype=np.float32)
    final_labels = np.full((max_points,), -1, dtype=np.int64)
    
    num_generated = len(points)
    if num_generated > 0:
        shuffled_indices = np.random.permutation(num_generated)
        shuffled_points = np.array(points)[shuffled_indices]
        shuffled_labels = np.array(labels)[shuffled_indices]
        limit = min(num_generated, max_points)
        final_points[:limit] = shuffled_points[:limit]
        final_labels[:limit] = shuffled_labels[:limit]
    
    return final_points, final_labels


def generate_stratified_points(
    mask: np.ndarray,
    max_points: int = 10,
    n_strata: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points using spatial stratification.
    
    Divides the image into a grid and samples from each cell
    to ensure spatial coverage.
    
    Args:
        mask: Binary mask
        max_points: Maximum points to generate
        n_strata: Number of grid divisions per axis
        
    Returns:
        Tuple of (points [N, 2], labels [N])
    """
    mask_np = (mask.squeeze() > 0.5).astype(np.uint8)
    h, w = mask_np.shape
    
    points, labels = [], []
    
    cell_h = h // n_strata
    cell_w = w // n_strata
    
    n_pos = max_points // 2
    n_neg = max_points - n_pos
    
    # Collect positive and negative points per cell
    pos_candidates = []
    neg_candidates = []
    
    for i in range(n_strata):
        for j in range(n_strata):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            
            cell_mask = mask_np[y_start:y_end, x_start:x_end]
            
            # Foreground in this cell
            fg = np.argwhere(cell_mask == 1)
            if len(fg) > 0:
                idx = np.random.randint(len(fg))
                pos_candidates.append((fg[idx][1] + x_start, fg[idx][0] + y_start))
            
            # Background in this cell
            bg = np.argwhere(cell_mask == 0)
            if len(bg) > 0:
                idx = np.random.randint(len(bg))
                neg_candidates.append((bg[idx][1] + x_start, bg[idx][0] + y_start))
    
    # Sample from candidates
    if pos_candidates:
        np.random.shuffle(pos_candidates)
        for pt in pos_candidates[:n_pos]:
            points.append(pt)
            labels.append(1)
    
    if neg_candidates:
        np.random.shuffle(neg_candidates)
        for pt in neg_candidates[:n_neg]:
            points.append(pt)
            labels.append(0)
    
    # Format output
    final_points = np.full((max_points, 2), -1.0, dtype=np.float32)
    final_labels = np.full((max_points,), -1, dtype=np.int64)
    
    num_generated = len(points)
    if num_generated > 0:
        shuffled_indices = np.random.permutation(num_generated)
        shuffled_points = np.array(points)[shuffled_indices]
        shuffled_labels = np.array(labels)[shuffled_indices]
        final_points[:num_generated] = shuffled_points
        final_labels[:num_generated] = shuffled_labels
    
    return final_points, final_labels
