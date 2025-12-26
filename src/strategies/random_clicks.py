"""
Random Click Strategy (SEEM Baseline)

This module implements the random click selection strategy used in
the original SEEM model for interactive segmentation.

Key Features:
- Random sampling from error regions
- Area-based priority (larger errors first)
- Reproducible with seed control
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .base import StochasticStrategy


class RandomClickStrategy(StochasticStrategy):
    """
    Random click selection strategy (SEEM baseline).
    
    Randomly samples points from error regions with priority
    given to the larger error type (FN or FP).
    
    Args:
        seed: Random seed for reproducibility
        exclusion_radius: Radius around history points to exclude
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        exclusion_radius: int = 5
    ):
        super().__init__(seed)
        self.exclusion_radius = exclusion_radius
        self._iteration = 0
    
    @property
    def name(self) -> str:
        return "Random"
    
    def get_next_click(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        history_points: List[Tuple[int, int]],
        iteration_number: Optional[int] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Select next click randomly from error regions.
        
        Args:
            pred_mask: Current prediction (0-255)
            gt_mask: Ground truth (0-255)
            history_points: Previous click locations
            iteration_number: Override for iteration (for seeding)
            
        Returns:
            Click dictionary or None
        """
        if iteration_number is None:
            iteration_number = self._iteration
            self._iteration += 1
        
        return get_random_click(
            pred_mask, gt_mask, history_points,
            iteration_number=iteration_number,
            seed=self.seed if self.seed else 42,
            exclusion_radius=self.exclusion_radius
        )
    
    def reset(self):
        """Reset iteration counter."""
        super().reset()
        self._iteration = 0


def get_random_click(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    history_points: List[Tuple[int, int]],
    iteration_number: int,
    seed: int,
    exclusion_radius: int = 5,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    SEEM Baseline Strategy: Random sampling from error regions.
    
    This version is optimized for medical imaging by NOT removing
    small error regions, ensuring thin vessel segments are not ignored.
    
    Args:
        pred_mask: Predicted mask (0-255 or 0-1)
        gt_mask: Ground truth mask (0-255 or 0-1)
        history_points: List of (x, y) points already clicked
        iteration_number: Current click number (for seeding)
        seed: Base random seed
        exclusion_radius: Radius around history points to exclude
        
    Returns:
        Click dictionary or None
    """
    # Set seed for reproducible randomness
    np.random.seed(seed + iteration_number * 100)
    
    # Binarize masks
    pred_b = pred_mask > 0.5 if pred_mask.max() <= 1 else pred_mask > 127
    gt_b = gt_mask > 0.5 if gt_mask.max() <= 1 else gt_mask > 127
    
    # Compute raw error masks (NO cleaning for medical images)
    fn_mask = np.logical_and(gt_b, ~pred_b)
    fp_mask = np.logical_and(pred_b, ~gt_b)
    
    # Decide priority based on total error area
    use_fn_priority = fn_mask.sum() >= fp_mask.sum()
    
    # Create exclusion mask around history points
    exclusion_mask = np.zeros_like(pred_b, dtype=bool)
    if history_points:
        for hx, hy in history_points:
            ymin = max(0, int(hy) - exclusion_radius)
            ymax = min(pred_b.shape[0], int(hy) + exclusion_radius + 1)
            xmin = max(0, int(hx) - exclusion_radius)
            xmax = min(pred_b.shape[1], int(hx) + exclusion_radius + 1)
            exclusion_mask[ymin:ymax, xmin:xmax] = True
    
    # Find valid click coordinates
    valid_fn_mask = np.logical_and(fn_mask, ~exclusion_mask)
    valid_fp_mask = np.logical_and(fp_mask, ~exclusion_mask)
    
    coords, click_type = None, ''
    
    # Prioritize and select
    if use_fn_priority:
        if valid_fn_mask.sum() > 0:
            coords, click_type = np.argwhere(valid_fn_mask), 'FN'
        elif valid_fp_mask.sum() > 0:
            coords, click_type = np.argwhere(valid_fp_mask), 'FP'
    else:
        if valid_fp_mask.sum() > 0:
            coords, click_type = np.argwhere(valid_fp_mask), 'FP'
        elif valid_fn_mask.sum() > 0:
            coords, click_type = np.argwhere(valid_fn_mask), 'FN'
    
    # Sample if coordinates found
    if coords is not None and len(coords) > 0:
        # np.argwhere returns (row, col) = (y, x)
        y, x = coords[np.random.randint(len(coords))]
        return {
            'pt': (int(x), int(y)),
            'lbl': 1 if click_type == 'FN' else 0,
            'type': click_type,
            'method': 'random'
        }
    
    return None


def get_seem_interactive_click(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    history_points: List[Tuple[int, int]],
    iteration_number: int,
    seed: int
) -> Optional[Dict[str, Any]]:
    """
    Alias for get_random_click for backward compatibility.
    
    This is the exact function used in academic comparison scripts.
    """
    return get_random_click(
        pred_mask, gt_mask, history_points,
        iteration_number, seed
    )
