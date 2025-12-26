"""
Model Utilities for SEEM-OCTA

This module provides utilities for:
- Loading and setting up SEEM model
- Running inference with/without points
- Mask selection strategies
- Morphological post-processing
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Optional, List, Tuple, Dict, Any
from skimage.morphology import remove_small_objects

# SEEM imports (adjust paths as needed for your setup)
try:
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utils.arguments import load_opt_from_config_files
    from utils.distributed import init_distributed
    from utils.constants import COCO_PANOPTIC_CLASSES
except ImportError:
    print("Warning: SEEM modules not found. Make sure modeling/ and utils/ are in your path.")


# Default parameters
DEFAULT_CONFIG = "configs/seem/focall_unicl_lang_demo.yaml"
DEFAULT_THRESHOLDS_ORACLE = [0.1, 0.2, 0.4, 0.5, 0.65]
DEFAULT_THRESHOLDS_RANDOM = [0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_CLEAN_SIZE = 20
DEFAULT_PATCH_CLEAN_SIZE = 5


def setup_model(
    weights_path: str,
    config_path: str = DEFAULT_CONFIG,
    device: str = 'cuda'
) -> Any:
    """
    Initialize and load SEEM model.
    
    Args:
        weights_path: Path to model weights
        config_path: Path to model config YAML
        device: Target device
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model: {os.path.basename(weights_path)}")
    
    # Load config
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)
    
    # Build model
    model = BaseModel(opt, build_model(opt))
    
    # Load weights
    ckpt = torch.load(weights_path, map_location=device)
    model.model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    
    # Set to eval mode
    model.eval()
    model.to(device)
    
    # Initialize text embeddings
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], 
            is_eval=True
        )
    
    print(f"✓ Model loaded successfully")
    return model


@torch.no_grad()
def get_initial_mask(
    model: Any,
    img_pil: Image.Image,
    resize_dim: int = 640
) -> Tuple[np.ndarray, torch.Tensor, int, int]:
    """
    Get initial segmentation without any clicks.
    
    Args:
        model: SEEM model
        img_pil: Input PIL image (can be 3-channel)
        resize_dim: Dimension to resize to for inference
        
    Returns:
        Tuple of (initial_mask, image_tensor, transformed_h, transformed_w)
    """
    # Transform
    transform = transforms.Compose([
        transforms.Resize(resize_dim, interpolation=Image.BICUBIC)
    ])
    
    # Handle input - could be 3-channel grayscale stack or RGB
    img_np = np.array(transform(img_pil))
    
    # Ensure 3 channels
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=2)
    
    # Convert to tensor [C, H, W]
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).cuda()
    
    oh, ow = img_pil.height, img_pil.width
    th, tw = img_t.shape[1], img_t.shape[2]
    
    # Prepare data for model
    data = {
        "image": img_t,
        "height": th,
        "width": tw,
        "visual": {
            "points": torch.zeros((1, 0, 2)).cuda().float(),
            "label": torch.zeros((1, 0)).cuda().float()
        }
    }
    
    # Set task switches
    model.model.task_switch['panoptic'] = True
    model.model.task_switch['visual'] = False
    model.model.task_switch['spatial'] = True
    
    # Run inference
    results, _, _ = model.model.evaluate_demo([data])
    
    if 'pred_masks' not in results:
        return np.zeros((oh, ow), dtype=np.uint8), img_t, th, tw
    
    # Get mask and resize to original size
    mask = results['pred_masks'][0, 0, :th, :tw]
    mask = F.interpolate(
        mask[None, None], 
        (oh, ow), 
        mode='bilinear', 
        align_corners=False
    ).squeeze()
    
    mask_np = (mask.sigmoid() > 0.5).cpu().numpy().astype(np.uint8) * 255
    
    return mask_np, img_t, th, tw


@torch.no_grad()
def inference_with_points(
    model: Any,
    img_t: torch.Tensor,
    pts: torch.Tensor,
    lbls: torch.Tensor,
    oh: int, ow: int, th: int, tw: int
) -> Optional[List[np.ndarray]]:
    """
    Run inference with click points.
    
    Args:
        model: SEEM model
        img_t: Image tensor [C, H, W]
        pts: Points tensor [1, N, 2]
        lbls: Labels tensor [1, N]
        oh, ow: Original image dimensions
        th, tw: Transformed image dimensions
        
    Returns:
        List of probability masks or None
    """
    # Prepare data
    data = {
        "image": img_t,
        "height": th,
        "width": tw,
        "visual": {
            "points": pts,
            "label": lbls
        }
    }
    
    # Set task switches
    model.model.task_switch['panoptic'] = True
    model.model.task_switch['visual'] = False
    model.model.task_switch['spatial'] = True
    
    # Run inference
    results, _, _ = model.model.evaluate_demo([data])
    
    if 'pred_masks' not in results:
        return None
    
    # Extract all mask queries
    masks = []
    num_queries = results['pred_masks'].shape[1]
    
    for i in range(num_queries):
        m = results['pred_masks'][0, i, :th, :tw]
        m = F.interpolate(
            m[None, None], 
            (oh, ow), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        masks.append(m.sigmoid().cpu().numpy())
    
    return masks


def find_best_matching_mask(
    pred_masks_logits: torch.Tensor, 
    gt_mask: torch.Tensor
) -> torch.Tensor:
    """
    Find the mask query with best overlap with GT.
    
    Used during training to select which query to optimize.
    
    Args:
        pred_masks_logits: Model output logits [B, N_queries, H, W]
        gt_mask: Ground truth mask [B, 1, H, W]
        
    Returns:
        Best matching mask logits [B, 1, H, W]
    """
    num_queries = pred_masks_logits.shape[1]
    best_idx, best_dice = -1, -1.0
    
    with torch.no_grad():
        gt_mask_flat = gt_mask.view(-1)
        
        for i in range(num_queries):
            query_logits = pred_masks_logits[0, i, :, :]
            query_probs_flat = torch.sigmoid(query_logits).view(-1)
            
            intersection = (query_probs_flat * gt_mask_flat).sum()
            union = query_probs_flat.sum() + gt_mask_flat.sum()
            dice = (2.0 * intersection) / (union + 1e-6)
            
            if dice > best_dice:
                best_dice = dice
                best_idx = i
    
    if best_idx == -1:
        best_idx = 0
        
    return pred_masks_logits[:, best_idx, :, :].unsqueeze(1)


def select_mask_oracle(
    masks: List[np.ndarray],
    gt: np.ndarray,
    thresholds: List[float] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Oracle mask selection: select best mask using GT.
    
    Used in GIIR training and evaluation.
    
    Args:
        masks: List of probability masks from model
        gt: Ground truth mask
        thresholds: List of thresholds to try
        
    Returns:
        Tuple of (best_mask_binary, info_dict)
    """
    if not masks:
        return None, {}
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_ORACLE
    
    from .metrics import calculate_metrics
    
    best_dice = -1
    best_mask = None
    best_info = {}
    
    for mask_idx, mask_prob in enumerate(masks):
        for thresh in thresholds:
            mask_bin = (mask_prob > thresh).astype(np.uint8) * 255
            
            if mask_bin.sum() > 0:
                metrics = calculate_metrics(mask_bin, gt)
                dice = metrics['Dice']
                
                if dice > best_dice:
                    best_dice = dice
                    best_mask = mask_bin
                    best_info = {
                        'mask_idx': mask_idx,
                        'threshold': thresh,
                        'dice': dice
                    }
    
    return best_mask, best_info


def select_mask_random(
    masks: List[np.ndarray],
    iteration_number: int,
    seed: int,
    thresholds: List[float] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Random mask selection: SEEM baseline behavior.
    
    Args:
        masks: List of probability masks
        iteration_number: Current click number (for seeding)
        seed: Base random seed
        thresholds: List of possible thresholds
        
    Returns:
        Tuple of (selected_mask_binary, info_dict)
    """
    if not masks:
        return None, {}
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_RANDOM
    
    # Set reproducible seed
    np.random.seed(seed + iteration_number * 100 + 7)
    
    # Random selections
    mask_idx = np.random.randint(len(masks))
    thresh = np.random.choice(thresholds)
    
    mask_bin = (masks[mask_idx] > thresh).astype(np.uint8) * 255
    
    # Fallback if empty
    if mask_bin.sum() == 0:
        for alt_thresh in thresholds:
            mask_bin = (masks[mask_idx] > alt_thresh).astype(np.uint8) * 255
            if mask_bin.sum() > 0:
                return mask_bin.copy(), {'mask_idx': mask_idx, 'threshold': alt_thresh}
        return None, {}
    
    return mask_bin.copy(), {'mask_idx': mask_idx, 'threshold': thresh}


def select_mask_giir_enhanced(
    masks: List[np.ndarray],
    gt: np.ndarray,
    current_mask: np.ndarray,
    click_pt: Tuple[int, int],
    click_lbl: int,
    local_radius: int = 30,
    thresholds: List[float] = None,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    GIIR Enhanced: Click-guided mask selection with local refinement.
    
    Combines:
    1. Global structure from oracle-selected best mask
    2. Local corrections guided by click position
    
    Args:
        masks: List of probability masks
        gt: Ground truth mask
        current_mask: Current segmentation
        click_pt: (x, y) click position
        click_lbl: 1 for positive, 0 for negative
        local_radius: Radius for local refinement region
        thresholds: Thresholds for mask selection
        debug: Print debug info
        
    Returns:
        Enhanced mask or None
    """
    if not masks:
        return None
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_ORACLE
    
    from .metrics import calculate_metrics
    
    # Step 1: Find best global mask
    best_dice, best_mask_prob, best_thresh = -1, None, 0.5
    
    for mask_idx, mask_prob in enumerate(masks):
        for thresh in thresholds:
            mask_bin = (mask_prob > thresh).astype(np.uint8) * 255
            if mask_bin.sum() > 0:
                dice = calculate_metrics(mask_bin, gt)['Dice']
                if dice > best_dice:
                    best_dice = dice
                    best_mask_prob = mask_prob
                    best_thresh = thresh
    
    if best_mask_prob is None:
        return None
    
    # Step 2: Create base mask
    base_mask = (best_mask_prob > best_thresh).astype(np.uint8) * 255
    
    # Step 3: Define local region around click
    cx, cy = int(click_pt[0]), int(click_pt[1])
    h, w = base_mask.shape
    Y, X = np.ogrid[:h, :w]
    local_region = ((X - cx)**2 + (Y - cy)**2 <= local_radius**2)
    
    # Step 4: Find locally optimal mask
    best_local_dice = -1
    best_local_mask = None
    
    for mask_prob in masks:
        for thresh in thresholds:
            mask_bin = (mask_prob > thresh).astype(np.uint8) * 255
            
            local_pred = mask_bin[local_region]
            local_gt = gt[local_region]
            
            if local_gt.sum() > 0 or local_pred.sum() > 0:
                intersection = np.logical_and(local_pred > 127, local_gt > 127).sum()
                local_dice = 2 * intersection / (local_pred.sum()/255 + local_gt.sum()/255 + 1e-8)
                
                if local_dice > best_local_dice:
                    best_local_dice = local_dice
                    best_local_mask = mask_bin
    
    # Step 5: Combine global + local
    if best_local_mask is not None:
        result = base_mask.copy()
        result[local_region] = best_local_mask[local_region]
        
        if debug:
            combined_dice = calculate_metrics(result, gt)['Dice']
            print(f"      [GIIR Enhanced] Global Dice={best_dice:.4f}, "
                  f"Local Dice={best_local_dice:.4f}, Combined={combined_dice:.4f}")
        
        return result
    
    return base_mask


def apply_morphological_cleaning(
    mask: np.ndarray,
    min_size: int = DEFAULT_PATCH_CLEAN_SIZE
) -> np.ndarray:
    """
    Clean small regions from mask using morphological operations.
    
    Args:
        mask: Input binary mask (0-255)
        min_size: Minimum region size to keep
        
    Returns:
        Cleaned mask
    """
    if mask is None:
        return None
    
    cleaned = remove_small_objects(mask > 127, min_size=min_size)
    return (cleaned * 255).astype(np.uint8)


def apply_mask_update(
    new_mask: np.ndarray,
    min_size: int = DEFAULT_CLEAN_SIZE
) -> Optional[np.ndarray]:
    """
    Apply morphological cleaning to new mask.
    
    In standard interactive segmentation:
    - Model receives ALL accumulated clicks
    - Model outputs COMPLETE new segmentation
    - We just clean artifacts and use directly
    
    Args:
        new_mask: New mask from model
        min_size: Minimum region size to keep
        
    Returns:
        Cleaned mask or None
    """
    if new_mask is None:
        return None
    
    return apply_morphological_cleaning(new_mask, min_size=min_size)


def load_3channel_image(
    sample_id: str,
    data_dir: str,
    dataset: str = "OCTA_3mm"
) -> Tuple[Image.Image, Image.Image]:
    """
    Load and stack 3 OCTA projections as a 3-channel image.
    
    Args:
        sample_id: Filename (e.g., "10491.bmp")
        data_dir: Base data directory
        dataset: Dataset subdirectory (OCTA_3mm or OCTA_6mm)
        
    Returns:
        Tuple of (stacked_pil, viz_pil)
    """
    import cv2
    
    projection_paths = [
        os.path.join(data_dir, dataset, "OCTA(FULL)", sample_id),
        os.path.join(data_dir, dataset, "OCTA(ILM_OPL)", sample_id),
        os.path.join(data_dir, dataset, "OCTA(OPL_BM)", sample_id)
    ]
    
    projections = []
    for path in projection_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load projection: {path}")
        projections.append(img)
    
    # Stack as 3-channel: [FULL, ILM_OPL, OPL_BM]
    stacked_np = np.stack(projections, axis=2)
    stacked_pil = Image.fromarray(stacked_np)
    
    # For visualization, use ILM_OPL as RGB
    viz_pil = Image.fromarray(projections[1]).convert("RGB")
    
    return stacked_pil, viz_pil
