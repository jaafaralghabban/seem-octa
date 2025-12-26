"""
SEEM-OCTA Evaluator

This module provides evaluation utilities for comparing different
interactive segmentation strategies (GIIR vs Random/SEEM baseline).
"""

import os
import time
import json
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from ..core.metrics import calculate_metrics
from ..core.model_utils import (
    get_initial_mask,
    inference_with_points,
    select_mask_giir_enhanced,
    select_mask_random,
    apply_mask_update,
    load_3channel_image,
)
from ..strategies.giir import get_giir_click, get_best_click_index_with_threshold
from ..strategies.random_clicks import get_random_click
from ..training.config import EvaluationConfig


class SEEMOCTAEvaluator:
    """
    Evaluator for SEEM-OCTA models.
    
    Supports evaluation with both GIIR and Random click strategies.
    """
    
    def __init__(
        self,
        model: Any,
        config: EvaluationConfig = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config or EvaluationConfig()
        self.device = device
    
    def evaluate_giir(
        self,
        img_t: torch.Tensor,
        gt: np.ndarray,
        init_mask: np.ndarray,
        oh: int, ow: int, th: int, tw: int,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Run GIIR evaluation on a single sample."""
        curr_mask = init_mask.copy()
        pts, lbls = [], []
        
        metrics_at_click = [calculate_metrics(curr_mask, gt)]
        
        for click_num in range(1, self.config.num_clicks + 1):
            click = get_giir_click(curr_mask, gt, pts, debug=debug)
            
            if click is None:
                metrics_at_click.append(metrics_at_click[-1].copy())
                continue
            
            pts.append(click['pt'])
            lbls.append(click['lbl'])
            
            pts_np = np.array(pts, dtype=float)
            pts_np[:, 0] *= tw / ow
            pts_np[:, 1] *= th / oh
            
            masks = inference_with_points(
                self.model, img_t,
                torch.from_numpy(pts_np[None]).cuda().float(),
                torch.from_numpy(np.array(lbls)[None]).cuda().float(),
                oh, ow, th, tw
            )
            
            if masks is not None:
                new_mask = select_mask_giir_enhanced(
                    masks, gt, curr_mask,
                    click['pt'], click['lbl'],
                    debug=debug
                )
                if new_mask is not None:
                    curr_mask = apply_mask_update(new_mask)
                    if curr_mask is None:
                        curr_mask = init_mask.copy()
            
            metrics_at_click.append(calculate_metrics(curr_mask, gt))
        
        best_idx = get_best_click_index_with_threshold(metrics_at_click, 'Dice', threshold=0.0005)
        
        return {
            'metrics_at_click': metrics_at_click,
            'best_click': best_idx,
            'best_metrics': metrics_at_click[best_idx],
            'final_mask': curr_mask
        }
    
    def evaluate_seem(
        self,
        img_t: torch.Tensor,
        gt: np.ndarray,
        init_mask: np.ndarray,
        oh: int, ow: int, th: int, tw: int,
        seed: int
    ) -> Dict[str, Any]:
        """Run SEEM baseline evaluation on a single sample."""
        curr_mask = init_mask.copy()
        pts, lbls = [], []
        
        metrics_at_click = [calculate_metrics(curr_mask, gt)]
        
        for click_num in range(1, self.config.num_clicks + 1):
            click = get_random_click(curr_mask, gt, pts, click_num, seed)
            
            if click is None:
                metrics_at_click.append(metrics_at_click[-1].copy())
                continue
            
            pts.append(click['pt'])
            lbls.append(click['lbl'])
            
            pts_np = np.array(pts, dtype=float)
            pts_np[:, 0] *= tw / ow
            pts_np[:, 1] *= th / oh
            
            masks = inference_with_points(
                self.model, img_t,
                torch.from_numpy(pts_np[None]).cuda().float(),
                torch.from_numpy(np.array(lbls)[None]).cuda().float(),
                oh, ow, th, tw
            )
            
            if masks is not None:
                new_mask, _ = select_mask_random(masks, click_num, seed)
                if new_mask is not None:
                    curr_mask = apply_mask_update(new_mask)
                    if curr_mask is None:
                        curr_mask = init_mask.copy()
            
            metrics_at_click.append(calculate_metrics(curr_mask, gt))
        
        best_idx = get_best_click_index_with_threshold(metrics_at_click, 'Dice', threshold=0.0005)
        
        return {
            'metrics_at_click': metrics_at_click,
            'best_click': best_idx,
            'best_metrics': metrics_at_click[best_idx],
            'final_mask': curr_mask,
            'seed': seed
        }


def compare_strategies(
    model: Any,
    sample_ids: List[str],
    data_dir: str,
    gt_dir: str,
    output_dir: str,
    config: EvaluationConfig = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Compare GIIR and SEEM strategies on the same samples.
    """
    from ..visualization.plotting import plot_comparison_results
    
    os.makedirs(output_dir, exist_ok=True)
    
    if config is None:
        config = EvaluationConfig()
    
    evaluator = SEEMOCTAEvaluator(model, config)
    
    all_results = []
    
    for sample_id in tqdm(sample_ids, desc="Comparing Strategies"):
        img_3ch, viz_pil = load_3channel_image(sample_id, data_dir)
        gt_path = os.path.join(gt_dir, sample_id)
        gt = ((np.array(Image.open(gt_path).convert("L")) > 0) * 255).astype(np.uint8)
        
        init_mask, img_t, th, tw = get_initial_mask(model, img_3ch)
        oh, ow = gt.shape
        init_metrics = calculate_metrics(init_mask, gt)
        
        # Run GIIR
        giir_start = time.time()
        giir_result = evaluator.evaluate_giir(img_t, gt, init_mask, oh, ow, th, tw, debug=debug)
        giir_time = time.time() - giir_start
        
        # Run SEEM with multiple seeds
        seem_start = time.time()
        seem_seed_results = []
        for seed in config.seeds:
            seed_result = evaluator.evaluate_seem(img_t, gt, init_mask, oh, ow, th, tw, seed)
            seem_seed_results.append(seed_result)
        seem_time = time.time() - seem_start
        
        # Aggregate SEEM
        seem_aggregated = {
            'best_click': np.mean([sr['best_click'] for sr in seem_seed_results]),
            'best_metrics': {
                m: np.mean([sr['best_metrics'][m] for sr in seem_seed_results])
                for m in ['Dice', 'IoU', 'clDice', 'SE', 'SP']
            },
            'metrics_at_click': []
        }
        
        for click_idx in range(config.num_clicks + 1):
            click_metrics = {}
            for m in ['Dice', 'IoU', 'clDice', 'SE', 'SP']:
                vals = [sr['metrics_at_click'][click_idx][m] for sr in seem_seed_results]
                click_metrics[m] = np.mean(vals)
            seem_aggregated['metrics_at_click'].append(click_metrics)
        
        all_results.append({
            'sample_id': sample_id,
            'initial': init_metrics,
            'giir': giir_result,
            'seem': seem_aggregated,
            'seem_seed_results': seem_seed_results,
            'giir_time': giir_time,
            'seem_time': seem_time
        })
    
    # Generate plots
    plot_comparison_results(all_results, output_dir, config.num_clicks)
    
    # Save JSON
    json_results = {'config': {'num_samples': len(sample_ids)}, 'samples': []}
    for r in all_results:
        json_results['samples'].append({
            'sample_id': r['sample_id'],
            'initial': r['initial'],
            'giir_dice': r['giir']['best_metrics']['Dice'],
            'seem_dice': r['seem']['best_metrics']['Dice']
        })
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    giir_dices = [r['giir']['best_metrics']['Dice'] for r in all_results]
    seem_dices = [r['seem']['best_metrics']['Dice'] for r in all_results]
    
    print("\n" + "=" * 80)
    print(" COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  GIIR Dice: {np.mean(giir_dices):.4f} ± {np.std(giir_dices):.4f}")
    print(f"  SEEM Dice: {np.mean(seem_dices):.4f} ± {np.std(seem_dices):.4f}")
    print(f"  Improvement: +{np.mean(giir_dices) - np.mean(seem_dices):.4f}")
    print("=" * 80)
    
    return {'results': all_results}


def run_giir_evaluation(evaluator, sample_ids, data_dir, gt_dir, debug=False):
    """Run GIIR evaluation on multiple samples."""
    results = []
    for sample_id in tqdm(sample_ids, desc="GIIR Evaluation"):
        img_3ch, _ = load_3channel_image(sample_id, data_dir)
        gt = ((np.array(Image.open(os.path.join(gt_dir, sample_id)).convert("L")) > 0) * 255).astype(np.uint8)
        init_mask, img_t, th, tw = get_initial_mask(evaluator.model, img_3ch)
        result = evaluator.evaluate_giir(img_t, gt, init_mask, gt.shape[0], gt.shape[1], th, tw, debug)
        result['sample_id'] = sample_id
        results.append(result)
    return results


def run_seem_evaluation(evaluator, sample_ids, data_dir, gt_dir, seeds=None):
    """Run SEEM evaluation on multiple samples."""
    if seeds is None:
        seeds = evaluator.config.seeds
    results = []
    for sample_id in tqdm(sample_ids, desc="SEEM Evaluation"):
        img_3ch, _ = load_3channel_image(sample_id, data_dir)
        gt = ((np.array(Image.open(os.path.join(gt_dir, sample_id)).convert("L")) > 0) * 255).astype(np.uint8)
        init_mask, img_t, th, tw = get_initial_mask(evaluator.model, img_3ch)
        seed_results = [evaluator.evaluate_seem(img_t, gt, init_mask, gt.shape[0], gt.shape[1], th, tw, s) for s in seeds]
        results.append({'sample_id': sample_id, 'seed_results': seed_results})
    return results
