#!/usr/bin/env python3
"""
SEEM-OCTA Strategy Comparison Script

Compare GIIR and SEEM baseline strategies on the same test samples.

Usage:
    python scripts/compare_strategies.py --weights checkpoints/best_model.pth
    python scripts/compare_strategies.py --weights model.pth --num-samples 50 --num-seeds 10
"""

import argparse
import os
import sys
from glob import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.model_utils import setup_model
from src.evaluation.evaluator import compare_strategies
from src.training.config import EvaluationConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GIIR and SEEM strategies"
    )
    
    # Model
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/seem/focall_unicl_lang_demo.yaml",
        help="Path to model config"
    )
    
    # Evaluation settings
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds for SEEM")
    parser.add_argument("--num-clicks", type=int, default=10, help="Max clicks per sample")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="octa", help="Data directory")
    parser.add_argument("--dataset", type=str, default="OCTA_3mm", help="Dataset type")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    
    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print(" SEEM-OCTA Strategy Comparison")
    print("=" * 70)
    print(f" Weights: {args.weights}")
    print(f" Samples: {args.num_samples}")
    print(f" Seeds: {args.num_seeds}")
    print(f" Clicks: {args.num_clicks}")
    print("=" * 70 + "\n")
    
    # Load model
    model = setup_model(args.weights, args.config)
    
    # Get sample IDs (last N samples from dataset)
    image_dir = os.path.join(args.data_dir, args.dataset, "OCTA(ILM_OPL)")
    all_files = sorted(glob(os.path.join(image_dir, "*.bmp")))
    sample_ids = [os.path.basename(f) for f in all_files[-args.num_samples:]]
    
    print(f"Found {len(sample_ids)} samples for evaluation")
    
    # Ground truth directory
    gt_dir = os.path.join(args.data_dir, "Label", "Label", "GT_LargeVessel")
    
    # Evaluation config
    config = EvaluationConfig(
        num_samples=args.num_samples,
        num_seeds=args.num_seeds,
        num_clicks=args.num_clicks,
        seeds=[42, 123, 456, 789, 1024][:args.num_seeds]
    )
    
    # Run comparison
    results = compare_strategies(
        model=model,
        sample_ids=sample_ids,
        data_dir=args.data_dir,
        gt_dir=gt_dir,
        output_dir=args.output_dir,
        config=config,
        debug=args.debug
    )
    
    print(f"\n✓ Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
