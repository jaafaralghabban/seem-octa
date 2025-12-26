#!/usr/bin/env python3
"""
SEEM-OCTA Training Script

Unified training script for both GIIR and Random click strategies.

Usage:
    python scripts/train.py --strategy giir --epochs 50
    python scripts/train.py --strategy random --epochs 50 --batch-size 4
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.config import TrainingConfig
from src.training.trainer import SEEMOCTATrainer
from src.data.dataset import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEEM-OCTA model with interactive segmentation"
    )
    
    # Strategy
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="giir",
        choices=["giir", "random"],
        help="Click selection strategy (default: giir)"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    
    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="octa", help="Data directory")
    parser.add_argument("--dataset", type=str, default="OCTA_3mm", help="Dataset type")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    parser.add_argument("--max-clicks", type=int, default=10, help="Max clicks per sample")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--pretrained", type=str, default="seem_focall_v0.pt")
    parser.add_argument("--config", type=str, default="configs/seem/focall_unicl_lang_demo.yaml")
    
    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print(" SEEM-OCTA Training")
    print("=" * 70)
    print(f" Strategy: {args.strategy.upper()}")
    print(f" Epochs: {args.epochs}")
    print(f" Batch Size: {args.batch_size}")
    print(f" LoRA Rank: {args.lora_rank}")
    print("=" * 70 + "\n")
    
    # Create configuration
    config = TrainingConfig(
        # Strategy
        strategy=args.strategy,
        
        # Training
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_clicks_train=args.max_clicks,
        
        # LoRA
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        
        # Data
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        image_size=(args.image_size, args.image_size),
        
        # Paths
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        pretrained_path=args.pretrained,
        config_path=args.config,
        
        # System
        device=args.device,
        num_workers=args.num_workers,
        mixed_precision=not args.no_amp,
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        max_points=config.max_points,
        dataset_type=config.dataset_type,
        num_workers=config.num_workers,
    )
    
    # Create trainer
    trainer = SEEMOCTATrainer(config)
    
    # Train
    history = trainer.train(train_loader, val_loader, test_loader)
    
    print("\n✓ Training complete!")
    print(f"  Best checkpoint: {config.checkpoint_dir}/{config.checkpoint_name}")
    print(f"  Merged checkpoint: {config.checkpoint_dir}/{config.merged_checkpoint_name}")


if __name__ == "__main__":
    main()
