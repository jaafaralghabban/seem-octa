"""
Training Configuration

This module defines configuration classes for training SEEM-OCTA models.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for SEEM-OCTA training.
    
    Attributes:
        # LoRA Parameters
        lora_rank: Rank for LoRA adaptation
        lora_alpha: Scaling factor for LoRA
        
        # Training Parameters
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        weight_decay: Weight decay for optimizer
        
        # Interactive Training
        max_clicks_train: Maximum clicks per sample during training
        strategy: Click selection strategy ('giir' or 'random')
        
        # Data Parameters
        image_size: Target image size (H, W)
        max_points: Maximum initial points
        dataset_type: 'OCTA_3mm' or 'OCTA_6mm'
        
        # Paths
        data_dir: Root data directory
        checkpoint_dir: Directory for saving checkpoints
        results_dir: Directory for results and plots
        config_path: Path to SEEM config YAML
        pretrained_path: Path to pretrained weights
        
        # System
        device: Training device
        num_workers: DataLoader workers
        mixed_precision: Use AMP
    """
    
    # LoRA Parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    
    # Training Parameters
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 2
    weight_decay: float = 0.05
    
    # Interactive Training
    max_clicks_train: int = 10
    strategy: str = 'giir'  # 'giir' or 'random'
    
    # Data Parameters
    image_size: tuple = (512, 512)
    max_points: int = 10
    dataset_type: str = 'OCTA_3mm'
    
    # Paths
    data_dir: str = 'octa'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    config_path: str = 'configs/seem/focall_unicl_lang_demo.yaml'
    pretrained_path: str = 'seem_focall_v0.pt'
    
    # System
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    num_workers: int = 0
    mixed_precision: bool = True
    
    # Scheduler
    scheduler_patience: int = 7
    scheduler_factor: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.strategy in ['giir', 'random'], f"Unknown strategy: {self.strategy}"
        assert self.lora_rank > 0, "LoRA rank must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
    
    @property
    def checkpoint_name(self) -> str:
        """Generate checkpoint filename based on config."""
        return (f"best_model_epochs{self.num_epochs}_"
                f"3channel_{self.strategy.upper()}_rank{self.lora_rank}.pth")
    
    @property
    def merged_checkpoint_name(self) -> str:
        """Generate merged checkpoint filename."""
        return (f"best_model_merged_epochs{self.num_epochs}_"
                f"3channel_{self.strategy.upper()}_rank{self.lora_rank}.pth")


@dataclass 
class EvaluationConfig:
    """
    Configuration for model evaluation.
    
    Attributes:
        num_samples: Number of samples to evaluate
        num_seeds: Number of random seeds for stochastic methods
        num_clicks: Maximum clicks per sample
        seeds: List of random seeds to use
    """
    
    num_samples: int = 20
    num_seeds: int = 5
    num_clicks: int = 10
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    
    # Thresholds
    thresholds_oracle: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.4, 0.5, 0.65]
    )
    thresholds_random: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7]
    )
    
    # Cleaning
    base_clean_size: int = 20
    patch_clean_size: int = 5
    local_update_radius: int = 50
