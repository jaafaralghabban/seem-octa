"""
Base Click Strategy Interface

This module defines the abstract interface for click selection strategies
used in interactive segmentation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class ClickStrategy(ABC):
    """
    Abstract base class for click selection strategies.
    
    All click strategies should inherit from this class and implement
    the get_next_click method.
    """
    
    @abstractmethod
    def get_next_click(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        history_points: List[Tuple[int, int]],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Select the next click point based on current prediction and ground truth.
        
        Args:
            pred_mask: Current predicted mask (0-255 scale)
            gt_mask: Ground truth mask (0-255 scale)
            history_points: List of previously selected points [(x, y), ...]
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary with click information:
                {
                    'pt': (x, y),      # Click coordinates
                    'lbl': 0 or 1,     # 0=negative, 1=positive
                    'type': 'FN'/'FP'  # Error type
                }
            Or None if no click is needed (perfect prediction)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name for logging."""
        pass
    
    def reset(self):
        """Reset any internal state. Override if strategy has state."""
        pass


class DeterministicStrategy(ClickStrategy):
    """
    Base class for deterministic click strategies.
    
    Deterministic strategies always produce the same click given
    the same input (no randomness involved).
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def is_deterministic(self) -> bool:
        return True


class StochasticStrategy(ClickStrategy):
    """
    Base class for stochastic click strategies.
    
    Stochastic strategies involve randomness and may produce
    different clicks for the same input.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    @property
    def is_deterministic(self) -> bool:
        return False
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def reset(self):
        """Reset random state."""
        self._rng = np.random.RandomState(self.seed)
