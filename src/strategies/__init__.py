"""Click selection strategies for interactive segmentation."""

from .base import ClickStrategy
from .giir import GIIRClickStrategy, get_giir_click, get_giir_click_train_robust
from .random_clicks import RandomClickStrategy, get_random_click

__all__ = [
    "ClickStrategy",
    "GIIRClickStrategy",
    "get_giir_click",
    "get_giir_click_train_robust",
    "RandomClickStrategy",
    "get_random_click",
]
