"""
SEEM-OCTA: Geometry-Informed Interactive Refinement for OCTA Vessel Segmentation

This package provides tools for interactive vessel segmentation in OCTA images
using the SEEM foundation model with parameter-efficient LoRA adaptation.
"""

__version__ = "1.0.0"
__author__ = "Jaafar"

from . import core
from . import data
from . import strategies
from . import training
from . import evaluation
from . import visualization
