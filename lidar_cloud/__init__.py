"""Cloud detection from observation-level lidar signals."""

__version__ = "1.0.0"

from .inference import CloudDetector, Prediction
from .weights import load_backbone, load_model

__all__ = ["CloudDetector", "Prediction", "load_backbone", "load_model"]
