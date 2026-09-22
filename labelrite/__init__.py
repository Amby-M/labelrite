"""LabelRite — reference implementation.

Real-time detection of anomalous annotators in human labeling pipelines
via reference-distribution modeling.

Method paper:
    Majumdar, A. (2026). "LabelRite: Real-Time Detection of Anomalous
    Annotators in Human Labeling Pipelines via Reference-Distribution
    Modeling." Zenodo. https://doi.org/10.5281/zenodo.22887754
"""

from .actions import recommended_action
from .data import synthetic_workforce
from .detector import AnnotatorResult, LabelRiteDetector

__version__ = "0.1.0"

__all__ = [
    "LabelRiteDetector",
    "AnnotatorResult",
    "recommended_action",
    "synthetic_workforce",
]
