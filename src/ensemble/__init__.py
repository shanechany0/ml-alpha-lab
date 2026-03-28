"""Ensemble methods package for the ML Alpha Lab trading system."""

from src.ensemble.stacking import StackingEnsemble
from src.ensemble.weighted_ensemble import WeightedEnsemble

__all__ = [
    "StackingEnsemble",
    "WeightedEnsemble",
]
