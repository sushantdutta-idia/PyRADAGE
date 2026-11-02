"""PyRADAGE scientific implementation; legacy import path remains supported."""

from .fitting import FitBounds, FitResult, fit_classic, fit_joint
from .model import (
    Geometry,
    KDAForwardModel,
    ModelAssumptions,
    Observation,
    SpectrumPoint,
)
from .remnant import (
    KDAExtForwardModel,
    RemnantDynamicsState,
    RemnantFitBounds,
    fit_remnant_joint,
)
from .restarted import RestartedFitResult, fit_restarted
from .radio_classes import RADIO_CLASSES, normalize_radio_class
from .kinematics import compute_kinematic_age

__all__ = [
    "FitBounds",
    "FitResult",
    "Geometry",
    "KDAForwardModel",
    "KDAExtForwardModel",
    "ModelAssumptions",
    "Observation",
    "RemnantDynamicsState",
    "RemnantFitBounds",
    "RestartedFitResult",
    "SpectrumPoint",
    "fit_classic",
    "fit_joint",
    "fit_remnant_joint",
    "fit_restarted",
    "RADIO_CLASSES",
    "normalize_radio_class",
    "compute_kinematic_age",
]

__version__ = "0.9.0"
