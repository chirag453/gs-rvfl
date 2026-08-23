"""
Generalized Structured Random Vector Functional Link Network (GS-RVFL)

A structured extension of the Random Vector Functional Link network that partitions
deterministic inputs into structured direct-link operators and adaptive bias operators.
"""

from .core import GSRVFL, GSRVFLClassifier, GSRVFLRegressor
from .operators import (
    StructuredDirectLink,
    AdaptiveBias,
    VelocityOperator,
    DisplacementOperator,
    JointVelocityOperator,
    JointDisplacementOperator
)
from .utils import (
    standardize,
    compute_velocity,
    compute_displacement,
    compute_acceleration,
    one_hot_encode
)
from .base import BaseRVFL

__version__ = "1.0.0"
__all__ = [
    'GSRVFL',
    'GSRVFLClassifier',
    'GSRVFLRegressor',
    'StructuredDirectLink',
    'AdaptiveBias',
    'VelocityOperator',
    'DisplacementOperator',
    'JointVelocityOperator',
    'JointDisplacementOperator',
    'standardize',
    'compute_velocity',
    'compute_displacement',
    'compute_acceleration',
    'one_hot_encode',
    'BaseRVFL'
]