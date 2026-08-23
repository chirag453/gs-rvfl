"""
Structured direct-link and adaptive bias operators for GS-RVFL.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin


class BaseOperator(ABC):
    """Abstract base class for GS-RVFL operators."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input data."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def n_features_out(self) -> int:
        """Number of output features."""
        return self._n_features_out if hasattr(self, '_n_features_out') else 0


class StructuredDirectLink(BaseOperator):
    """
    Structured direct-link operator that preserves semantically meaningful variables.
    
    This operator is used when a variable or deterministic transformation has
    direct semantic meaning and should be preserved in the prediction function.
    """
    
    def __init__(
        self,
        columns: Optional[List[int]] = None,
        name: str = "StructuredDirectLink",
        transform_fn: Optional[Callable] = None
    ):
        super().__init__(name)
        self.columns = columns
        self.transform_fn = transform_fn
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        if self.columns is not None:
            self._n_features_out = len(self.columns)
        else:
            self._n_features_out = X.shape[1]
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the operator."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        if self.columns is not None:
            X_selected = X[:, self.columns]
        else:
            X_selected = X
        
        if self.transform_fn is not None:
            return self.transform_fn(X_selected)
        return X_selected
    
    def __repr__(self):
        return f"StructuredDirectLink(name={self.name}, n_features={self._n_features_out})"


class AdaptiveBias(BaseOperator):
    """
    Adaptive bias operator that encodes contextual information.
    
    This operator is used when a variable is expected to shift the output surface
    based on context, operating conditions, or system state.
    """
    
    def __init__(
        self,
        columns: Optional[List[int]] = None,
        name: str = "AdaptiveBias",
        transform_fn: Optional[Callable] = None
    ):
        super().__init__(name)
        self.columns = columns
        self.transform_fn = transform_fn
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        if self.columns is not None:
            self._n_features_out = len(self.columns)
        else:
            self._n_features_out = X.shape[1]
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the operator."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        if self.columns is not None:
            X_selected = X[:, self.columns]
        else:
            X_selected = X
        
        if self.transform_fn is not None:
            return self.transform_fn(X_selected)
        return X_selected
    
    def __repr__(self):
        return f"AdaptiveBias(name={self.name}, n_features={self._n_features_out})"


class VelocityOperator(BaseOperator):
    """
    Velocity operator for motion analysis.
    
    Computes velocity from position trajectories over time.
    """
    
    def __init__(
        self,
        time_window: int = 1,
        smoothing: bool = True,
        name: str = "Velocity"
    ):
        super().__init__(name)
        self.time_window = time_window
        self.smoothing = smoothing
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        # Assume X is (n_samples, n_features) where features include positions
        # For simplicity, we use all features as positions
        self._n_features_out = X.shape[1]
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute velocity."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        # Compute finite difference for velocity
        velocity = np.zeros_like(X)
        for i in range(self.time_window, X.shape[0]):
            velocity[i] = (X[i] - X[i - self.time_window]) / self.time_window
        
        # First few rows: use forward difference
        for i in range(min(self.time_window, X.shape[0])):
            velocity[i] = (X[min(i + self.time_window, X.shape[0] - 1)] - X[i]) / max(1, self.time_window)
        
        if self.smoothing:
            # Simple moving average smoothing
            window = 3
            kernel = np.ones(window) / window
            velocity = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'),
                axis=0,
                arr=velocity
            )
        
        return velocity
    
    def __repr__(self):
        return f"VelocityOperator(window={self.time_window}, smoothing={self.smoothing})"


class DisplacementOperator(BaseOperator):
    """
    Displacement operator for motion analysis.
    
    Computes cumulative displacement from position trajectories.
    """
    
    def __init__(
        self,
        time_window: int = 1,
        normalize: bool = True,
        name: str = "Displacement"
    ):
        super().__init__(name)
        self.time_window = time_window
        self.normalize = normalize
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        self._n_features_out = X.shape[1]
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute cumulative displacement."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        # Compute cumulative displacement from starting position
        displacement = np.zeros_like(X)
        for i in range(1, X.shape[0]):
            displacement[i] = displacement[i - 1] + np.abs(X[i] - X[i - 1])
        
        if self.normalize:
            # Normalize by time window
            displacement = displacement / max(1, self.time_window)
        
        return displacement
    
    def __repr__(self):
        return f"DisplacementOperator(window={self.time_window}, normalize={self.normalize})"


class JointVelocityOperator(BaseOperator):
    """
    Joint velocity operator for skeleton-based action recognition.
    
    Computes velocity of individual joints over time.
    """
    
    def __init__(
        self,
        joint_indices: Optional[List[int]] = None,
        time_window: int = 1,
        name: str = "JointVelocity"
    ):
        super().__init__(name)
        self.joint_indices = joint_indices
        self.time_window = time_window
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        # Assume X is (n_samples, n_joints * n_coords)
        n_joints = X.shape[1] // 3 if X.shape[1] % 3 == 0 else X.shape[1]
        if self.joint_indices is None:
            self.joint_indices = list(range(n_joints))
        self._n_features_out = len(self.joint_indices) * 3  # x, y, z velocity
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute joint velocities."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        n_samples = X.shape[0]
        n_joints = len(self.joint_indices)
        velocity = np.zeros((n_samples, n_joints * 3))
        
        # Reshape if needed (flattened joints)
        if X.shape[1] != n_joints * 3:
            # Try to infer structure
            joint_size = X.shape[1] // len(self.joint_indices)
            for i, idx in enumerate(self.joint_indices):
                start = idx * joint_size
                end = start + min(3, joint_size)
                # Compute velocity for each coordinate
                for c in range(end - start):
                    velocity[:, i * 3 + c] = np.gradient(X[:, start + c])
        else:
            for i, idx in enumerate(self.joint_indices):
                for c in range(3):
                    velocity[:, i * 3 + c] = np.gradient(X[:, idx * 3 + c])
        
        return velocity
    
    def __repr__(self):
        return f"JointVelocityOperator(joints={self.joint_indices}, window={self.time_window})"


class JointDisplacementOperator(BaseOperator):
    """
    Joint displacement operator for skeleton-based action recognition.
    
    Computes cumulative displacement of individual joints.
    """
    
    def __init__(
        self,
        joint_indices: Optional[List[int]] = None,
        time_window: int = 1,
        normalize: bool = True,
        name: str = "JointDisplacement"
    ):
        super().__init__(name)
        self.joint_indices = joint_indices
        self.time_window = time_window
        self.normalize = normalize
        self._n_features_out = 0
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the operator."""
        X = np.asarray(X)
        n_joints = X.shape[1] // 3 if X.shape[1] % 3 == 0 else X.shape[1]
        if self.joint_indices is None:
            self.joint_indices = list(range(n_joints))
        self._n_features_out = len(self.joint_indices)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute joint displacements."""
        X = np.asarray(X)
        if not self._fitted:
            raise RuntimeError("Operator must be fitted before transformation")
        
        n_samples = X.shape[0]
        n_joints = len(self.joint_indices)
        displacement = np.zeros((n_samples, n_joints))
        
        for i, idx in enumerate(self.joint_indices):
            # Compute cumulative Euclidean displacement
            if X.shape[1] >= idx * 3 + 3:
                pos = X[:, idx * 3:(idx * 3 + 3)]
                diff = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
                displacement[1:, i] = np.cumsum(diff)
            else:
                # Fallback: use all features for this joint
                pos = X[:, idx * 3:idx * 3 + min(3, X.shape[1] - idx * 3)]
                diff = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
                displacement[1:, i] = np.cumsum(diff)
        
        if self.normalize:
            displacement = displacement / max(1, self.time_window)
        
        return displacement
    
    def __repr__(self):
        return f"JointDisplacementOperator(joints={self.joint_indices}, normalize={self.normalize})"


class OperatorPipeline:
    """Pipeline for applying multiple operators sequentially."""
    
    def __init__(self, operators: List[BaseOperator]):
        self.operators = operators
        self._fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all operators."""
        for op in self.operators:
            op.fit(X, y)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all operators and concatenate results."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transformation")
        
        results = []
        for op in self.operators:
            result = op.transform(X)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)
        
        return np.hstack(results) if results else X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def n_features_out(self) -> int:
        """Total number of output features."""
        return sum(op.n_features_out for op in self.operators)