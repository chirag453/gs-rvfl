"""
Base classes for RVFL and GS-RVFL implementations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from scipy.special import expit


class BaseRVFL(ABC):
    """Abstract base class for RVFL-based models."""
    
    def __init__(
        self,
        n_hidden: int = 100,
        scale: float = 1.0,
        lambda_reg: float = 1e-4,
        activation: str = 'sigmoid',
        random_state: Optional[int] = None,
        use_bias: bool = True
    ):
        self.n_hidden = n_hidden
        self.scale = scale
        self.lambda_reg = lambda_reg
        self.activation = activation
        self.random_state = random_state
        self.use_bias = use_bias
        
        self._activation_functions = {
            'sigmoid': expit,
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'sine': np.sin,
            'radbas': lambda x: np.exp(-(x ** 2)),
            'linear': lambda x: x,
            'softplus': lambda x: np.log(1 + np.exp(x)),
            'elu': lambda x: np.where(x > 0, x, 0.5 * (np.exp(x) - 1))
        }
        
        self._fitted = False
        self.W = None
        self.b = None
        self.beta = None
        self.scaler = None
        
    def _get_activation(self) -> Callable:
        """Get activation function by name."""
        if self.activation not in self._activation_functions:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self._activation_functions[self.activation]
    
    def _initialize_random_weights(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize random hidden layer weights and biases."""
        rng = np.random.RandomState(self.random_state)
        W = rng.uniform(-self.scale, self.scale, (n_features, self.n_hidden))
        b = rng.uniform(-self.scale, self.scale, (1, self.n_hidden))
        return W, b
    
    def _compute_hidden(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden layer activations."""
        H = self._get_activation()(X @ self.W + self.b)
        return H
    
    def _compute_design_matrix(self, H: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Construct design matrix [H, X, 1]."""
        if self.use_bias:
            return np.hstack([H, X, np.ones((X.shape[0], 1))])
        return np.hstack([H, X])
    
    def _ridge_solution(self, D: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Solve ridge regression problem."""
        if D.shape[0] >= D.shape[1]:
            # Primal solution
            return np.linalg.solve(
                D.T @ D + self.lambda_reg * np.eye(D.shape[1]),
                D.T @ Y
            )
        else:
            # Dual solution (more efficient when samples < features)
            return D.T @ np.linalg.solve(
                D @ D.T + self.lambda_reg * np.eye(D.shape[0]),
                Y
            )
    
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass


class BaseRVFLClassifier(BaseRVFL, ClassifierMixin):
    """Abstract base class for RVFL classifiers."""
    
    def __init__(
        self,
        n_hidden: int = 100,
        scale: float = 1.0,
        lambda_reg: float = 1e-4,
        activation: str = 'sigmoid',
        random_state: Optional[int] = None,
        use_bias: bool = True,
        use_logistic: bool = False
    ):
        super().__init__(
            n_hidden=n_hidden,
            scale=scale,
            lambda_reg=lambda_reg,
            activation=activation,
            random_state=random_state,
            use_bias=use_bias
        )
        self.use_logistic = use_logistic
        self._classes = None
        self._n_classes = None
    
    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute class scores."""
        H = self._compute_hidden(X)
        D = self._compute_design_matrix(H, X)
        scores = D @ self.beta
        
        if self.use_logistic and self._n_classes == 2:
            # Binary classification with logistic output
            return expit(scores)
        return scores
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the classifier."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Store classes
        self._classes = np.unique(Y)
        self._n_classes = len(self._classes)
        
        # One-hot encode targets if multi-class
        if self._n_classes > 2:
            Y_encoded = np.eye(self._n_classes)[Y.astype(int)]
        else:
            # Binary classification
            Y_encoded = Y.reshape(-1, 1)
            if self.use_logistic:
                Y_encoded = Y_encoded.astype(float)
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize random weights
        self.W, self.b = self._initialize_random_weights(X_scaled.shape[1])
        
        # Compute hidden layer
        H = self._compute_hidden(X_scaled)
        
        # Construct design matrix
        D = self._compute_design_matrix(H, X_scaled)
        
        # Solve ridge regression
        self.beta = self._ridge_solution(D, Y_encoded)
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        scores = self._compute_scores(X_scaled)
        
        if self._n_classes == 2:
            if self.use_logistic:
                return (scores >= 0.5).astype(int).flatten()
            else:
                return (scores >= 0).astype(int).flatten()
        else:
            return self._classes[np.argmax(scores, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        scores = self._compute_scores(X_scaled)
        
        if self._n_classes == 2:
            # Binary classification
            if self.use_logistic:
                probs = np.column_stack([1 - scores.flatten(), scores.flatten()])
            else:
                probs = np.column_stack([1 - expit(scores.flatten()), expit(scores.flatten())])
            return probs
        else:
            # Multi-class: softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class BaseRVFLRegressor(BaseRVFL, RegressorMixin):
    """Abstract base class for RVFL regressors."""
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the regressor."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize random weights
        self.W, self.b = self._initialize_random_weights(X_scaled.shape[1])
        
        # Compute hidden layer
        H = self._compute_hidden(X_scaled)
        
        # Construct design matrix
        D = self._compute_design_matrix(H, X_scaled)
        
        # Solve ridge regression
        self.beta = self._ridge_solution(D, Y)
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        
        H = self._compute_hidden(X_scaled)
        D = self._compute_design_matrix(H, X_scaled)
        predictions = D @ self.beta
        
        if predictions.shape[1] == 1:
            return predictions.flatten()
        return predictions