"""
Core implementation of GS-RVFL (Generalized Structured Random Vector Functional Link Network).
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import expit, softmax

from .base import BaseRVFLClassifier, BaseRVFLRegressor
from .operators import (
    StructuredDirectLink,
    AdaptiveBias,
    VelocityOperator,
    DisplacementOperator,
    JointVelocityOperator,
    JointDisplacementOperator,
    OperatorPipeline,
    BaseOperator
)


class GSRVFLClassifier(BaseRVFLClassifier):
    """
    Generalized Structured Random Vector Functional Link Network Classifier.
    
    This classifier extends RVFL by partitioning deterministic inputs into
    structured direct-link operators and adaptive bias operators.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of random enhancement nodes.
    scale : float, default=1.0
        Scale factor for random weight initialization.
    lambda_reg : float, default=1e-4
        Regularization parameter for ridge regression.
    activation : str, default='sigmoid'
        Activation function for hidden nodes.
    random_state : int, optional
        Random seed for reproducibility.
    use_bias : bool, default=True
        Whether to include bias term in design matrix.
    use_logistic : bool, default=False
        Whether to apply logistic activation for binary classification.
    g_operators : list, optional
        List of structured direct-link operators.
    b_operators : list, optional
        List of adaptive bias operators.
    """
    
    def __init__(
        self,
        n_hidden: int = 100,
        scale: float = 1.0,
        lambda_reg: float = 1e-4,
        activation: str = 'sigmoid',
        random_state: Optional[int] = None,
        use_bias: bool = True,
        use_logistic: bool = False,
        g_operators: Optional[List[BaseOperator]] = None,
        b_operators: Optional[List[BaseOperator]] = None
    ):
        super().__init__(
            n_hidden=n_hidden,
            scale=scale,
            lambda_reg=lambda_reg,
            activation=activation,
            random_state=random_state,
            use_bias=use_bias,
            use_logistic=use_logistic
        )
        self.g_operators = g_operators or []
        self.b_operators = b_operators or []
        
        self._g_pipeline = None
        self._b_pipeline = None
        self._n_g_features = 0
        self._n_b_features = 0
        
    def _build_operator_pipelines(self):
        """Build operator pipelines from operator lists."""
        if self.g_operators:
            self._g_pipeline = OperatorPipeline(self.g_operators)
        else:
            self._g_pipeline = None
            
        if self.b_operators:
            self._b_pipeline = OperatorPipeline(self.b_operators)
        else:
            self._b_pipeline = None
    
    def _compute_structured_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute structured direct-link and adaptive bias features."""
        G = None
        B = None
        
        if self._g_pipeline is not None:
            G = self._g_pipeline.transform(X)
            self._n_g_features = G.shape[1]
        else:
            self._n_g_features = 0
            
        if self._b_pipeline is not None:
            B = self._b_pipeline.transform(X)
            self._n_b_features = B.shape[1]
        else:
            self._n_b_features = 0
            
        return G, B
    
    def _compute_design_matrix(self, H: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Construct GS-RVFL design matrix [H, G(X), B(X), 1].
        """
        G, B = self._compute_structured_features(X)
        
        components = [H]
        
        if G is not None:
            components.append(G)
        elif self._g_pipeline is None:
            # If no G operators, use X as direct links (standard RVFL behavior)
            components.append(X)
            
        if B is not None:
            components.append(B)
            
        if self.use_bias:
            components.append(np.ones((X.shape[0], 1)))
        
        return np.hstack(components)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'GSRVFLClassifier':
        """Fit the GS-RVFL classifier."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Store classes
        self._classes = np.unique(Y)
        self._n_classes = len(self._classes)
        
        # Encode labels if needed
        if Y.dtype.kind not in 'iu':
            self._label_encoder = LabelEncoder()
            Y_encoded = self._label_encoder.fit_transform(Y)
        else:
            self._label_encoder = None
            Y_encoded = Y.astype(int)
        
        # One-hot encode targets
        if self._n_classes > 2:
            Y_target = np.eye(self._n_classes)[Y_encoded]
        else:
            Y_target = Y_encoded.reshape(-1, 1).astype(float)
            if self.use_logistic:
                Y_target = Y_target.astype(float)
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build operator pipelines
        self._build_operator_pipelines()
        
        # Fit operators
        if self._g_pipeline is not None:
            self._g_pipeline.fit(X_scaled, Y)
        if self._b_pipeline is not None:
            self._b_pipeline.fit(X_scaled, Y)
        
        # Initialize random weights
        n_features = X_scaled.shape[1]
        self.W, self.b = self._initialize_random_weights(n_features)
        
        # Compute hidden layer
        H = self._compute_hidden(X_scaled)
        
        # Construct design matrix
        D = self._compute_design_matrix(H, X_scaled)
        
        # Solve ridge regression
        self.beta = self._ridge_solution(D, Y_target)
        
        # Store operator dimensions for prediction
        self._n_g_features = sum(op.n_features_out for op in self.g_operators) if self.g_operators else 0
        self._n_b_features = sum(op.n_features_out for op in self.b_operators) if self.b_operators else 0
        
        self._fitted = True
        return self
    
    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute class scores."""
        X_scaled = self.scaler.transform(X)
        H = self._compute_hidden(X_scaled)
        D = self._compute_design_matrix(H, X_scaled)
        scores = D @ self.beta
        
        if self.use_logistic and self._n_classes == 2:
            return expit(scores)
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        scores = self._compute_scores(X)
        
        if self._n_classes == 2:
            if self.use_logistic:
                probs = np.column_stack([1 - scores.flatten(), scores.flatten()])
            else:
                probs = np.column_stack([1 - expit(scores.flatten()), expit(scores.flatten())])
            return probs
        else:
            return softmax(scores, axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        scores = self._compute_scores(X)
        
        if self._n_classes == 2:
            if self.use_logistic:
                preds = (scores >= 0.5).astype(int).flatten()
            else:
                preds = (scores >= 0).astype(int).flatten()
        else:
            preds = np.argmax(scores, axis=1)
        
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(preds)
        return preds
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'n_hidden': self.n_hidden,
            'scale': self.scale,
            'lambda_reg': self.lambda_reg,
            'activation': self.activation,
            'random_state': self.random_state,
            'use_bias': self.use_bias,
            'use_logistic': self.use_logistic,
            'g_operators': self.g_operators,
            'b_operators': self.b_operators
        }
        return params
    
    def set_params(self, **params) -> 'GSRVFLClassifier':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GSRVFLRegressor(BaseRVFLRegressor):
    """
    Generalized Structured Random Vector Functional Link Network Regressor.
    
    This regressor extends RVFL by partitioning deterministic inputs into
    structured direct-link operators and adaptive bias operators.
    """
    
    def __init__(
        self,
        n_hidden: int = 100,
        scale: float = 1.0,
        lambda_reg: float = 1e-4,
        activation: str = 'sigmoid',
        random_state: Optional[int] = None,
        use_bias: bool = True,
        g_operators: Optional[List[BaseOperator]] = None,
        b_operators: Optional[List[BaseOperator]] = None
    ):
        super().__init__(
            n_hidden=n_hidden,
            scale=scale,
            lambda_reg=lambda_reg,
            activation=activation,
            random_state=random_state,
            use_bias=use_bias
        )
        self.g_operators = g_operators or []
        self.b_operators = b_operators or []
        
        self._g_pipeline = None
        self._b_pipeline = None
        self._n_g_features = 0
        self._n_b_features = 0
        
    def _build_operator_pipelines(self):
        """Build operator pipelines from operator lists."""
        if self.g_operators:
            self._g_pipeline = OperatorPipeline(self.g_operators)
        else:
            self._g_pipeline = None
            
        if self.b_operators:
            self._b_pipeline = OperatorPipeline(self.b_operators)
        else:
            self._b_pipeline = None
    
    def _compute_structured_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute structured direct-link and adaptive bias features."""
        G = None
        B = None
        
        if self._g_pipeline is not None:
            G = self._g_pipeline.transform(X)
            self._n_g_features = G.shape[1]
        else:
            self._n_g_features = 0
            
        if self._b_pipeline is not None:
            B = self._b_pipeline.transform(X)
            self._n_b_features = B.shape[1]
        else:
            self._n_b_features = 0
            
        return G, B
    
    def _compute_design_matrix(self, H: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Construct GS-RVFL design matrix [H, G(X), B(X), 1]."""
        G, B = self._compute_structured_features(X)
        
        components = [H]
        
        if G is not None:
            components.append(G)
        elif self._g_pipeline is None:
            components.append(X)
            
        if B is not None:
            components.append(B)
            
        if self.use_bias:
            components.append(np.ones((X.shape[0], 1)))
        
        return np.hstack(components)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'GSRVFLRegressor':
        """Fit the GS-RVFL regressor."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build operator pipelines
        self._build_operator_pipelines()
        
        # Fit operators
        if self._g_pipeline is not None:
            self._g_pipeline.fit(X_scaled, Y)
        if self._b_pipeline is not None:
            self._b_pipeline.fit(X_scaled, Y)
        
        # Initialize random weights
        n_features = X_scaled.shape[1]
        self.W, self.b = self._initialize_random_weights(n_features)
        
        # Compute hidden layer
        H = self._compute_hidden(X_scaled)
        
        # Construct design matrix
        D = self._compute_design_matrix(H, X_scaled)
        
        # Solve ridge regression
        self.beta = self._ridge_solution(D, Y)
        
        self._n_g_features = sum(op.n_features_out for op in self.g_operators) if self.g_operators else 0
        self._n_b_features = sum(op.n_features_out for op in self.b_operators) if self.b_operators else 0
        
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
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'n_hidden': self.n_hidden,
            'scale': self.scale,
            'lambda_reg': self.lambda_reg,
            'activation': self.activation,
            'random_state': self.random_state,
            'use_bias': self.use_bias,
            'g_operators': self.g_operators,
            'b_operators': self.b_operators
        }
        return params
    
    def set_params(self, **params) -> 'GSRVFLRegressor':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Convenience aliases
GSRVFL = GSRVFLClassifier