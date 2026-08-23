"""
Data preprocessing utilities for GS-RVFL experiments.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Data preprocessing for GS-RVFL experiments.
    """
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        random_state: Optional[int] = None
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        scaling_method : str, default='standard'
            Scaling method: 'standard', 'minmax', or 'robust'.
        random_state : int, optional
            Random seed.
        """
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.scaler = None
        
        self._scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """Fit preprocessor to data."""
        X = np.asarray(X)
        
        if self.scaling_method in self._scalers:
            self.scaler = self._scalers[self.scaling_method]()
            self.scaler.fit(X)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        X = np.asarray(X)
        
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target labels.
        test_size : float, default=0.2
            Test set proportion.
        stratify : bool, default=True
            Whether to stratify by class labels.
        
        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
            Split data.
        """
        stratify_labels = y if stratify and len(np.unique(y)) > 1 else None
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
    
    def create_time_series_features(
        self,
        X: np.ndarray,
        window_size: int = 1,
        include_derivatives: bool = True
    ) -> np.ndarray:
        """
        Create time series features for sequential data.
        
        Parameters
        ----------
        X : np.ndarray
            Time series data of shape (n_samples, n_features).
        window_size : int, default=1
            Window size for feature creation.
        include_derivatives : bool, default=True
            Whether to include derivative features.
        
        Returns
        -------
        X_features : np.ndarray
            Feature matrix with time series features.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Rolling window features
        features = []
        for i in range(window_size, n_samples):
            window = X[i - window_size:i]
            features.append(window.flatten())
        
        if include_derivatives:
            # Add velocity and acceleration
            for i in range(window_size + 1, n_samples):
                velocity = X[i] - X[i - 1]
                acceleration = velocity - (X[i - 1] - X[i - 2])
                features.append(np.concatenate([velocity, acceleration]))
        
        return np.array(features) if features else X


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Convenience function for preprocessing and splitting data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    test_size : float, default=0.2
        Test set proportion.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split and scaled data.
    scaler : StandardScaler
        Fitted scaler.
    """
    preprocessor = DataPreprocessor(random_state=random_state)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=test_size)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler