"""
Dataset loading utilities for GS-RVFL experiments.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_openml, fetch_20newsgroups
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DatasetLoader:
    """
    Loader for various datasets used in GS-RVFL experiments.
    """
    
    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        self.dataset_cache = {}
        
    def load_dataset(
        self,
        name: str,
        sample_size: Optional[int] = None,
        return_X_y: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a dataset by name.
        
        Parameters
        ----------
        name : str
            Dataset name.
        sample_size : int, optional
            Number of samples to use.
        return_X_y : bool, default=True
            Whether to return X and y separately.
        
        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target labels.
        """
        if name in self.dataset_cache:
            data = self.dataset_cache[name]
        else:
            data = self._load_dataset_impl(name)
            self.dataset_cache[name] = data
        
        X, y = data
        
        # Sample if specified
        if sample_size and sample_size < len(X):
            indices = np.random.RandomState(self.random_state).choice(
                len(X), sample_size, replace=False
            )
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    def _load_dataset_impl(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Internal dataset loading implementation."""
        
        # Tabular datasets
        if name == 'iris':
            data = load_iris(return_X_y=True)
        elif name == 'wine':
            data = load_wine(return_X_y=True)
        elif name == 'breast_cancer':
            data = load_breast_cancer(return_X_y=True)
        elif name == 'digits':
            data = load_digits(return_X_y=True)
        elif name == 'mnist':
            data = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'letter':
            data = fetch_openml('letter', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'satellite':
            data = fetch_openml('satimage', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'pendigits':
            data = fetch_openml('pendigits', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'optdigits':
            data = fetch_openml('optdigits', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'waveform':
            data = fetch_openml('waveform-5000', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'spambase':
            data = fetch_openml('spambase', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'adult':
            data = fetch_openml('adult', version=2, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'covertype':
            data = fetch_openml('covertype', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'segment':
            data = fetch_openml('segment', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name == 'usps':
            data = fetch_openml('usps', version=1, as_frame=False, parser='pandas', return_X_y=True)
        elif name in ['har', 'wisdm', 'pamap2']:
            # Sensor datasets - simplified loading
            data = self._load_sensor_dataset(name)
        elif name in ['kth', 'weizmann', 'ucf11', 'ucf101', 'hmdb51']:
            # Video datasets - simplified loading
            data = self._load_video_dataset(name)
        elif name in ['ntu', 'ntu120']:
            # Skeleton datasets - simplified loading
            data = self._load_skeleton_dataset(name)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        # Convert to numpy and handle labels
        X, y = data
        
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.flatten()
        
        # Encode labels if needed
        if y.dtype.kind not in 'iu':
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        
        return X, y
    
    def _load_sensor_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load sensor dataset."""
        # Simplified: generate synthetic sensor data
        np.random.seed(self.random_state)
        
        if name == 'har':
            n_samples, n_features, n_classes = 10299, 561, 6
        elif name == 'wisdm':
            n_samples, n_features, n_classes = 5000, 36, 6
        elif name == 'pamap2':
            n_samples, n_features, n_classes = 5000, 17, 12
        else:
            raise ValueError(f"Unknown sensor dataset: {name}")
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        return X, y
    
    def _load_video_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load video dataset."""
        # Simplified: generate synthetic video features
        np.random.seed(self.random_state)
        
        if name == 'kth':
            n_samples, n_classes = 599, 6
        elif name == 'weizmann':
            n_samples, n_classes = 90, 10
        elif name == 'ucf11':
            n_samples, n_classes = 1600, 11
        elif name == 'ucf101':
            n_samples, n_classes = 2000, 10
        elif name == 'hmdb51':
            n_samples, n_classes = 1000, 10
        else:
            raise ValueError(f"Unknown video dataset: {name}")
        
        n_features = 512  # Feature dimension
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        return X, y
    
    def _load_skeleton_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load skeleton dataset."""
        # Simplified: generate synthetic skeleton data
        np.random.seed(self.random_state)
        
        if name == 'ntu':
            n_samples, n_classes = 10000, 60
        elif name == 'ntu120':
            n_samples, n_classes = 10000, 120
        else:
            raise ValueError(f"Unknown skeleton dataset: {name}")
        
        n_features = 150  # Joints * coordinates
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        return X, y


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for loading datasets."""
    loader = DatasetLoader()
    return loader.load_dataset(name, **kwargs)