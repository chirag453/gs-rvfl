"""
Utility functions for GS-RVFL.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.special import expit, softmax


def standardize(
    X: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Standardize data using mean and standard deviation.
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    mu : np.ndarray, optional
        Mean vector for standardization.
    sigma : np.ndarray, optional
        Standard deviation vector for standardization.
    return_params : bool, default=False
        Whether to return the mean and standard deviation.
    
    Returns
    -------
    X_norm : np.ndarray
        Standardized data.
    mu, sigma : np.ndarray
        Mean and standard deviation (if return_params=True).
    """
    X = np.asarray(X)
    
    if mu is None or sigma is None:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
    
    X_norm = (X - mu) / sigma
    
    if return_params:
        return X_norm, mu, sigma
    return X_norm


def compute_velocity(
    positions: np.ndarray,
    time_window: int = 1,
    smoothing: bool = True
) -> np.ndarray:
    """
    Compute velocity from position trajectories.
    
    Parameters
    ----------
    positions : np.ndarray
        Position data of shape (n_samples, n_features).
    time_window : int, default=1
        Time window for computing velocity.
    smoothing : bool, default=True
        Whether to apply smoothing to velocity estimates.
    
    Returns
    -------
    velocity : np.ndarray
        Velocity estimates of same shape as positions.
    """
    positions = np.asarray(positions)
    velocity = np.zeros_like(positions)
    
    # Compute finite difference
    for i in range(time_window, positions.shape[0]):
        velocity[i] = (positions[i] - positions[i - time_window]) / time_window
    
    # Forward difference for first few rows
    for i in range(min(time_window, positions.shape[0])):
        idx = min(i + time_window, positions.shape[0] - 1)
        velocity[i] = (positions[idx] - positions[i]) / max(1, time_window - i)
    
    if smoothing:
        window = 3
        kernel = np.ones(window) / window
        velocity = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=0,
            arr=velocity
        )
    
    return velocity


def compute_displacement(
    positions: np.ndarray,
    normalize: bool = True,
    time_window: int = 1
) -> np.ndarray:
    """
    Compute cumulative displacement from position trajectories.
    
    Parameters
    ----------
    positions : np.ndarray
        Position data of shape (n_samples, n_features).
    normalize : bool, default=True
        Whether to normalize by time window.
    time_window : int, default=1
        Time window for normalization.
    
    Returns
    -------
    displacement : np.ndarray
        Cumulative displacement of same shape as positions.
    """
    positions = np.asarray(positions)
    displacement = np.zeros_like(positions)
    
    for i in range(1, positions.shape[0]):
        displacement[i] = displacement[i - 1] + np.abs(positions[i] - positions[i - 1])
    
    if normalize:
        displacement = displacement / max(1, time_window)
    
    return displacement


def compute_acceleration(
    velocity: np.ndarray,
    time_window: int = 1,
    smoothing: bool = True
) -> np.ndarray:
    """
    Compute acceleration from velocity trajectories.
    
    Parameters
    ----------
    velocity : np.ndarray
        Velocity data of shape (n_samples, n_features).
    time_window : int, default=1
        Time window for computing acceleration.
    smoothing : bool, default=True
        Whether to apply smoothing to acceleration estimates.
    
    Returns
    -------
    acceleration : np.ndarray
        Acceleration estimates of same shape as velocity.
    """
    velocity = np.asarray(velocity)
    acceleration = np.zeros_like(velocity)
    
    for i in range(time_window, velocity.shape[0]):
        acceleration[i] = (velocity[i] - velocity[i - time_window]) / time_window
    
    for i in range(min(time_window, velocity.shape[0])):
        idx = min(i + time_window, velocity.shape[0] - 1)
        acceleration[i] = (velocity[idx] - velocity[i]) / max(1, time_window - i)
    
    if smoothing:
        window = 3
        kernel = np.ones(window) / window
        acceleration = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=0,
            arr=acceleration
        )
    
    return acceleration


def one_hot_encode(
    y: np.ndarray,
    n_classes: Optional[int] = None
) -> np.ndarray:
    """
    One-hot encode class labels.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels of shape (n_samples,).
    n_classes : int, optional
        Number of classes. If None, inferred from y.
    
    Returns
    -------
    y_one_hot : np.ndarray
        One-hot encoded labels of shape (n_samples, n_classes).
    """
    y = np.asarray(y)
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    y_one_hot = np.zeros((y.shape[0], n_classes))
    y_one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    
    return y_one_hot


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    return np.mean(y_true == y_pred)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Compute F1 score.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str, default='macro'
        Averaging strategy: 'macro', 'weighted', or 'micro'.
    
    Returns
    -------
    f1 : float
        F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    f1_scores = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        if average == 'weighted':
            weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        weights = np.array(weights) / np.sum(weights)
        return np.sum(np.array(f1_scores) * weights)
    else:  # micro
        tp_total = np.sum(y_true == y_pred)
        return tp_total / len(y_true)


def compute_rademacher_complexity(
    D: np.ndarray,
    beta: np.ndarray,
    n_samples: int
) -> float:
    """
    Compute empirical Rademacher complexity.
    
    Parameters
    ----------
    D : np.ndarray
        Design matrix of shape (n_samples, n_features).
    beta : np.ndarray
        Coefficient matrix.
    n_samples : int
        Number of samples.
    
    Returns
    -------
    complexity : float
        Empirical Rademacher complexity.
    """
    # Generate Rademacher random variables
    sigma = np.random.choice([-1, 1], size=n_samples)
    
    # Compute complexity
    complexity = np.mean(np.abs(sigma @ D @ beta))
    
    return complexity


def compute_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    
    # Confusion matrix
    confusion = np.zeros((len(classes), len(classes)))
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    # Observed agreement
    po = np.trace(confusion) / n
    
    # Expected agreement
    pe = 0
    for i in range(len(classes)):
        pe += (np.sum(confusion[i, :]) * np.sum(confusion[:, i])) / (n * n)
    
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    
    return kappa


def create_motion_operators(
    joint_indices: Optional[List[int]] = None,
    time_window: int = 1
) -> Tuple[List, List]:
    """
    Create motion operators for human action recognition.
    
    Parameters
    ----------
    joint_indices : list, optional
        Indices of joints to use for motion computation.
    time_window : int, default=1
        Time window for velocity and displacement computation.
    
    Returns
    -------
    g_operators : list
        Structured direct-link operators (velocity-based).
    b_operators : list
        Adaptive bias operators (displacement-based).
    """
    g_operators = [
        JointVelocityOperator(
            joint_indices=joint_indices,
            time_window=time_window,
            name="JointVelocity"
        )
    ]
    
    b_operators = [
        JointDisplacementOperator(
            joint_indices=joint_indices,
            time_window=time_window,
            normalize=True,
            name="JointDisplacement"
        )
    ]
    
    return g_operators, b_operators


def create_sensor_operators(
    time_window: int = 1
) -> Tuple[List, List]:
    """
    Create operators for sensor-based activity recognition.
    
    Parameters
    ----------
    time_window : int, default=1
        Time window for velocity and displacement computation.
    
    Returns
    -------
    g_operators : list
        Structured direct-link operators (velocity-based).
    b_operators : list
        Adaptive bias operators (displacement-based).
    """
    g_operators = [
        VelocityOperator(
            time_window=time_window,
            smoothing=True,
            name="SensorVelocity"
        )
    ]
    
    b_operators = [
        DisplacementOperator(
            time_window=time_window,
            normalize=True,
            name="SensorDisplacement"
        )
    ]
    
    return g_operators, b_operators