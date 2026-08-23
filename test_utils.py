"""
Unit tests for GS-RVFL utilities.
"""

import unittest
import numpy as np
from gs_rvfl.utils import (
    standardize,
    compute_velocity,
    compute_displacement,
    compute_acceleration,
    one_hot_encode,
    accuracy_score,
    f1_score,
    compute_rademacher_complexity,
    compute_cohen_kappa,
    create_train_test_split,
    compute_class_weights,
    normalize_data
)


class TestUtils(unittest.TestCase):
    """Test cases for GS-RVFL utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 3, 100)
        self.y_binary = np.random.randint(0, 2, 100)
    
    def test_standardize(self):
        """Test standardization function."""
        X_norm, mu, sigma = standardize(self.X, return_params=True)
        
        np.testing.assert_almost_equal(np.mean(X_norm, axis=0), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(X_norm, axis=0), 1, decimal=10)
        self.assertEqual(len(mu), self.X.shape[1])
        self.assertEqual(len(sigma), self.X.shape[1])
    
    def test_standardize_with_params(self):
        """Test standardization with provided parameters."""
        X_norm, mu, sigma = standardize(self.X, return_params=True)
        X_norm2 = standardize(self.X, mu=mu, sigma=sigma)
        
        np.testing.assert_almost_equal(X_norm, X_norm2)
    
    def test_compute_velocity(self):
        """Test velocity computation."""
        positions = np.cumsum(np.random.randn(100, 5), axis=0)
        velocity = compute_velocity(positions, time_window=1)
        
        self.assertEqual(velocity.shape, positions.shape)
        # Velocity should be finite
        self.assertTrue(np.all(np.isfinite(velocity)))
    
    def test_compute_velocity_with_smoothing(self):
        """Test velocity computation with smoothing."""
        positions = np.cumsum(np.random.randn(100, 5), axis=0)
        velocity_no_smooth = compute_velocity(positions, smoothing=False)
        velocity_smooth = compute_velocity(positions, smoothing=True)
        
        self.assertEqual(velocity_no_smooth.shape, velocity_smooth.shape)
        # Smoothing changes values (may be similar for small data)
        self.assertTrue(np.all(np.isfinite(velocity_smooth)))
    
    def test_compute_displacement(self):
        """Test displacement computation."""
        positions = np.cumsum(np.random.randn(100, 5), axis=0)
        displacement = compute_displacement(positions)
        
        self.assertEqual(displacement.shape, positions.shape)
        # Displacement should be non-negative
        self.assertTrue(np.all(displacement >= 0))
    
    def test_compute_displacement_no_normalize(self):
        """Test displacement computation without normalization."""
        positions = np.cumsum(np.random.randn(100, 5), axis=0)
        displacement = compute_displacement(positions, normalize=False)
        
        self.assertEqual(displacement.shape, positions.shape)
        self.assertTrue(np.all(displacement >= 0))
    
    def test_compute_acceleration(self):
        """Test acceleration computation."""
        positions = np.cumsum(np.random.randn(100, 5), axis=0)
        velocity = compute_velocity(positions)
        acceleration = compute_acceleration(velocity)
        
        self.assertEqual(acceleration.shape, velocity.shape)
        self.assertTrue(np.all(np.isfinite(acceleration)))
    
    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        y = np.array([0, 1, 2, 1, 0, 2])
        y_one_hot = one_hot_encode(y)
        
        self.assertEqual(y_one_hot.shape, (6, 3))
        np.testing.assert_almost_equal(np.sum(y_one_hot, axis=1), 1)
        self.assertEqual(np.argmax(y_one_hot, axis=1).tolist(), y.tolist())
    
    def test_one_hot_encode_with_classes(self):
        """Test one-hot encoding with specified classes."""
        y = np.array([0, 1, 2, 1, 0, 2])
        y_one_hot = one_hot_encode(y, n_classes=5)
        
        self.assertEqual(y_one_hot.shape, (6, 5))
        np.testing.assert_almost_equal(np.sum(y_one_hot, axis=1), 1)
    
    def test_accuracy_score(self):
        """Test accuracy score."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 4/6)  # 4 correct out of 6
    
    def test_f1_score_macro(self):
        """Test macro F1 score."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 0, 2])
        
        f1 = f1_score(y_true, y_pred, average='macro')
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1)
    
    def test_f1_score_weighted(self):
        """Test weighted F1 score."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 0, 2])
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1)
    
    def test_compute_rademacher_complexity(self):
        """Test Rademacher complexity computation."""
        D = np.random.randn(100, 20)
        beta = np.random.randn(20, 3)
        
        complexity = compute_rademacher_complexity(D, beta)
        self.assertGreater(complexity, 0)
    
    def test_compute_rademacher_complexity_with_samples(self):
        """Test Rademacher complexity with limited samples."""
        D = np.random.randn(100, 20)
        beta = np.random.randn(20, 3)
        
        complexity_full = compute_rademacher_complexity(D, beta)
        complexity_sample = compute_rademacher_complexity(D, beta, n_samples=50)
        
        self.assertGreater(complexity_full, 0)
        self.assertGreater(complexity_sample, 0)
    
    def test_compute_cohen_kappa(self):
        """Test Cohen's kappa."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0])
        
        kappa = compute_cohen_kappa(y_true, y_pred)
        self.assertGreater(kappa, 0)
        self.assertLessEqual(kappa, 1)
    
    def test_create_train_test_split(self):
        """Test train-test split."""
        X_train, X_test, y_train, y_test = create_train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
    
    def test_create_train_test_split_stratified(self):
        """Test stratified train-test split."""
        X_train, X_test, y_train, y_test = create_train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=True
        )
        
        # Check class distribution is preserved
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)
        
        # Classes should be present in both
        self.assertEqual(len(np.unique(y_train)), 3)
        self.assertEqual(len(np.unique(y_test)), 3)
    
    def test_compute_class_weights(self):
        """Test class weight computation."""
        weights = compute_class_weights(self.y)
        
        self.assertEqual(len(weights), 3)
        self.assertTrue(np.all(weights > 0))
        
        # Imbalanced case
        y_imbalanced = np.array([0] * 10 + [1] * 90)
        weights = compute_class_weights(y_imbalanced)
        self.assertGreater(weights[0], weights[1])  # Minority class gets higher weight
    
    def test_normalize_data_standard(self):
        """Test standard normalization."""
        X_norm = normalize_data(self.X, method='standard')
        
        np.testing.assert_almost_equal(np.mean(X_norm, axis=0), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(X_norm, axis=0), 1, decimal=10)
    
    def test_normalize_data_minmax(self):
        """Test min-max normalization."""
        X_norm = normalize_data(self.X, method='minmax')
        
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))
        
        # Check min and max
        for i in range(X_norm.shape[1]):
            self.assertAlmostEqual(np.min(X_norm[:, i]), 0, places=5)
            self.assertAlmostEqual(np.max(X_norm[:, i]), 1, places=5)
    
    def test_normalize_data_robust(self):
        """Test robust normalization."""
        X_norm = normalize_data(self.X, method='robust')
        
        # Should be centered around median
        medians = np.median(X_norm, axis=0)
        self.assertTrue(np.all(np.abs(medians) < 1e-10))
    
    def test_normalize_data_invalid_method(self):
        """Test invalid normalization method."""
        with self.assertRaises(ValueError):
            normalize_data(self.X, method='invalid')


if __name__ == '__main__':
    unittest.main()