"""
Unit tests for GS-RVFL core implementation.
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from gs_rvfl import GSRVFLClassifier, GSRVFLRegressor
from gs_rvfl.operators import (
    VelocityOperator,
    DisplacementOperator,
    StructuredDirectLink,
    AdaptiveBias
)


class TestGSRVFL(unittest.TestCase):
    """Test cases for GS-RVFL core functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Classification data
        self.X_class, self.y_class = load_iris(return_X_y=True)
        self.X_train_c, self.X_test_c, self.y_train_c, self.y_test_c = train_test_split(
            self.X_class, self.y_class, test_size=0.3, random_state=42
        )
        
        # Regression data
        self.X_reg, self.y_reg = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        self.X_train_r, self.X_test_r, self.y_train_r, self.y_test_r = train_test_split(
            self.X_reg, self.y_reg, test_size=0.3, random_state=42
        )
    
    def test_classifier_basic(self):
        """Test basic classifier functionality."""
        model = GSRVFLClassifier(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42
        )
        model.fit(self.X_train_c, self.y_train_c)
        y_pred = model.predict(self.X_test_c)
        
        self.assertIsNotNone(model.W)
        self.assertIsNotNone(model.b)
        self.assertIsNotNone(model.beta)
        self.assertTrue(model._fitted)
        self.assertEqual(len(y_pred), len(self.y_test_c))
        self.assertGreater(accuracy_score(self.y_test_c, y_pred), 0.8)
    
    def test_classifier_with_operators(self):
        """Test classifier with structured operators."""
        g_ops = [StructuredDirectLink()]
        b_ops = [AdaptiveBias()]
        
        model = GSRVFLClassifier(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42,
            g_operators=g_ops,
            b_operators=b_ops
        )
        model.fit(self.X_train_c, self.y_train_c)
        y_pred = model.predict(self.X_test_c)
        
        self.assertTrue(model._use_structured)
        self.assertGreater(accuracy_score(self.y_test_c, y_pred), 0.8)
    
    def test_classifier_with_motion_operators(self):
        """Test classifier with motion operators."""
        g_ops = [VelocityOperator()]
        b_ops = [DisplacementOperator()]
        
        model = GSRVFLClassifier(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42,
            g_operators=g_ops,
            b_operators=b_ops
        )
        model.fit(self.X_train_c, self.y_train_c)
        y_pred = model.predict(self.X_test_c)
        
        self.assertTrue(model._use_structured)
        self.assertEqual(model.n_g_features, self.X_train_c.shape[1])
        self.assertEqual(model.n_b_features, self.X_train_c.shape[1])
    
    def test_classifier_probabilities(self):
        """Test probability predictions."""
        model = GSRVFLClassifier(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42
        )
        model.fit(self.X_train_c, self.y_train_c)
        probs = model.predict_proba(self.X_test_c)
        
        self.assertEqual(probs.shape[0], len(self.X_test_c))
        self.assertEqual(probs.shape[1], len(np.unique(self.y_train_c)))
        np.testing.assert_almost_equal(np.sum(probs, axis=1), 1.0, decimal=6)
    
    def test_regressor_basic(self):
        """Test basic regressor functionality."""
        model = GSRVFLRegressor(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42
        )
        model.fit(self.X_train_r, self.y_train_r)
        y_pred = model.predict(self.X_test_r)
        
        self.assertIsNotNone(model.W)
        self.assertIsNotNone(model.b)
        self.assertIsNotNone(model.beta)
        self.assertTrue(model._fitted)
        self.assertEqual(len(y_pred), len(self.y_test_r))
        self.assertLess(mean_squared_error(self.y_test_r, y_pred), 1.0)
    
    def test_regressor_with_operators(self):
        """Test regressor with structured operators."""
        g_ops = [StructuredDirectLink()]
        b_ops = [AdaptiveBias()]
        
        model = GSRVFLRegressor(
            n_hidden=50,
            lambda_reg=1e-4,
            random_state=42,
            g_operators=g_ops,
            b_operators=b_ops
        )
        model.fit(self.X_train_r, self.y_train_r)
        y_pred = model.predict(self.X_test_r)
        
        self.assertTrue(model._use_structured)
        self.assertLess(mean_squared_error(self.y_test_r, y_pred), 1.0)
    
    def test_get_params(self):
        """Test parameter getter."""
        model = GSRVFLClassifier(
            n_hidden=100,
            lambda_reg=1e-3,
            random_state=42
        )
        params = model.get_params()
        
        self.assertEqual(params['n_hidden'], 100)
        self.assertEqual(params['lambda_reg'], 1e-3)
        self.assertEqual(params['random_state'], 42)
    
    def test_set_params(self):
        """Test parameter setter."""
        model = GSRVFLClassifier()
        model.set_params(n_hidden=200, lambda_reg=1e-2)
        
        self.assertEqual(model.n_hidden, 200)
        self.assertEqual(model.lambda_reg, 1e-2)
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['sigmoid', 'tanh', 'relu', 'sine', 'radbas']
        
        for activation in activations:
            model = GSRVFLClassifier(
                n_hidden=50,
                activation=activation,
                random_state=42
            )
            model.fit(self.X_train_c, self.y_train_c)
            y_pred = model.predict(self.X_test_c)
            
            self.assertGreater(accuracy_score(self.y_test_c, y_pred), 0.7)
    
    def test_fallback_to_rvfl(self):
        """Test fallback to standard RVFL."""
        model = GSRVFLClassifier(
            n_hidden=50,
            random_state=42,
            use_standard_rvfl_fallback=True
        )
        model.fit(self.X_train_c, self.y_train_c)
        
        self.assertFalse(model._use_structured)
        self.assertEqual(model.n_g_features, self.X_train_c.shape[1])
    
    def test_operator_feature_dimensions(self):
        """Test operator feature dimensions."""
        g_ops = [StructuredDirectLink()]
        b_ops = [AdaptiveBias()]
        
        model = GSRVFLClassifier(
            n_hidden=50,
            random_state=42,
            g_operators=g_ops,
            b_operators=b_ops
        )
        model.fit(self.X_train_c, self.y_train_c)
        
        self.assertEqual(model.n_g_features, self.X_train_c.shape[1])
        self.assertEqual(model.n_b_features, self.X_train_c.shape[1])
        self.assertEqual(
            model.n_total_features,
            model.n_hidden + model.n_g_features + model.n_b_features + 1
        )


class TestGSRVFLEdgeCases(unittest.TestCase):
    """Test edge cases for GS-RVFL."""
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.random.randn(1, 10)
        y = np.array([0])
        
        model = GSRVFLClassifier(n_hidden=10, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        self.assertEqual(len(y_pred), 1)
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = np.random.randint(0, 2, 100)
        
        model = GSRVFLClassifier(n_hidden=10, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        self.assertEqual(y_pred.shape[0], 100)
    
    def test_binary_classification_logistic(self):
        """Test binary classification with logistic output."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        
        model = GSRVFLClassifier(
            n_hidden=50,
            use_logistic=True,
            random_state=42
        )
        model.fit(X, y)
        probs = model.predict_proba(X)
        
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        self.assertEqual(probs.shape[1], 2)
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        X = np.random.randn(100, 500)
        y = np.random.randint(0, 2, 100)
        
        model = GSRVFLClassifier(n_hidden=50, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        self.assertEqual(y_pred.shape[0], 100)
    
    def test_no_operators(self):
        """Test with no operators (should fall back to RVFL)."""
        model = GSRVFLClassifier(
            n_hidden=50,
            g_operators=[],
            b_operators=[],
            use_standard_rvfl_fallback=True,
            random_state=42
        )
        model.fit(self.X_train_c, self.y_train_c)
        
        self.assertFalse(model._use_structured)


if __name__ == '__main__':
    unittest.main()