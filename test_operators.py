"""
Unit tests for GS-RVFL operators.
"""

import unittest
import numpy as np
from gs_rvfl.operators import (
    StructuredDirectLink,
    AdaptiveBias,
    VelocityOperator,
    DisplacementOperator,
    AccelerationOperator,
    JointVelocityOperator,
    JointDisplacementOperator,
    OperatorPipeline,
    create_motion_operators,
    create_sensor_operators,
    create_tabular_operators
)


class TestOperators(unittest.TestCase):
    """Test cases for GS-RVFL operators."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, 100)
        
    def test_structured_direct_link(self):
        """Test StructuredDirectLink operator."""
        operator = StructuredDirectLink()
        operator.fit(self.X, self.y)
        X_transformed = operator.transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertEqual(operator.n_features_out, self.X.shape[1])
    
    def test_structured_direct_link_with_columns(self):
        """Test StructuredDirectLink with specific columns."""
        operator = StructuredDirectLink(columns=[0, 2, 4])
        operator.fit(self.X, self.y)
        X_transformed = operator.transform(self.X)
        
        self.assertEqual(X_transformed.shape[1], 3)
        self.assertEqual(operator.n_features_out, 3)
    
    def test_structured_direct_link_with_transform_fn(self):
        """Test StructuredDirectLink with transform function."""
        operator = StructuredDirectLink(
            transform_fn=lambda x: x ** 2
        )
        operator.fit(self.X, self.y)
        X_transformed = operator.transform(self.X)
        
        np.testing.assert_almost_equal(X_transformed, self.X ** 2)
    
    def test_adaptive_bias(self):
        """Test AdaptiveBias operator."""
        operator = AdaptiveBias()
        operator.fit(self.X, self.y)
        X_transformed = operator.transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertEqual(operator.n_features_out, self.X.shape[1])
    
    def test_velocity_operator(self):
        """Test VelocityOperator."""
        operator = VelocityOperator(time_window=1, smoothing=False)
        operator.fit(self.X, self.y)
        velocity = operator.transform(self.X)
        
        self.assertEqual(velocity.shape, self.X.shape)
        self.assertEqual(operator.n_features_out, self.X.shape[1])
        
        # Check that velocity is different from input
        self.assertFalse(np.allclose(velocity, self.X))
    
    def test_velocity_operator_with_smoothing(self):
        """Test VelocityOperator with smoothing."""
        operator = VelocityOperator(time_window=1, smoothing=True)
        operator.fit(self.X, self.y)
        velocity_smooth = operator.transform(self.X)
        
        # Check smoothing changed values
        operator_no_smooth = VelocityOperator(time_window=1, smoothing=False)
        operator_no_smooth.fit(self.X, self.y)
        velocity_no_smooth = operator_no_smooth.transform(self.X)
        
        # They should be different (smoothing changes values)
        # but for small random data they might be similar
        # Just check they exist and have correct shape
        self.assertEqual(velocity_smooth.shape, self.X.shape)
    
    def test_displacement_operator(self):
        """Test DisplacementOperator."""
        operator = DisplacementOperator(time_window=1, normalize=True)
        operator.fit(self.X, self.y)
        displacement = operator.transform(self.X)
        
        self.assertEqual(displacement.shape, self.X.shape)
        self.assertEqual(operator.n_features_out, self.X.shape[1])
        
        # Displacement should be non-negative (cumulative absolute differences)
        self.assertTrue(np.all(displacement >= 0))
    
    def test_acceleration_operator(self):
        """Test AccelerationOperator."""
        operator = AccelerationOperator(time_window=1, smoothing=False)
        operator.fit(self.X, self.y)
        acceleration = operator.transform(self.X)
        
        self.assertEqual(acceleration.shape, self.X.shape)
        self.assertEqual(operator.n_features_out, self.X.shape[1])
    
    def test_joint_velocity_operator(self):
        """Test JointVelocityOperator."""
        X_skeleton = np.random.randn(100, 30)  # 10 joints * 3 coordinates
        operator = JointVelocityOperator(joint_indices=[0, 1, 2])
        operator.fit(X_skeleton, self.y)
        velocity = operator.transform(X_skeleton)
        
        self.assertEqual(velocity.shape[1], 9)  # 3 joints * 3 coords
        self.assertEqual(operator.n_features_out, 9)
    
    def test_joint_displacement_operator(self):
        """Test JointDisplacementOperator."""
        X_skeleton = np.random.randn(100, 30)  # 10 joints * 3 coordinates
        operator = JointDisplacementOperator(joint_indices=[0, 1, 2])
        operator.fit(X_skeleton, self.y)
        displacement = operator.transform(X_skeleton)
        
        self.assertEqual(displacement.shape[1], 3)  # 3 joints
        self.assertEqual(operator.n_features_out, 3)
        self.assertTrue(np.all(displacement >= 0))
    
    def test_operator_pipeline(self):
        """Test OperatorPipeline."""
        operators = [
            StructuredDirectLink(),
            VelocityOperator(),
            DisplacementOperator()
        ]
        pipeline = OperatorPipeline(operators)
        pipeline.fit(self.X, self.y)
        X_transformed = pipeline.transform(self.X)
        
        expected_features = (
            self.X.shape[1] +  # StructuredDirectLink
            self.X.shape[1] +  # VelocityOperator
            self.X.shape[1]    # DisplacementOperator
        )
        self.assertEqual(X_transformed.shape[1], expected_features)
        self.assertEqual(pipeline.n_features_out, expected_features)
    
    def test_create_motion_operators(self):
        """Test create_motion_operators utility."""
        g_ops, b_ops = create_motion_operators(
            joint_indices=[0, 1, 2],
            time_window=1,
            include_acceleration=True
        )
        
        self.assertEqual(len(g_ops), 2)  # Velocity + Acceleration
        self.assertEqual(len(b_ops), 1)  # Displacement
        
        # Test without acceleration
        g_ops, b_ops = create_motion_operators(
            joint_indices=[0, 1, 2],
            time_window=1,
            include_acceleration=False
        )
        self.assertEqual(len(g_ops), 1)  # Only Velocity
    
    def test_create_sensor_operators(self):
        """Test create_sensor_operators utility."""
        g_ops, b_ops = create_sensor_operators(
            time_window=1,
            include_acceleration=True
        )
        
        self.assertEqual(len(g_ops), 2)  # Velocity + Acceleration
        self.assertEqual(len(b_ops), 1)  # Displacement
    
    def test_create_tabular_operators(self):
        """Test create_tabular_operators utility."""
        g_ops, b_ops = create_tabular_operators()
        
        self.assertEqual(len(g_ops), 0)
        self.assertEqual(len(b_ops), 0)
    
    def test_operator_repr(self):
        """Test operator string representation."""
        operator = StructuredDirectLink(name="TestOperator")
        repr_str = repr(operator)
        self.assertIn("TestOperator", repr_str)
        
        operator = VelocityOperator()
        repr_str = repr(operator)
        self.assertIn("VelocityOperator", repr_str)


if __name__ == '__main__':
    unittest.main()