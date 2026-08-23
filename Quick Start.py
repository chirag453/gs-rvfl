from gs_rvfl import GSRVFLClassifier
from gs_rvfl.operators import VelocityOperator, DisplacementOperator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define operators
g_operators = [VelocityOperator()]
b_operators = [DisplacementOperator()]

# Create and train GS-RVFL
model = GSRVFLClassifier(
    n_hidden=100,
    lambda_reg=1e-4,
    g_operators=g_operators,
    b_operators=b_operators
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)