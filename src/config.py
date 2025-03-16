import numpy as np

# Define states
STATES = ["Clean", "Acquisition", "Infected", "Fraud"]

# Transition probability matrix
TRANSITION_MATRIX = np.array([
    [0.00, 1.00, 0.00, 0.00],  # Clean -> Other states
    [0.40, 0.00, 0.60, 0.00],  # Acquisition -> Other states
    [0.10, 0.60, 0.00, 0.30],  # Infected -> Other states
    [0.00, 0.00, 0.25, 0.75]   # Fraud -> Other states
])

# Weibull parameters (shape, scale)
SOJOURN_PARAMS = {
    "Clean": (2.0, 10.0),
    "Acquisition": (1.5, 5.0),
    "Infected": (1.20, 3.0),
    "Fraud": (1.1, 2.0)
}
