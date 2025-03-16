import numpy as np
from config import SOJOURN_PARAMS

def generate_sojourn_time(state):
    """Generates the time spent in a state using a Weibull distribution."""
    shape, scale = SOJOURN_PARAMS[state]
    return np.random.weibull(shape) * scale
