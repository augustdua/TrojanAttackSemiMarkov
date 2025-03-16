import pytest
from src.sojourn import generate_sojourn_time

def test_sojourn_time():
    time = generate_sojourn_time("Infected")
    assert time > 0, "Sojourn time should be positive"
