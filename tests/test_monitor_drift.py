import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monitor_drift import population_stability_index  # noqa: E402


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000)
    psi = population_stability_index(data, data)
    assert psi < 1e-6


def test_psi_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(loc=0, scale=1, size=1000)
    shifted = rng.normal(loc=5, scale=1, size=1000)
    psi = population_stability_index(reference, shifted)
    assert psi > 0.2
