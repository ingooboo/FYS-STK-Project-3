"""Analytical reference solution utilities."""

import autograd.numpy as np

# Analytical reference solution of the 1D heat equation for comparison.
def g_analytic(point):
    """Compute analytical solution at a point (x, t).

    Args:
        point: Tuple/array with (x, t).

    Returns:
        Analytical solution value at (x, t).
    """
    x, t = point
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
