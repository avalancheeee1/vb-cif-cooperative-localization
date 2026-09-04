"""
M1 -- Inter-vehicle range-bearing measurement model (paper Sec. II-B).

For a receiver ``i`` interrogating neighbor ``j``:

    d     = || p_i - p_j ||                       [m]
    theta = atan2(y_j - y_i, x_j - x_i) - psi_i   [rad]

with ``p = [x, y]`` the 2-D position and ``psi`` the heading.  Both the
prediction function and its analytical Jacobian ``H = dh / dx_i`` (with the
neighbor state treated as a fixed reference) are provided; ``draw()`` adds a
zero-mean range/bearing noise sample plus an optional multipath outlier to
the range, drawing from a caller-supplied ``rng`` so the realization is
shared across methods (strictly paired protocol).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_MEAS = 2


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def predicted_measurement(x_i: NDArray, x_j: NDArray) -> NDArray:
    """Deterministic range-bearing prediction h(x_i, x_j) -> [d, theta]."""
    dx = x_j[0] - x_i[0]
    dy = x_j[1] - x_i[1]
    d = float(np.hypot(dx, dy))
    # theta is left *unwrapped* (in [-2pi, 2pi]): wrapping it here would make
    # the CKF/VB cubature mean of the bearing catastrophically wrong whenever
    # the cubature points straddle +/-pi.  The bearing *innovation* is wrapped
    # separately (wrap_innovation), which is the correct place to handle the
    # circular boundary.  This matches the legacy channel model.
    theta = np.arctan2(dy, dx) - x_i[2]
    return np.array([d, theta])


def measurement_jacobian(x_i: NDArray, x_j: NDArray) -> NDArray:
    """Jacobian H = dh/dx_i (2 x 6), neighbor state treated as fixed."""
    dx = x_i[0] - x_j[0]      # receiver - neighbor
    dy = x_i[1] - x_j[1]
    d2 = dx ** 2 + dy ** 2
    d = np.sqrt(d2)
    if d < 1e-9:
        d = 1e-9
        d2 = 1e-18

    H = np.zeros((N_MEAS, 6))
    H[0, 0] = dx / d
    H[0, 1] = dy / d
    H[1, 0] = -dy / d2
    H[1, 1] = dx / d2
    H[1, 2] = -1.0
    return H


def wrap_innovation(innovation: NDArray) -> NDArray:
    """Wrap the bearing component of a range-bearing innovation to (-pi, pi]."""
    out = innovation.copy()
    out[1] = _wrap(out[1])
    return out


def draw_measurement(x_i: NDArray, x_j: NDArray, rng: np.random.Generator,
                     sigma_d: float, sigma_theta: float,
                     is_outlier: bool, outlier_mag: float
                     ) -> NDArray:
    """Draw one noisy range-bearing measurement ``z = [d, theta]``.

    ``sigma_d``/``sigma_theta`` already include the sea-state scaling;
    ``outlier_mag`` (3--12 m) is added to the range only when
    ``is_outlier`` is True.
    """
    z = predicted_measurement(x_i, x_j)
    z[0] += rng.normal(0.0, sigma_d)
    z[1] = z[1] + rng.normal(0.0, sigma_theta)
    if is_outlier:
        z[0] += outlier_mag
    return z
