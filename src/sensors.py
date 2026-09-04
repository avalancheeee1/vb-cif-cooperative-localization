"""
M0 -- Onboard sensors: GNSS (anchors), DVL, and electronic compass.

Each vehicle's onboard measurement at an acoustic interrogation epoch is

    z_on = [x_g, y_g, psi, u, v]^T   (anchor,  dim 5)
    z_on = [psi, u, v]^T             (follower, dim 3)

with a known diagonal covariance R_on.  Per the strictly paired protocol, the
onboard noise realization is drawn once per vehicle per epoch and shared by
every method; this module therefore exposes a pure function that draws from a
caller-supplied ``rng`` rather than holding its own RNG state.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from params import SensorParams

# State index of each onboard quantity (matches x = [x, y, psi, u, v, r]).
_IX = {"x": 0, "y": 1, "psi": 2, "u": 3, "v": 4}


def onboard_measurement(x: NDArray, sensor: SensorParams,
                        rng: np.random.Generator, is_anchor: bool
                        ) -> tuple[NDArray, NDArray]:
    """Draw one onboard measurement for a vehicle at state ``x``.

    Returns ``(z_on, R_on)``.  Heading is wrapped to (-pi, pi].
    """
    # Single heading-noise draw (a compass reports *one* heading perturbation),
    # applied to the angle before wrapping -- not two independent draws in the
    # sin/cos branches (which would double the effective noise variance).
    n_psi = rng.normal(0.0, sensor.compass_sigma)
    z_psi = float(np.arctan2(np.sin(x[_IX["psi"]] + n_psi),
                             np.cos(x[_IX["psi"]] + n_psi)))

    z_u = float(x[_IX["u"]] + rng.normal(0.0, sensor.dvl_sigma_vel))
    z_v = float(x[_IX["v"]] + rng.normal(0.0, sensor.dvl_sigma_vel))

    if is_anchor:
        z_x = float(x[_IX["x"]] + rng.normal(0.0, sensor.gnss_sigma_pos))
        z_y = float(x[_IX["y"]] + rng.normal(0.0, sensor.gnss_sigma_pos))
        return np.array([z_x, z_y, z_psi, z_u, z_v]), sensor.R_on_anchor

    return np.array([z_psi, z_u, z_v]), sensor.R_on_follower
