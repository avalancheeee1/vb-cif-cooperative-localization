"""
M6 -- Per-epoch shared draws (strictly paired protocol, Appendix B).

For each acoustic epoch the scenario draws, *once*, and hands to every method:

  (a) the channel realization: per-link fading, reception flag, outlier flag,
      noise standard deviations;
  (b) the measurement noise realization for every in-range directed link;
  (c) the onboard (compass/DVL/GNSS) noise realization for every vehicle.

The unmodeled prediction disturbance is drawn in ``dynamics.generate_trajectory``
(added to the truth), and the initial perturbation is drawn once per vehicle in
the runner -- both are shared across methods by construction.

The scheduling policy decides *which* links a follower interrogates; the
scenario supplies the full channel/noise realization for *all* candidate links
so that any policy sees the identical draws for the links it selects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from params import Params
from src import channel as ch
from src.measurement import predicted_measurement
from src.sensors import onboard_measurement


@dataclass(frozen=True)
class LinkDraw:
    """Shared per-link channel + measurement-noise realization."""
    i: int                       # receiver
    j: int                       # neighbor (transmitter)
    d: float                     # true range (m)
    snr: float                   # received SNR (dB, incl. fading)
    p_succ: float                # intrinsic packet-success prob (before PLR)
    sigma_d: float               # range noise std (m, incl. sea-state)
    sigma_theta: float           # bearing noise std (rad, incl. sea-state)
    received: bool               # packet reception flag
    is_outlier: bool             # multipath outlier flag
    eta_d: float                 # range noise draw (m)
    eta_theta: float             # bearing noise draw (rad)
    tof_jitter: float            # ToF jitter (s), for the CQM
    z: NDArray                   # measurement [d_meas, theta_meas]


@dataclass(frozen=True)
class OnboardDraw:
    """Shared per-vehicle onboard measurement + covariance."""
    z_on: NDArray
    R_on: NDArray
    is_anchor: bool


@dataclass(frozen=True)
class EpochDraw:
    """All shared draws for one epoch."""
    k: int
    wind: float
    links: tuple[LinkDraw, ...]
    onboard: tuple[OnboardDraw, ...]


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def draw_epoch(truth_k: NDArray, k: int, params: Params,
               rng: np.random.Generator, plr: float,
               anchors: frozenset[int]) -> EpochDraw:
    """Draw the shared realization for epoch ``k`` (``truth_k`` = (n_usvs, 6))."""
    chp = params.channel
    sen = params.sensors
    n = params.system.n_usvs
    t = k * params.system.acoustic_period * params.system.dt
    wind = chp.wind(t)

    links: list[LinkDraw] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.hypot(truth_k[i, 0] - truth_k[j, 0],
                               truth_k[i, 1] - truth_k[j, 1]))
            if d > chp.max_range:
                continue

            fading = rng.normal(0.0, chp.fading_std_db)
            snr = ch.snr_db(d, wind, fading, chp)
            sigma_d = ch.range_noise_std(snr, wind, chp)
            sigma_theta = ch.bearing_noise_std(snr, wind, chp)

            p_succ = ch.packet_success(snr, chp)
            received = bool(rng.random() < p_succ * (1.0 - plr))
            is_outlier = bool(rng.random() < ch.outlier_probability(snr, wind, chp))
            # Signed multipath outlier (matches legacy): the range is perturbed
            # by +/- U(3, 12) m, not always upward.
            outlier_mag = 0.0
            if is_outlier:
                outlier_mag = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 12.0)

            eta_d = rng.normal(0.0, sigma_d)
            eta_theta = rng.normal(0.0, sigma_theta)

            z = predicted_measurement(truth_k[i], truth_k[j])
            z = z.copy()
            z[0] += eta_d + outlier_mag
            # Bearing left *unwrapped* (matches the unwrapped prediction): the
            # circular boundary is handled by wrap_innovation at update time.
            z[1] = z[1] + eta_theta

            links.append(LinkDraw(
                i=i, j=j, d=d, snr=snr, p_succ=p_succ, sigma_d=sigma_d,
                sigma_theta=sigma_theta, received=received,
                is_outlier=is_outlier, eta_d=eta_d, eta_theta=eta_theta,
                tof_jitter=ch.tof_jitter(sigma_d, is_outlier, chp),
                z=z))

    onboard: list[OnboardDraw] = []
    for i in range(n):
        is_anchor = i in anchors
        z_on, R_on = onboard_measurement(truth_k[i], sen, rng, is_anchor)
        onboard.append(OnboardDraw(z_on=z_on, R_on=R_on, is_anchor=is_anchor))

    return EpochDraw(k=k, wind=wind, links=tuple(links),
                     onboard=tuple(onboard))
