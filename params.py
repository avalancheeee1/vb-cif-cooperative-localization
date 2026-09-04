"""
Single source of truth for every simulation parameter.

These values are transcribed from the manuscript
``CooperativeLocalization_GNSSDenied_Study.tex`` (the paper), which is the
authoritative reference.  The legacy ``simulation/config/default.yaml`` is
*not* used: several of its values (beta, lambda, gamma, duration, M) drifted
from the paper and are stale.

The dataclasses are frozen so that a run can never silently mutate its own
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


def _deg2rad(deg: float) -> float:
    return float(np.deg2rad(deg))


@dataclass(frozen=True)
class SystemParams:
    """Swarm, sampling, and horizon (paper Sec. ``Experimental setup``)."""
    n_usvs: int = 8                      # N = 8 Cybership-II USVs
    dt: float = 0.1                      # Ts (s), zero-order-hold sampling
    duration: float = 120.0              # 120 s simulation horizon
    acoustic_rate: float = 1.0           # acoustic interrogation epoch = 1 Hz
    n_anchors: int = 0                   # 0 for S1/S2, 2 for S3

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt))

    @property
    def acoustic_period(self) -> int:
        """Number of dt steps between acoustic interrogation epochs."""
        return int(round(self.acoustic_rate ** -1 / self.dt))

    @property
    def n_epochs(self) -> int:
        """Number of acoustic interrogation epochs over the horizon."""
        return self.n_steps // self.acoustic_period


@dataclass(frozen=True)
class DynamicsParams:
    """Cybership-II 3-DOF Fossen model (Skjetne et al. 2005)."""
    # Normalized inertia matrix M (incl. added mass)
    m11: float = 25.8
    m22: float = 33.8
    m23: float = 1.0115
    m32: float = 1.0115
    m33: float = 2.76
    # Normalized damping matrix D
    d11: float = 2.0
    d22: float = 7.0
    d23: float = 0.1
    d32: float = 0.1
    d33: float = 0.5
    # Hull geometry (m) -- used for the rudder yaw-moment model
    length: float = 1.255

    @cached_property
    def M(self) -> np.ndarray:
        return np.array([[self.m11, 0, 0],
                         [0, self.m22, self.m23],
                         [0, self.m32, self.m33]])

    @cached_property
    def D(self) -> np.ndarray:
        return np.array([[self.d11, 0, 0],
                         [0, self.d22, self.d23],
                         [0, self.d32, self.d33]])


@dataclass(frozen=True)
class ProcessNoiseParams:
    """Process-noise covariance Q (paper Eq. for w ~ N(0, Q))."""
    sigma_pos: float = 0.1              # m
    sigma_heading: float = _deg2rad(0.5)  # rad
    sigma_vel: float = 0.01             # m/s
    sigma_yaw_rate: float = _deg2rad(0.1)  # rad/s
    # Unmodeled position disturbance (currents / model mismatch), drawn once
    # per epoch and added to the *estimate* only (not the truth, not P).  This
    # is the term the residual-dominance guard is designed to reject.
    unmodeled_sigma_pos: float = 0.05   # m per sqrt(epoch)

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.sigma_pos ** 2, self.sigma_pos ** 2,
                        self.sigma_heading ** 2, self.sigma_vel ** 2,
                        self.sigma_vel ** 2, self.sigma_yaw_rate ** 2])


@dataclass(frozen=True)
class SensorParams:
    """Onboard sensors (paper Sec. II)."""
    dvl_sigma_vel: float = 0.05          # m/s (1 Hz)
    compass_sigma: float = _deg2rad(0.5)  # rad (0.5 deg, high-end AHRS)
    # The compass (heading) noise is the dominant dead-reckoning error source
    # and therefore the key knob for the residual-dominance failure: 2-5 deg
    # (low-cost MEMS magnetometer) makes the neighbor's position estimate drift
    # enough to dominate the cooperative residual and bias VB-CIF's noise
    # estimate.  See s1_mechanism.py (the honest failure-mechanism experiment).
    gnss_sigma_pos: float = 3.0          # m (5 Hz, anchors only)

    @property
    def R_on_anchor(self) -> np.ndarray:
        """Onboard covariance for an anchor: [x_g, y_g, psi, u, v]."""
        return np.diag([self.gnss_sigma_pos ** 2, self.gnss_sigma_pos ** 2,
                        self.compass_sigma ** 2, self.dvl_sigma_vel ** 2,
                        self.dvl_sigma_vel ** 2])

    @property
    def R_on_follower(self) -> np.ndarray:
        """Onboard covariance for a follower: [psi, u, v]."""
        return np.diag([self.compass_sigma ** 2, self.dvl_sigma_vel ** 2,
                        self.dvl_sigma_vel ** 2])


@dataclass(frozen=True)
class ChannelParams:
    """Shallow-water acoustic channel (paper Appendix A)."""
    sl: float = 190.0                    # dB re 1 uPa @ 1 m
    alpha_abs: float = 6.5               # dB/km Thorp absorption @ 25 kHz
    di: float = 3.0                      # dB directivity index
    c: float = 1500.0                    # m/s sound speed
    b_eff: float = 5000.0                # Hz effective bandwidth
    fading_std_db: float = 6.0           # dB per-link log-normal fading
    beta: float = 0.5                    # packet-success sigmoid steepness
    snr_th: float = 10.0                 # dB packet-success threshold
    max_range: float = 500.0             # m communication range
    multipath_floor: float = 1.5         # m ranging floor (calm SNR)
    # Sea-state switching: noise stds x2 when wind exceeds this (m/s)
    sea_state_threshold: float = 11.0
    sea_state_scale: float = 2.0

    # Wind profile: calm baseline plus two Gaussian gust bumps (paper
    # Sec. Experimental setup): the first peaks at t=120 s (sea state 3-4 ->
    # 5-6, crossing the x2 noise threshold), the second is beyond the 120 s
    # horizon and contributes negligibly.
    wind_baseline: float = 8.0           # m/s baseline (sea state 3-4)
    gust1_amp: float = 5.0               # m/s first-gust amplitude
    gust1_center: float = 120.0          # s -- first-gust peak
    gust1_width: float = 25.0            # s -- first-gust width
    gust2_amp: float = 3.0               # m/s second-gust amplitude
    gust2_center: float = 240.0          # s -- second-gust peak (beyond horizon)
    gust2_width: float = 20.0            # s -- second-gust width

    @property
    def nl_base(self) -> float:
        """Knudsen in-band noise floor at w = 8 m/s: 122 dB."""
        return 122.0

    def nl(self, wind: float) -> float:
        """Effective in-band noise level NL = 122 + 30*log10(w/8) dB."""
        return self.nl_base + 30.0 * np.log10(max(wind, 1e-3) / 8.0)

    def wind(self, t: float) -> float:
        """Wind speed (m/s) at time t: two Gaussian gust bumps on a baseline."""
        return (self.wind_baseline
                + self.gust1_amp * np.exp(-0.5 * ((t - self.gust1_center)
                                                  / self.gust1_width) ** 2)
                + self.gust2_amp * np.exp(-0.5 * ((t - self.gust2_center)
                                                  / self.gust2_width) ** 2))


@dataclass(frozen=True)
class FilterParams:
    """Filter common settings + VB-CIF prior (paper Sec. Methods)."""
    sigma_d_fixed: float = 2.5           # m  (fixed R_c range std)
    sigma_theta_fixed: float = _deg2rad(5.0)  # rad (fixed R_c bearing std)

    # VB-CIF inverse-Wishart prior (baseline)
    tau_max: int = 25
    tau_min: int = 3
    eta_inflation: float = 2.0
    sigma_d_nominal: float = 1.5         # m
    sigma_theta_nominal: float = _deg2rad(3.0)  # rad
    max_vb_iter: int = 6
    vb_tol: float = 1e-4
    alpha_fixed: float = 0.5             # link-reliability held fixed for VB-CIF

    # G-VB-CIF residual-dominance guard
    guard_nis_kappa: float = 5.99        # chi2_{0.95}(2)

    # Initial state: perturbation (drawn once per vehicle, shared) + covariance.
    sigma_init_pos: float = 1.0          # m  (initial position perturbation std)
    sigma_init_heading: float = _deg2rad(2.0)  # rad (initial heading perturb)
    p0_pos: float = 2.0                  # m^2 initial position variance
    p0_heading: float = 0.02             # rad^2 initial heading variance
    p0_vel: float = 0.5                  # (m/s)^2 initial velocity variance
    p0_yawrate: float = 0.02             # (rad/s)^2 initial yaw-rate variance

    @property
    def P0(self) -> np.ndarray:
        """Initial state covariance (diagonal)."""
        return np.diag([self.p0_pos, self.p0_pos, self.p0_heading,
                        self.p0_vel, self.p0_vel, self.p0_yawrate])

    @property
    def R_c(self) -> np.ndarray:
        """Fixed cooperative measurement covariance."""
        return np.diag([self.sigma_d_fixed ** 2, self.sigma_theta_fixed ** 2])


@dataclass(frozen=True)
class CQMParams:
    """Communication Quality Metric (paper Sec. Methods)."""
    snr_ref: float = 20.0                # dB soft-SNR reference
    window_size: int = 20                # PRR sliding-window slots
    sigma_ref: float = 0.002             # 2 ms reference ToF jitter
    forgetting_factor: float = 0.92      # lambda
    smooth_gain: float = 0.2             # gamma
    threshold_reject: float = 0.3
    threshold_reliable: float = 0.7
    prr_lr_lo: float = 0.3               # PRR likelihood-ratio bounds
    prr_lr_hi: float = 3.0
    anchor_value_weight: float = 4.0     # v_a (S3 only)
    # Indicator ablation switches (M10): turning one off sets its likelihood
    # ratio to unity so that indicator contributes no information to alpha.
    use_snr: bool = True
    use_prr: bool = True
    use_tofj: bool = True


@dataclass(frozen=True)
class StatisticsParams:
    """Statistical protocol (paper Sec. Strictly paired evaluation).

    alpha is the two-sided significance level used for the paired t-test, the
    TOST equivalence test, and the Bonferroni correction; the confidence
    intervals are reported at the corresponding ``1 - alpha`` level (95%).  The
    equivalence margin ``tost_margin`` is 0.5 m, a conservative 20% of the fixed
    range-noise standard deviation (2.5 m).
    """
    n_trials: int = 100                  # M for headline comparisons (S1 mechanism sweep)
    n_trials_small: int = 20             # M for ablation / robustness scans
    alpha: float = 0.05                  # two-sided significance level
    power: float = 0.8                   # power for MDD
    tost_margin: float = 0.5             # m equivalence margin
    divergence_threshold: float = 10.0   # m


@dataclass(frozen=True)
class Params:
    """Aggregate configuration."""
    system: SystemParams = field(default_factory=SystemParams)
    dynamics: DynamicsParams = field(default_factory=DynamicsParams)
    noise: ProcessNoiseParams = field(default_factory=ProcessNoiseParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    filter: FilterParams = field(default_factory=FilterParams)
    cqm: CQMParams = field(default_factory=CQMParams)
    stats: StatisticsParams = field(default_factory=StatisticsParams)


# Module-level convenience: the single shared configuration object.
DEFAULT_PARAMS = Params()
