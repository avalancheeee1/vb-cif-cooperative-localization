"""
M0 -- USV dynamics and ground-truth trajectory generation.

Cybership-II 3-DOF Fossen model (surge, sway, yaw) integrated with RK4,
driven by a PD LOS (line-of-sight) heading controller that produces lawnmower
survey trajectories with 50 m spacing at 1 m/s nominal speed.

State ``x = [x, y, psi, u, v, r]`` (North-East position, heading, body-frame
surge/sway velocity, yaw rate); control ``tau = [tau_u, tau_r]`` (surge
thrust, rudder angle).

Consistency with the strictly paired protocol: the fleet executes an
*open-loop* pre-planned survey, so the control sequence is computed from the
(shared) truth once and reused verbatim by every filter's prediction.  The
"unmodeled prediction disturbance" ``w ~ N(0, Q)`` is drawn *once per acoustic
epoch* (not per ``dt`` sub-step) and added to the truth at epoch boundaries;
the filter models the same disturbance with the same per-epoch covariance Q.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from params import DynamicsParams, Params

N_STATES = 6
N_CTRL = 2


def rotation_matrix(psi: float) -> NDArray:
    """Earth-fixed to body-fixed rotation R(psi) in SO(2) (embedded in R^3)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass(frozen=True)
class CybershipII:
    """Inertia and damping matrices of the Cybership-II model."""
    params: DynamicsParams

    @cached_property
    def M_inv(self) -> NDArray:
        return np.linalg.inv(self.params.M)

    @cached_property
    def D(self) -> NDArray:
        return self.params.D

    @cached_property
    def _m11(self) -> float:
        return self.params.m11

    @cached_property
    def _m22(self) -> float:
        return self.params.m22

    @cached_property
    def _m23(self) -> float:
        return self.params.m23

    @cached_property
    def _length(self) -> float:
        return self.params.length

    def continuous_dynamics(self, x: NDArray, tau: NDArray) -> NDArray:
        """Return dx/dt = f(x, tau) for the 3-DOF Fossen model."""
        nu = x[3:6]                        # [u, v, r]
        psi = x[2]

        c, s = np.cos(psi), np.sin(psi)
        eta_dot = np.array([c * nu[0] - s * nu[1],
                            s * nu[0] + c * nu[1],
                            nu[2]])

        # Coriolis-centripetal matrix (simplified low-speed form).
        c21 = self._m22 * nu[1] + self._m23 * nu[2]
        cor = np.array([
            -c21 * nu[2],
            self._m11 * nu[0] * nu[2],
            c21 * nu[0] - self._m11 * nu[0] * nu[1],
        ])

        # Underactuated control: surge thrust + rudder yaw moment.
        tau_vec = np.array([tau[0], 0.0, -self._length * tau[0] * tau[1]])

        nu_dot = self.M_inv @ (tau_vec - cor - self.D @ nu)
        return np.concatenate([eta_dot, nu_dot])


def continuous_jacobian(model: CybershipII, x: NDArray,
                        tau: NDArray) -> NDArray:
    """Analytic Jacobian ``A = df/dx`` of the 3-DOF Fossen model (6x6).

    The control ``tau`` is treated as fixed (open-loop), so ``A`` carries no
    ``d(tau)/dx`` terms.  Used to build the exact RK4 state-transition
    Jacobian without finite differencing.
    """
    nu = x[3:6]
    psi = x[2]
    u, v, r = nu
    c, s = np.cos(psi), np.sin(psi)

    m11 = model._m11
    m22 = model._m22
    m23 = model._m23

    # d(eta_dot)/d(psi) for eta_dot = [c*u - s*v, s*u + c*v, r].
    d_eta_psi = np.array([-s * u - c * v, c * u - s * v, 0.0])
    # d(eta_dot)/d(nu).
    d_eta_nu = np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]])

    # d(cor)/d(nu) for the Coriolis-centripetal force vector.
    dcor = np.array([
        [0.0, -m22 * r, -m22 * v - 2.0 * m23 * r],
        [m11 * r, 0.0, m11 * u],
        [m22 * v + m23 * r - m11 * v, m22 * u - m11 * u, m23 * u],
    ])

    d_nu = model.M_inv @ (-dcor - model.D)

    A = np.zeros((6, 6))
    A[0:3, 2] = d_eta_psi
    A[0:3, 3:6] = d_eta_nu
    A[3:6, 3:6] = d_nu
    return A


def rk4_step(model: CybershipII, x: NDArray, tau: NDArray,
             dt: float) -> NDArray:
    """One deterministic RK4 integration step (no process noise)."""
    h = dt
    k1 = model.continuous_dynamics(x, tau)
    k2 = model.continuous_dynamics(x + 0.5 * h * k1, tau)
    k3 = model.continuous_dynamics(x + 0.5 * h * k2, tau)
    k4 = model.continuous_dynamics(x + h * k3, tau)
    x_next = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[2] = _wrap(x_next[2])
    return x_next


def rk4_step_with_jac(model: CybershipII, x: NDArray, tau: NDArray,
                      dt: float) -> tuple[NDArray, NDArray]:
    """One RK4 step plus its exact state-transition Jacobian.

    Returns ``(x_next, J)`` with ``J = d(x_next)/dx``, obtained by
    differentiating the RK4 tableau analytically (no finite differencing).
    """
    h = dt
    k1 = model.continuous_dynamics(x, tau)
    x2 = x + 0.5 * h * k1
    k2 = model.continuous_dynamics(x2, tau)
    x3 = x + 0.5 * h * k2
    k3 = model.continuous_dynamics(x3, tau)
    x4 = x + h * k3
    k4 = model.continuous_dynamics(x4, tau)

    x_next = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[2] = _wrap(x_next[2])

    eye = np.eye(6)
    A1 = continuous_jacobian(model, x, tau)
    A2 = continuous_jacobian(model, x2, tau)
    A3 = continuous_jacobian(model, x3, tau)
    A4 = continuous_jacobian(model, x4, tau)

    jk1 = A1
    jk2 = A2 @ (eye + 0.5 * h * jk1)
    jk3 = A3 @ (eye + 0.5 * h * jk2)
    jk4 = A4 @ (eye + h * jk3)

    J = eye + (h / 6.0) * (jk1 + 2.0 * jk2 + 2.0 * jk3 + jk4)
    return x_next, J


def waypoint_control(x: NDArray, waypoint: NDArray,
                     k_heading: float = 0.8, thrust_base: float = 2.0,
                     thrust_extra: float = 0.0, dist_ref: float = 50.0,
                     rudder_max: float = 0.5) -> NDArray:
    """Waypoint-following controller (paper Sec. Experimental setup).

    Proportional heading controller toward ``waypoint`` with a rudder that
    saturates at ``rudder_max``.  The surge thrust is calibrated to the paper's
    "1 m/s nominal" speed: steady-state surge u = thrust / d11 = 2.0 N / 2.0,
    so ``thrust_base = 2.0`` gives u = 1 m/s.  (The legacy ``thrust_base = 5.0``
    gave 2.5 m/s, which — combined with the corrected lawnmower heading loop —
    produced yaw rates up to ~2 rad/s that the 1 Hz compass/DVL cannot track and
    inflated the dead-reckoning error ~8x.)
    """
    diff = waypoint[:2] - x[:2]
    dist = float(np.hypot(diff[0], diff[1]))
    des_heading = _wrap(np.arctan2(diff[1], diff[0]))
    e = _wrap(des_heading - x[2])
    # Negative feedback: an aft rudder maps *positive* deflection to a *negative*
    # yaw moment (see tau_vec in continuous_dynamics), so a positive heading
    # error e = des_heading - psi requires a *negative* rudder to close the loop.
    # The original +k_heading*e sign inverted the loop (positive feedback), which
    # made the vehicles steer AWAY from the waypoint and wander instead of
    # surveying the lawnmower.  This is the corrected heading controller.
    tau_r = float(np.clip(-k_heading * e, -rudder_max, rudder_max))
    tau_u = thrust_base + thrust_extra * (1.0 - min(dist / dist_ref, 1.0))
    return np.array([tau_u, tau_r])


def _lawnmower_waypoints(origin_x: float, leg: float, spacing: float,
                         n_legs: int = 12) -> NDArray:
    """Per-vehicle lawnmower waypoints (paper Sec. Experimental setup).

    Vehicle ``i`` starts at ``(i*spacing, leg)`` and surveys a boustrophedon:
    east along ``y = leg``, south to ``y = 0``, east, north, ... so the fleet
    forms a collinear line of USVs scanning in parallel lanes.  The y coordinate
    alternates leg/0/0/leg (two consecutive waypoints share a lane before the
    north-south sweep), i.e. y = leg when ``k % 4 in (0, 1)``.  The legacy
    ``k % 2 == 1`` alternation instead produced a diagonal sawtooth (missing the
    "east along y = 0" legs), which is not a boustrophedon.
    """
    wps = [(origin_x, leg)]                       # (i*spacing, leg)
    for k in range(1, n_legs):
        y = leg if k % 4 in (0, 1) else 0.0
        x = origin_x + (k + 1) // 2 * spacing
        wps.append((x, y))
    return np.array(wps)


# Lawnmower geometry (paper: 50 m spacing, 1 m/s nominal, 120 s horizon).
_SPACING = 50.0           # m between adjacent vehicles
_LEG = 150.0              # m north-south survey leg
_N_LEGS = 12              # enough legs for the 120 s horizon
_WP_RADIUS = 8.0          # m waypoint reach radius


def finite_diff_jacobian(f, x: NDArray, eps: float = 1e-6) -> NDArray:
    """Forward-difference Jacobian of f: R^n -> R^n at x."""
    n = len(x)
    f0 = f(x)
    J = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (f(xp) - f0) / eps
    return J


def propagate_epoch(model: CybershipII, x: NDArray, controls: NDArray,
                    dt: float) -> tuple[NDArray, NDArray]:
    """Deterministic propagation over one acoustic epoch.

    ``controls`` has shape ``(acoustic_period, 2)``.  Returns ``(x_next, F)``
    where ``F = d(x_next)/dx`` is the exact Jacobian of the epoch transition,
    chained from the per-step analytic RK4 Jacobians.
    """
    xx = x.copy()
    F = np.eye(6)
    for tau in controls:
        xx, J = rk4_step_with_jac(model, xx, tau, dt)
        F = J @ F
    return xx, F


def _propagate_fleet(sys, model: CybershipII, Q: NDArray, dt: float,
                     rng: np.random.Generator, truth: NDArray,
                     controls: NDArray, wps_all: list[NDArray]) -> None:
    """Shared open-loop propagation: waypoint-follow + per-``dt`` process noise.

    Fills ``truth`` and ``controls`` in place.  ``wps_all`` is a per-vehicle
    ``(n_wps, 2)`` waypoint list; the fleet starts at ``truth[:, 0]`` (already
    set by the caller, so a collinear or random topology only differs in its
    initial states and waypoints -- the propagation loop is identical).
    """
    n = sys.n_usvs
    n_wps = wps_all[0].shape[0]
    wp_idx = np.ones(n, dtype=int)          # target wps[1] initially

    for k in range(sys.n_steps):
        for i in range(n):
            wps = wps_all[i]
            target = wps[min(wp_idx[i], n_wps - 1)]
            diff = target[:2] - truth[i, k, :2]
            if np.hypot(diff[0], diff[1]) < _WP_RADIUS and wp_idx[i] < n_wps - 1:
                wp_idx[i] += 1
                target = wps[wp_idx[i]]
            controls[i, k] = waypoint_control(truth[i, k], target)
            w = rng.multivariate_normal(np.zeros(N_STATES), Q * dt)
            truth[i, k + 1] = rk4_step(model, truth[i, k], controls[i, k], dt)
            truth[i, k + 1] += w
            truth[i, k + 1, 2] = _wrap(truth[i, k + 1, 2])


def generate_trajectory(params: Params,
                        rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """Generate shared ground truth and the open-loop control sequence.

    Returns
    -------
    truth : (n_usvs, n_steps + 1, 6)
    controls : (n_usvs, n_steps, 2)
        ``controls[i, k]`` is the waypoint-following control computed from
        ``truth[i, k]``; shared verbatim by every filter's prediction step.

    The fleet is a *collinear line*: vehicle ``i`` starts at ``(i*50, 150)``
    heading east at 1 m/s (paper Sec. Experimental setup).  The unmodeled
    prediction disturbance ``w ~ N(0, Q*dt)`` is drawn every ``dt`` sub-step
    (a continuous-time random walk) and added to the truth; the filter models
    the same disturbance with the per-epoch covariance ``Q`` (the sum of
    ``Q*dt`` over the epoch).
    """
    sys = params.system
    model = CybershipII(params.dynamics)
    Q = params.noise.Q
    dt = sys.dt
    n = sys.n_usvs

    wps_all = [_lawnmower_waypoints(i * _SPACING, _LEG, _SPACING, _N_LEGS)
               for i in range(n)]

    truth = np.zeros((n, sys.n_steps + 1, N_STATES))
    controls = np.zeros((n, sys.n_steps, N_CTRL))
    for i in range(n):
        truth[i, 0] = [i * _SPACING, _LEG, 0.0, 1.0, 0.0, 0.0]

    _propagate_fleet(sys, model, Q, dt, rng, truth, controls, wps_all)
    return truth, controls


def generate_trajectory_random(params: Params,
                               rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """Random-topology variant of :func:`generate_trajectory` (paper S6).

    Same survey *footprint* as the collinear fleet -- a box of width
    ``(n-1)*50`` m and height ``150`` m -- but each vehicle starts at a
    uniformly scattered (non-collinear) position with a random heading and
    follows an independent random sequence of waypoints inside the box.  The
    inter-vehicle range/bearing link graph is therefore a *random topology*
    rather than the parallel-lane boustrophedon; everything else (the same
    per-``dt`` process-noise walk, the same waypoint controller, the same
    shared verbatim control) is identical, so any VB-vs-CKF gap that reappears
    here is a property of the residual-dominance mechanism, not of the
    collinear geometry.
    """
    sys = params.system
    model = CybershipII(params.dynamics)
    Q = params.noise.Q
    dt = sys.dt
    n = sys.n_usvs

    area_x = (n - 1) * _SPACING
    truth = np.zeros((n, sys.n_steps + 1, N_STATES))
    controls = np.zeros((n, sys.n_steps, N_CTRL))
    wps_all = []
    for i in range(n):
        x0 = float(rng.uniform(0.0, area_x))
        y0 = float(rng.uniform(0.0, _LEG))
        psi0 = float(rng.uniform(-np.pi, np.pi))
        truth[i, 0] = [x0, y0, psi0, 1.0, 0.0, 0.0]
        wps = [(x0, y0)]
        for _ in range(_N_LEGS - 1):
            wps.append((float(rng.uniform(0.0, area_x)),
                        float(rng.uniform(0.0, _LEG))))
        wps_all.append(np.array(wps))

    _propagate_fleet(sys, model, Q, dt, rng, truth, controls, wps_all)
    return truth, controls
