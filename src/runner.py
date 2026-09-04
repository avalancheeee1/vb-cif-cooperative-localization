"""
M6 -- Strictly paired Monte-Carlo runner (Appendix B, critical path).

Structure: the epoch is the *outer* loop and the method is the *inner* loop,
so that every method in a trial sees the identical truth, control sequence,
channel realization, measurement noise, onboard noise, initial perturbation,
and (for the unmodeled disturbance) the same process-noise covariance.  Only
the algorithm under test differs.

Within an epoch each vehicle predicts through the known Fossen model with the
known control sequence, applies its onboard update, then folds in the received
cooperative links in a *canonical* order (sorted by true distance, shared
across methods).  Neighbor estimates are frozen at the epoch boundary (Jacobi
style), so there is no intra-epoch vehicle-ordering dependence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from params import Params
from src import dynamics as dyn
from src.filters import (
    CubatureRule, coop_update, gvb_cif_update, huber_ckf_coop_update,
    iae_coop_update, onboard_update)
from src.scenario import EpochDraw, draw_epoch

METHODS_S1 = ("ekf", "ckf", "iwcf", "vb", "gvb")


@dataclass
class MethodResult:
    """Per-method summary over one trial."""
    armse: float = 0.0                    # mean position RMSE over epochs (m)
    divergence: bool = False              # any epoch RMSE > threshold
    max_rmse: float = 0.0                 # max per-epoch RMSE (m)
    rmse_per_epoch: NDArray = None        # (n_epochs,) position RMSE
    guard_n_total: int = 0                # G-VB-CIF gate trials
    guard_n_disabled: int = 0             # G-VB-CIF gate falls back to CKF-CL

    @property
    def guard_disable_rate(self) -> float:
        return (self.guard_n_disabled / self.guard_n_total
                if self.guard_n_total else 0.0)


@dataclass
class TrialResult:
    """All method results for one trial (shared truth/perturbation/draws)."""
    plr: float
    methods: dict[str, MethodResult] = field(default_factory=dict)


def draw_perturbations(params: Params,
                       rng: np.random.Generator) -> NDArray:
    """Initial position/heading perturbation, drawn once per vehicle (shared)."""
    fp = params.filter
    n = params.system.n_usvs
    pert = np.zeros((n, 6))
    for i in range(n):
        pert[i, 0] = rng.normal(0.0, fp.sigma_init_pos)
        pert[i, 1] = rng.normal(0.0, fp.sigma_init_pos)
        pert[i, 2] = rng.normal(0.0, fp.sigma_init_heading)
    return pert


def _links_by_rx(epoch: EpochDraw) -> dict[int, list]:
    """Group received links by receiver and sort each by true distance."""
    by_rx: dict[int, list] = {}
    for link in epoch.links:
        if not link.received:
            continue
        by_rx.setdefault(link.i, []).append(link)
    for links in by_rx.values():
        links.sort(key=lambda lk: lk.d)       # canonical order
    return by_rx


def run_trial(params: Params, rng: np.random.Generator, plr: float,
              methods: tuple[str, ...] = METHODS_S1,
              anchors: frozenset[int] = frozenset(),
              trajectory_fn=dyn.generate_trajectory) -> TrialResult:
    """Run one strictly paired trial and return per-method ARMSE/divergence.

    ``trajectory_fn`` selects the truth/control generator (collinear lawnmower
    by default; ``dyn.generate_trajectory_random`` for the random-topology
    robustness sweep S6).  Every method in the trial sees the same output.
    """
    sys = params.system
    fp = params.filter
    model = dyn.CybershipII(params.dynamics)
    rule = CubatureRule(6)
    Q = params.noise.Q
    R_c = fp.R_c
    P0 = fp.P0
    n = sys.n_usvs
    period = sys.acoustic_period
    n_epochs = sys.n_epochs

    # Shared truth + open-loop control sequence.
    truth, controls = trajectory_fn(params, rng)
    # Shared initial perturbation (once per vehicle).
    pert = draw_perturbations(params, rng)

    # Method state: {method: (x (n,6), P (n,6,6))}.
    state: dict[str, tuple[NDArray, NDArray]] = {}
    for m in methods:
        x0 = truth[:, 0].copy() + pert
        x0[:, 2] = np.arctan2(np.sin(x0[:, 2]), np.cos(x0[:, 2]))
        state[m] = (x0, np.broadcast_to(P0, (n, 6, 6)).copy())

    results = {m: MethodResult() for m in methods}
    # Persistent per-receiver adaptive-R state for the IAE baseline (reset to
    # the fixed R_c at trial start; carried across epochs thereafter).
    rhat: dict[str, NDArray] = {
        m: np.broadcast_to(R_c, (n, 2, 2)).copy()
        for m in methods if m == "iae"}
    anchors_sorted = sorted(anchors)

    dt_tot = period * sys.dt
    for k in range(n_epochs):
        sub = (k + 1) * period                      # sub-step of this epoch
        epoch = draw_epoch(truth[:, sub], k, params, rng, plr, anchors)
        links_by_rx = _links_by_rx(epoch)
        # Shared unmodeled position disturbance (once per epoch, all methods).
        disturb = rng.normal(0.0, params.noise.unmodeled_sigma_pos,
                             (n, 2)) * np.sqrt(dt_tot)

        for m in methods:
            x_prev, P_prev = state[m]
            x_new = np.empty_like(x_prev)
            P_new = np.empty_like(P_prev)
            guard_total = 0
            guard_disabled = 0

            # Predicted state of every vehicle, frozen at the epoch boundary
            # (Jacobi style): each receiver's cooperative update uses its
            # neighbors' *predicted* states (propagated through the known
            # control, no disturbance, no onboard correction).
            x_prop_all = np.empty_like(x_prev)
            for i in range(n):
                x_prop_all[i], _F = dyn.propagate_epoch(
                    model, x_prev[i], controls[i, k * period:sub], sys.dt)

            for i in range(n):
                is_anchor = i in anchors
                x_pred = x_prop_all[i].copy()
                P_pred = P_prev[i] + Q
                x_pred[:2] += disturb[i]           # unmodeled disturbance

                z_on = epoch.onboard[i].z_on
                R_on = epoch.onboard[i].R_on
                x_on, P_on = onboard_update(x_pred, P_pred, z_on, R_on,
                                            is_anchor)

                for link in links_by_rx.get(i, ()):
                    x_nb = x_prop_all[link.j]      # predicted neighbor state
                    if m == "gvb":
                        x_on, P_on, gate = gvb_cif_update(
                            x_on, P_on, link.z, x_nb, 0.5, fp, R_c, rule)
                        guard_total += 1
                        guard_disabled += int(gate)
                    elif m == "iae":
                        x_on, P_on, rhat[m][i] = iae_coop_update(
                            x_on, P_on, link.z, x_nb, rhat[m][i], fp, rule)
                    elif m == "huber":
                        x_on, P_on = huber_ckf_coop_update(
                            x_on, P_on, link.z, x_nb, R_c, rule)
                    else:
                        x_on, P_on = coop_update(
                            m, x_on, P_on, link.z, x_nb, R_c, fp, rule,
                            alpha=0.5)

                x_new[i] = x_on
                P_new[i] = P_on

            state[m] = (x_new, P_new)
            results[m].guard_n_total += guard_total
            results[m].guard_n_disabled += guard_disabled

        # Per-epoch position RMSE for every method (shared sub-step index).
        for m in methods:
            pos_err = np.linalg.norm(state[m][0][:, :2] - truth[:, sub, :2],
                                     axis=1)
            rmse = float(np.sqrt(np.mean(pos_err ** 2)))
            results[m].rmse_per_epoch = np.append(
                results[m].rmse_per_epoch, rmse) if results[m].rmse_per_epoch is not None \
                else np.array([rmse])

    for m in methods:
        results[m].armse = float(results[m].rmse_per_epoch.mean())
        results[m].max_rmse = float(results[m].rmse_per_epoch.max())
        results[m].divergence = results[m].max_rmse > params.stats.divergence_threshold

    return TrialResult(plr=plr, methods=results)
