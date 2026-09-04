"""
M8 -- Strictly paired Monte-Carlo runner for the scheduling experiment (S2).

This is the scheduling-aware counterpart of ``runner.run_trial``.  It compares
(filter x scheduling-policy) combinations under an interrogation budget ``B``:
each receiver may interrogate at most ``B`` of its in-range neighbors per epoch.
The proposed method is CQM-driven scheduling of the VB-CIF filter
(``cqa_vbcif``); the baselines are CKF/VB under nearest or random scheduling and
the oracle (true-noise) upper bound.

  ckf_cl      CKF, nearest-B            (baseline filter + baseline schedule)
  ckf_random  CKF, random-B             (baseline filter + random schedule)
  vb_ckf      VB-CIF, nearest-B         (proposed filter + baseline schedule)
  cqa_vbcif   VB-CIF, CQM-driven B      (proposed: filter + scheduling)
  ckf_oracle  CKF, true-noise-B         (unachievable upper bound)

Strict pairing is preserved: the shared truth, channel realization, measurement
noise, onboard noise, initial perturbation and unmodeled disturbance are all
drawn from the *trial* RNG once per epoch (never per method).  The one
method-specific source of randomness -- the ``random`` policy's link choice --
uses a per-method RNG seeded from a fixed preamble so it cannot shift the shared
stream and confound the comparison.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from params import Params
from src import dynamics as dyn
from src.cqm import CQM
from src.filters import CubatureRule, coop_update, onboard_update
from src.runner import MethodResult, TrialResult, draw_perturbations
from src.scenario import draw_epoch
from src.scheduler import select_links

# (method name, filter, scheduling policy) -- order defines the report.
SCHED_SPECS = (
    ("ckf_cl",     "ckf", "nearest"),
    ("ckf_random", "ckf", "random"),
    ("vb_ckf",     "vb",  "nearest"),
    ("cqa_vbcif",  "vb",  "cqm"),
    ("ckf_oracle", "ckf", "oracle"),
)
SCHED_METHODS = tuple(s[0] for s in SCHED_SPECS)

# S3 (anchors): all methods use the CKF filter so the *scheduling policy* is the
# only varying factor, isolating the anchor-selection contribution.
SCHED_SPECS_S3 = (
    ("ckf_nearest",      "ckf", "nearest"),
    ("ckf_random",       "ckf", "random"),
    ("ckf_anchor_first", "ckf", "anchor_first"),
    ("ckf_cqm",          "ckf", "cqm"),
    ("ckf_oracle",       "ckf", "oracle"),
)
S3_METHODS = tuple(s[0] for s in SCHED_SPECS_S3)

_SPEC_REGISTRY = {s[0]: (s[1], s[2]) for s in SCHED_SPECS + SCHED_SPECS_S3}


def run_trial_sched(params: Params, rng: np.random.Generator, plr: float,
                    budget: int, methods: tuple[str, ...] = SCHED_METHODS,
                    anchors: frozenset[int] = frozenset()) -> TrialResult:
    """Run one strictly paired scheduling trial at interrogation budget ``B``."""
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
    spec_map = {m: _SPEC_REGISTRY[m] for m in methods}

    # Shared truth + open-loop control sequence; shared initial perturbation.
    truth, controls = dyn.generate_trajectory(params, rng)
    pert = draw_perturbations(params, rng)

    # Per-method policy RNGs: seeded from a fixed preamble so the `random`
    # policy's draws never shift the shared stream (see module docstring).
    seeds = rng.integers(0, 2 ** 31, size=len(methods))
    policy_rng = {m: np.random.default_rng(int(s))
                  for m, s in zip(methods, seeds)}

    # Method state: (x (n,6), P (n,6,6)); a CQM instance per method (only the
    # cqm policy reads/writes it, but constructing one each is cheap).
    state: dict[str, tuple[NDArray, NDArray]] = {}
    cqms: dict[str, CQM] = {}
    for m in methods:
        x0 = truth[:, 0].copy() + pert
        x0[:, 2] = np.arctan2(np.sin(x0[:, 2]), np.cos(x0[:, 2]))
        state[m] = (x0, np.broadcast_to(P0, (n, 6, 6)).copy())
        cqms[m] = CQM(params.cqm)

    results = {m: MethodResult() for m in methods}
    dt_tot = period * sys.dt

    for k in range(n_epochs):
        sub = (k + 1) * period
        epoch = draw_epoch(truth[:, sub], k, params, rng, plr, anchors)
        # All in-range links per receiver (received OR not): the scheduler must
        # choose over the full candidate set; `received` only gates fusion.
        candidates_by_rx: dict[int, list] = {}
        for link in epoch.links:
            candidates_by_rx.setdefault(link.i, []).append(link)
        disturb = rng.normal(0.0, params.noise.unmodeled_sigma_pos,
                             (n, 2)) * np.sqrt(dt_tot)

        for m in methods:
            filt, policy = spec_map[m]
            x_prev, P_prev = state[m]
            cqm = cqms[m]
            x_new = np.empty_like(x_prev)
            P_new = np.empty_like(P_prev)

            x_prop_all = np.empty_like(x_prev)
            for i in range(n):
                x_prop_all[i], _F = dyn.propagate_epoch(
                    model, x_prev[i], controls[i, k * period:sub], sys.dt)

            for i in range(n):
                is_anchor = i in anchors
                x_pred = x_prop_all[i].copy()
                P_pred = P_prev[i] + Q
                x_pred[:2] += disturb[i]

                x_on, P_on = onboard_update(
                    x_pred, P_pred, epoch.onboard[i].z_on,
                    epoch.onboard[i].R_on, is_anchor)

                candidates = candidates_by_rx.get(i, ())
                x_nb = {lk.j: x_prop_all[lk.j] for lk in candidates}
                selected = select_links(
                    policy, i, list(candidates), x_on, P_on, x_nb, cqm,
                    fp, policy_rng[m], budget, anchors,
                    params.cqm.anchor_value_weight)

                if policy == "cqm":
                    selected_ids = {lk.j for lk in selected}
                    for lk in candidates:
                        attempted = lk.j in selected_ids
                        cqm.update(i, lk.j, lk.snr, lk.tof_jitter,
                                   lk.received and attempted, attempted)

                for lk in selected:                     # canonical (true-range) order
                    if not lk.received:
                        continue
                    x_on, P_on = coop_update(
                        filt, x_on, P_on, lk.z, x_nb[lk.j], R_c, fp, rule,
                        alpha=0.5)

                x_new[i] = x_on
                P_new[i] = P_on

            state[m] = (x_new, P_new)

        for m in methods:
            pos_err = np.linalg.norm(state[m][0][:, :2] - truth[:, sub, :2],
                                     axis=1)
            rmse = float(np.sqrt(np.mean(pos_err ** 2)))
            results[m].rmse_per_epoch = (
                np.append(results[m].rmse_per_epoch, rmse)
                if results[m].rmse_per_epoch is not None else np.array([rmse]))

    for m in methods:
        results[m].armse = float(results[m].rmse_per_epoch.mean())
        results[m].max_rmse = float(results[m].rmse_per_epoch.max())
        results[m].divergence = (results[m].max_rmse
                                 > params.stats.divergence_threshold)

    return TrialResult(plr=plr, methods=results)
