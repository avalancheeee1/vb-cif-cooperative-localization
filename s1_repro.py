"""
M7 diagnostic -- reproduce the paper's S1 qualitative result (VB-CIF worse
than CKF-CL) and resolve the clean-vs-legacy metric/RNG question.

Runs the strictly paired runner for N seeds at the S1 config (n=8, 120 s,
plr=0, all links) and reports, for each method, BOTH:
  * MAE  = mean over (vehicle, epoch) of |pos error|      (legacy 'armse')
  * RMS  = mean over epoch of sqrt(mean over vehicle pe^2) (clean 'armse')

so the two definitions can be compared on identical data.  It also prints the
VB-vs-CKF sign so we can see whether "VB worse than CKF" is robust across seeds
or a small-sample artifact of the oscillatory truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src import dynamics as dyn
from src.filters import CubatureRule, coop_update, gvb_cif_update, onboard_update
from src.scenario import draw_epoch
from src.runner import draw_perturbations, _links_by_rx

METHODS = ("ekf", "ckf", "iwcf", "vb", "gvb")


def run_trial_metrics(seed: int, params) -> dict:
    sysp = params.system
    fp = params.filter
    model = dyn.CybershipII(params.dynamics)
    rule = CubatureRule(6)
    Q = params.noise.Q
    R_c = fp.R_c
    P0 = fp.P0
    n = sysp.n_usvs
    period = sysp.acoustic_period
    n_epochs = sysp.n_epochs

    rng = np.random.default_rng(seed)
    truth, controls = dyn.generate_trajectory(params, rng)
    pert = draw_perturbations(params, rng)

    state = {}
    for m in METHODS:
        x0 = truth[:, 0].copy() + pert
        x0[:, 2] = np.arctan2(np.sin(x0[:, 2]), np.cos(x0[:, 2]))
        state[m] = (x0, np.broadcast_to(P0, (n, 6, 6)).copy())

    mae_sum = {m: 0.0 for m in METHODS}
    rms_sum = {m: 0.0 for m in METHODS}

    dt_tot = period * sysp.dt
    for k in range(n_epochs):
        sub = (k + 1) * period
        epoch = draw_epoch(truth[:, sub], k, params, rng, 0.0, frozenset())
        links_by_rx = _links_by_rx(epoch)
        disturb = rng.normal(0.0, params.noise.unmodeled_sigma_pos,
                             (n, 2)) * np.sqrt(dt_tot)

        for m in METHODS:
            x_prev, P_prev = state[m]
            x_prop_all = np.empty_like(x_prev)
            for i in range(n):
                x_prop_all[i], _F = dyn.propagate_epoch(
                    model, x_prev[i], controls[i, k * period:sub], sysp.dt)

            x_new = np.empty_like(x_prev)
            P_new = np.empty_like(P_prev)
            for i in range(n):
                x_pred = x_prop_all[i].copy()
                P_pred = P_prev[i] + Q
                x_pred[:2] += disturb[i]
                z_on = epoch.onboard[i].z_on
                R_on = epoch.onboard[i].R_on
                x_on, P_on = onboard_update(x_pred, P_pred, z_on, R_on, False)
                for link in links_by_rx.get(i, ()):
                    x_nb = x_prop_all[link.j]
                    if m == "gvb":
                        x_on, P_on, _g = gvb_cif_update(
                            x_on, P_on, link.z, x_nb, 0.5, fp, R_c, rule)
                    else:
                        x_on, P_on = coop_update(
                            m, x_on, P_on, link.z, x_nb, R_c, fp, rule, alpha=0.5)
                x_new[i] = x_on
                P_new[i] = P_on

            state[m] = (x_new, P_new)
            pe = np.linalg.norm(x_new[:, :2] - truth[:, sub, :2], axis=1)
            mae_sum[m] += pe.sum()
            rms_sum[m] += float(np.sqrt(np.mean(pe ** 2)))

    # MAE normalizes by the total number of (vehicle, epoch) samples, which is
    # n_epochs * n (NOT per-method: ``cnt`` was previously incremented inside the
    # method loop, inflating the divisor by n_methods = 5).
    out = {}
    for m in METHODS:
        out[m] = {"mae": mae_sum[m] / (n_epochs * n), "rms": rms_sum[m] / n_epochs}
    return out


def main(seeds=30):
    params = DEFAULT_PARAMS
    acc = {m: {"mae": [], "rms": []} for m in METHODS}
    vb_gt_ckf_mae = 0
    vb_gt_ckf_rms = 0
    for s in range(seeds):
        r = run_trial_metrics(s, params)
        for m in METHODS:
            acc[m]["mae"].append(r[m]["mae"])
            acc[m]["rms"].append(r[m]["rms"])
        if r["vb"]["mae"] > r["ckf"]["mae"]:
            vb_gt_ckf_mae += 1
        if r["vb"]["rms"] > r["ckf"]["rms"]:
            vb_gt_ckf_rms += 1

    print(f"{'method':>5} | {'MAE mean':>9} {'MAE std':>8} | {'RMS mean':>9} {'RMS std':>8}")
    for m in METHODS:
        a = np.array(acc[m]["mae"]); b = np.array(acc[m]["rms"])
        print(f"{m:>5} | {a.mean():>9.3f} {a.std(ddof=1):>8.3f} | {b.mean():>9.3f} {b.std(ddof=1):>8.3f}")
    print(f"\nVB > CKF (i.e. VB worse): MAE {vb_gt_ckf_mae}/{seeds}, RMS {vb_gt_ckf_rms}/{seeds}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
