"""
M7.7 -- VB-CIF fitted-noise-covariance evolution (review mandatory revision #1).

The review asked for a *quantitative* trace of how the VB-CIF's estimated
measurement-noise covariance drifts over time, contrasted against (i) the true
per-link noise realized in the shared channel draw and (ii) the fixed
covariance R_c used by CKF-CL, at compass sigma_psi in {0.5, 5} deg.

For every acoustic epoch this records, aggregated over all *received* links and
all vehicles:

    sigma_d_hat   = sqrt(diag(R_hat)[0])   VB-CIF fitted range noise (m)
    sigma_th_hat  = sqrt(diag(R_hat)[1])   VB-CIF fitted bearing noise (rad)
    sigma_d_true  = mean(link.sigma_d)      true range noise (m)
    sigma_th_true = mean(link.sigma_theta)  true bearing noise (rad)

where R_hat = R_inv^-1 is the VB-CIF posterior measurement covariance returned
by ``vb_cif_update``.  The fixed R_c diagonal (2.5 m, 5 deg) is a constant and
is therefore not re-run here; it is drawn as a reference line in the figure.

Results are written to outputs/s1_noise_covariance.json.  The plot companion is
``paper_v2/make_figures_revision.py`` (``draw_fig8_noise_cov``).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src import dynamics as dyn
from src.filters import CubatureRule, onboard_update, vb_cif_update
from src.runner import _links_by_rx, draw_perturbations
from src.scenario import draw_epoch

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

COMPASS_DEG = (0.5, 5.0)


def run_noise_cov_trial(params, rng, plr, anchors=frozenset()):
    """Run one strictly paired VB-CIF trial, recording the per-epoch fitted and
    true noise standard deviations (aggregated over received links/vehicles).

    Returns dicts keyed by epoch index (0..n_epochs-1) holding the per-epoch
    mean of the fitted and true range/bearing noise standard deviations.
    """
    sys = params.system
    fp = params.filter
    model = dyn.CybershipII(params.dynamics)
    rule = CubatureRule(6)
    Q = params.noise.Q
    P0 = fp.P0
    n = sys.n_usvs
    period = sys.acoustic_period
    n_epochs = sys.n_epochs

    truth, controls = dyn.generate_trajectory(params, rng)
    pert = draw_perturbations(params, rng)

    x0 = truth[:, 0].copy() + pert
    x0[:, 2] = np.arctan2(np.sin(x0[:, 2]), np.cos(x0[:, 2]))
    x = x0
    P = np.broadcast_to(P0, (n, 6, 6)).copy()

    dt_tot = period * sys.dt

    rec = {
        "sigma_d_hat": np.empty(n_epochs),
        "sigma_th_hat": np.empty(n_epochs),
        "sigma_d_true": np.empty(n_epochs),
        "sigma_th_true": np.empty(n_epochs),
    }

    for k in range(n_epochs):
        sub = (k + 1) * period
        epoch = draw_epoch(truth[:, sub], k, params, rng, plr, anchors)
        links_by_rx = _links_by_rx(epoch)
        disturb = rng.normal(0.0, params.noise.unmodeled_sigma_pos,
                             (n, 2)) * np.sqrt(dt_tot)

        x_new = np.empty_like(x)
        P_new = np.empty_like(P)

        # Predicted (frozen, Jacobi-style) neighbor states for this epoch.
        x_prop_all = np.empty_like(x)
        for i in range(n):
            x_prop_all[i], _F = dyn.propagate_epoch(
                model, x[i], controls[i, k * period:sub], sys.dt)

        sd_hat, st_hat, sd_true, st_true = [], [], [], []
        for i in range(n):
            x_pred = x_prop_all[i].copy()
            P_pred = P[i] + Q
            x_pred[:2] += disturb[i]

            z_on = epoch.onboard[i].z_on
            R_on = epoch.onboard[i].R_on
            x_on, P_on = onboard_update(x_pred, P_pred, z_on, R_on, False)

            for link in links_by_rx.get(i, ()):
                x_nb = x_prop_all[link.j]
                x_on, P_on, R_inv, _ni = vb_cif_update(
                    x_on, P_on, link.z, x_nb, 0.5, fp, rule)
                R_hat = np.linalg.inv(R_inv)
                sd_hat.append(float(np.sqrt(max(R_hat[0, 0], 0.0))))
                st_hat.append(float(np.sqrt(max(R_hat[1, 1], 0.0))))
                sd_true.append(float(link.sigma_d))
                st_true.append(float(link.sigma_theta))

            x_new[i] = x_on
            P_new[i] = P_on

        x, P = x_new, P_new
        rec["sigma_d_hat"][k] = float(np.mean(sd_hat)) if sd_hat else np.nan
        rec["sigma_th_hat"][k] = float(np.mean(st_hat)) if st_hat else np.nan
        rec["sigma_d_true"][k] = float(np.mean(sd_true)) if sd_true else np.nan
        rec["sigma_th_true"][k] = float(np.mean(st_true)) if st_true else np.nan

    return rec


def sweep_level(compass_deg, seeds):
    params = replace(
        DEFAULT_PARAMS,
        sensors=replace(DEFAULT_PARAMS.sensors,
                        compass_sigma=float(np.deg2rad(compass_deg))))

    acc = {key: [] for key in
           ("sigma_d_hat", "sigma_th_hat", "sigma_d_true", "sigma_th_true")}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        rec = run_noise_cov_trial(params, rng, 0.0)
        for key in acc:
            acc[key].append(rec[key])

    out = {
        "compass_deg": compass_deg,
        "n_trials": seeds,
        "sigma_d_fixed": params.filter.sigma_d_fixed,
        "sigma_theta_fixed_rad": float(params.filter.sigma_theta_fixed),
    }
    for key in acc:
        arr = np.asarray(acc[key])
        out[f"{key}_mean"] = np.nanmean(arr, axis=0).tolist()
        out[f"{key}_std"] = np.nanstd(arr, axis=0).tolist()
    return out


def main(seeds=10):
    levels = [sweep_level(c, seeds) for c in COMPASS_DEG]
    (OUT / "s1_noise_covariance.json").write_text(
        json.dumps({"levels": levels}, indent=2))

    for lv in levels:
        print(f"compass {lv['compass_deg']:g} deg: "
              f"final fitted sigma_d={lv['sigma_d_hat_mean'][-1]:.3f} m "
              f"(true {lv['sigma_d_true_mean'][-1]:.3f} m, "
              f"fixed {lv['sigma_d_fixed']:.3f} m); "
              f"sigma_th={np.rad2deg(lv['sigma_th_hat_mean'][-1]):.3f} deg "
              f"(true {np.rad2deg(lv['sigma_th_true_mean'][-1]):.3f} deg, "
              f"fixed {np.rad2deg(lv['sigma_theta_fixed_rad']):.3f} deg)")
    print(f"wrote {OUT / 's1_noise_covariance.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    main(args.seeds)
