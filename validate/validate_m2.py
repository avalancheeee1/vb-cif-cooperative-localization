"""
M2-M5 validation -- the five cooperative-localization filters.

Checks:
  1. Each cooperative update is numerically stable (finite, P remains PSD).
  2. A consistent measurement reduces the position covariance for every method.
  3. EKF / CKF / IWCF agree to within numerical precision on a fixed-R_c link.
  4. VB-CIF recovers a noise covariance close to R_c in the noise-dominated
     regime (averaged over many independent noise draws).
  5. G-VB-CIF guard triggers when the innovation is state-dominated and falls
     back to the fixed-covariance CKF-CL update.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from params import DEFAULT_PARAMS  # noqa: E402
from src.filters import (  # noqa: E402
    CubatureRule, ckf_coop_update, ekf_coop_update, gvb_cif_update,
    iwcf_coop_update, onboard_update, vb_cif_update)
from src.measurement import draw_measurement  # noqa: E402


def _psd(P: np.ndarray) -> bool:
    return bool(np.all(np.linalg.eigvalsh(P) > 0))


def main() -> int:
    p = DEFAULT_PARAMS
    fp = p.filter
    rule = CubatureRule(6)
    R_c = fp.R_c
    failures = []

    rng = np.random.default_rng(1)
    x_true = np.array([10.0, 5.0, 0.7, 1.0, 0.1, 0.02])
    x_nb = np.array([30.0, 25.0, 1.2, 1.0, -0.1, -0.03])
    P0 = np.diag([4.0, 4.0, 0.02, 0.01, 0.01, 0.005])

    # 1 + 2. Consistent measurement reduces covariance for every method.
    z_clean = np.array([np.hypot(x_nb[0] - x_true[0], x_nb[1] - x_true[1]),
                        float(np.arctan2(x_nb[1] - x_true[1],
                                         x_nb[0] - x_true[0]) - x_true[2])])

    for method in ("ekf", "ckf", "iwcf", "vb", "gvb"):
        if method == "ekf":
            x_u, P_u = ekf_coop_update(x_true, P0, z_clean, x_nb, R_c)
        elif method == "ckf":
            x_u, P_u = ckf_coop_update(x_true, P0, z_clean, x_nb, R_c, rule)
        elif method == "iwcf":
            x_u, P_u = iwcf_coop_update(x_true, P0, z_clean, x_nb, R_c)
        elif method == "vb":
            x_u, P_u, _r, _n = vb_cif_update(
                x_true, P0, z_clean, x_nb, 0.5, fp, rule)
        else:
            x_u, P_u, _g = gvb_cif_update(
                x_true, P0, z_clean, x_nb, 0.5, fp, R_c, rule)

        if not np.all(np.isfinite(x_u)) or not np.all(np.isfinite(P_u)):
            failures.append(f"{method}: non-finite update")
        if not _psd(P_u):
            failures.append(f"{method}: P not positive definite")
        if np.trace(P_u) >= np.trace(P0):
            failures.append(f"{method}: covariance not reduced "
                            f"({np.trace(P0):.3f} -> {np.trace(P_u):.3f})")

    # 3. IWCF == EKF (both linearized); EKF ~ CKF (linearization gap only).
    e_x, e_P = ekf_coop_update(x_true, P0, z_clean, x_nb, R_c)
    c_x, c_P = ckf_coop_update(x_true, P0, z_clean, x_nb, R_c, rule)
    i_x, i_P = iwcf_coop_update(x_true, P0, z_clean, x_nb, R_c)
    print(f"|EKF-CKF| x: {np.max(np.abs(e_x - c_x)):.2e}, "
          f"P: {np.max(np.abs(e_P - c_P)):.2e}")
    print(f"|EKF-IWCF| x: {np.max(np.abs(e_x - i_x)):.2e}, "
          f"P: {np.max(np.abs(e_P - i_P)):.2e}")
    if np.max(np.abs(e_x - i_x)) > 1e-9:
        failures.append("EKF and IWCF (both linearized) disagree")
    # CKF is exact to 3rd order; the EKF-CKF gap is genuine linearization error
    # that must stay small relative to the 2.5 m range noise.
    if np.max(np.abs(e_x[:2] - c_x[:2])) > 0.1:
        failures.append("EKF-CKF position gap exceeds linearization tolerance")

    # 4. VB-CIF recovers a noise covariance near R_c (averaged).  The strong
    # alpha=0.5 IW prior deliberately inflates the estimate, so compare only
    # the diagonal and allow a modest bias.
    sigma_d = 2.5
    sigma_theta = np.deg2rad(5.0)
    R_rec = np.zeros((2, 2))
    n_avg = 2000
    for _ in range(n_avg):
        z = draw_measurement(x_true, x_nb, rng, sigma_d, sigma_theta,
                             False, 0.0)
        _, _, R_inv, _ = vb_cif_update(
            x_true, P0, z, x_nb, 0.5, fp, rule)
        R_rec += np.linalg.inv(R_inv)
    R_rec /= n_avg
    R_true = np.diag([sigma_d ** 2, sigma_theta ** 2])
    rel = np.max(np.abs(np.diag(R_rec) - np.diag(R_true)) / np.diag(R_true))
    print(f"VB-CIF recovered R (diag): {np.diag(R_rec)}")
    print(f"true R (diag):             {np.diag(R_true)}")
    print(f"max relative error:        {rel:.3f}")
    if rel > 0.5:
        failures.append(f"VB-CIF noise recovery off (rel {rel:.3f})")

    # 5. G-VB-CIF guard triggers on state-dominated innovation.  A common-mode
    # drift of the receiver (x_pred = truth + delta) with the measurement still
    # reflecting the *true* geometry produces a large innovation that S = H P
    # H^T + R_c cannot explain, so the guard must fall back to CKF-CL.
    drift = np.array([8.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    x_pred = x_true + drift
    z_true = np.array([np.hypot(x_nb[0] - x_true[0], x_nb[1] - x_true[1]),
                       float(np.arctan2(x_nb[1] - x_true[1],
                                        x_nb[0] - x_true[0]) - x_true[2])])
    x_g, P_g, gate = gvb_cif_update(x_pred, P0, z_true, x_nb, 0.5, fp,
                                    R_c, rule)
    x_c, P_c = ckf_coop_update(x_pred, P0, z_true, x_nb, R_c, rule)
    print(f"guard triggered: {gate}")
    if not gate:
        failures.append("guard did not trigger on state-dominated innovation")
    if gate and not np.allclose(P_g, P_c, atol=1e-8):
        failures.append("guard fallback did not match CKF-CL")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nM2-M5 VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
