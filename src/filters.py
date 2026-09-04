"""
M2-M5 -- Cooperative localization filters (paper Sec. Methods / Sec. IV).

All filters share the prediction step (the known Fossen motion model with the
known control sequence) and the onboard (compass/DVL/GNSS) update; they differ
only in how cooperative range-bearing measurements are fused:

  EKF-CL   extended-Kalman update, fixed R_c
  CKF-CL   third-degree spherical-radial cubature update, fixed R_c
  IWCF     single-round information-form update, fixed R_c
  VB-CIF   variational-Bayesian cubature information filter (adaptive R)
  G-VB-CIF VB-CIF guarded by a residual-dominance NIS gate -> CKF-CL fallback

Everything here is a *pure function* of the state and measurement: the shared
channel/noise draws and per-vehicle state are owned by the scenario/runner
layers, so a method's update never draws its own randomness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.measurement import (
    measurement_jacobian, predicted_measurement, wrap_innovation)
from params import FilterParams

N_STATES = 6
N_MEAS = 2


# ---------------------------------------------------------------------------
# Cubature rule (third-degree spherical-radial, 2n = 12 points)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubatureRule:
    """2n cubature points xi_l = sqrt(n)[I, -I]_l and weights 1/(2n)."""
    n: int = N_STATES

    @property
    def points(self) -> NDArray:
        return np.sqrt(self.n) * np.concatenate(
            [np.eye(self.n), -np.eye(self.n)], axis=0)

    @property
    def weights(self) -> NDArray:
        return np.full(2 * self.n, 1.0 / (2 * self.n))


# ---------------------------------------------------------------------------
# Onboard update (linear KF on compass / DVL / GNSS)
# ---------------------------------------------------------------------------

def onboard_measurement_matrix(is_anchor: bool) -> NDArray:
    """H_on selecting the onboard observables from x = [x, y, psi, u, v, r]."""
    if is_anchor:
        idx = [0, 1, 2, 3, 4]      # [x, y, psi, u, v]
    else:
        idx = [2, 3, 4]            # [psi, u, v]
    H = np.zeros((len(idx), N_STATES))
    for row, col in enumerate(idx):
        H[row, col] = 1.0
    return H


def onboard_update(x: NDArray, P: NDArray, z_on: NDArray,
                   R_on: NDArray, is_anchor: bool) -> tuple[NDArray, NDArray]:
    """Linear-Kalman onboard update; wraps the heading innovation."""
    H = onboard_measurement_matrix(is_anchor)
    innov = z_on - H @ x
    heading_col = 2 if is_anchor else 0
    innov[heading_col] = float(np.arctan2(
        np.sin(innov[heading_col]), np.cos(innov[heading_col])))
    S = H @ P @ H.T + R_on
    K = P @ H.T @ np.linalg.inv(S)
    x_upd = x + K @ innov
    P_upd = (np.eye(N_STATES) - K @ H) @ P
    return x_upd, P_upd


# ---------------------------------------------------------------------------
# Cooperative updates
# ---------------------------------------------------------------------------

def ekf_coop_update(x: NDArray, P: NDArray, z: NDArray,
                    x_nb: NDArray, R_c: NDArray) -> tuple[NDArray, NDArray]:
    """EKF-CL: linearized update with fixed R_c."""
    H = measurement_jacobian(x, x_nb)
    innov = wrap_innovation(z - predicted_measurement(x, x_nb))
    S = H @ P @ H.T + R_c
    K = P @ H.T @ np.linalg.inv(S)
    x_upd = x + K @ innov
    P_upd = (np.eye(N_STATES) - K @ H) @ P
    return x_upd, P_upd


def ckf_coop_update(x: NDArray, P: NDArray, z: NDArray,
                    x_nb: NDArray, R_c: NDArray,
                    rule: CubatureRule) -> tuple[NDArray, NDArray]:
    """CKF-CL: cubature update with fixed R_c."""
    pts = rule.points
    wts = rule.weights
    sqrt_P = np.linalg.cholesky(P)
    X = x[None, :] + (sqrt_P @ pts.T).T
    Z = np.array([predicted_measurement(X[l], x_nb) for l in range(2 * rule.n)])
    z_pred = np.sum(wts[:, None] * Z, axis=0)

    Pzz = np.zeros((N_MEAS, N_MEAS))
    Pxz = np.zeros((N_STATES, N_MEAS))
    for l in range(2 * rule.n):
        dz = Z[l] - z_pred
        dx = X[l] - x
        Pzz += wts[l] * np.outer(dz, dz)
        Pxz += wts[l] * np.outer(dx, dz)

    S = Pzz + R_c
    K = Pxz @ np.linalg.inv(S)
    innov = wrap_innovation(z - z_pred)
    x_upd = x + K @ innov
    P_upd = P - K @ S @ K.T
    return x_upd, P_upd


def iwcf_coop_update(x: NDArray, P: NDArray, z: NDArray,
                     x_nb: NDArray, R_c: NDArray) -> tuple[NDArray, NDArray]:
    """IWCF: single-round information-form update with fixed R_c."""
    H = measurement_jacobian(x, x_nb)
    innov = wrap_innovation(z - predicted_measurement(x, x_nb))
    R_inv = np.linalg.inv(R_c)
    Y = np.linalg.inv(P) + H.T @ R_inv @ H
    y = np.linalg.inv(P) @ x + H.T @ R_inv @ (innov + H @ x)
    x_upd = np.linalg.solve(Y, y)
    P_upd = np.linalg.inv(Y)
    return x_upd, P_upd


# ---------------------------------------------------------------------------
# Robust / adaptive baselines (paper Sec. Results, S4)
# ---------------------------------------------------------------------------

def _ckf_moments(x: NDArray, P: NDArray, x_nb: NDArray, rule: CubatureRule
                 ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Shared CKF moments: predicted measurement, Pzz, Pxz, and the cubature set."""
    pts = rule.points
    wts = rule.weights
    sqrt_P = np.linalg.cholesky(P)
    X = x[None, :] + (sqrt_P @ pts.T).T
    Z = np.array([predicted_measurement(X[l], x_nb) for l in range(2 * rule.n)])
    z_pred = np.sum(wts[:, None] * Z, axis=0)

    Pzz = np.zeros((N_MEAS, N_MEAS))
    Pxz = np.zeros((N_STATES, N_MEAS))
    for l in range(2 * rule.n):
        dz = Z[l] - z_pred
        dx = X[l] - x
        Pzz += wts[l] * np.outer(dz, dz)
        Pxz += wts[l] * np.outer(dx, dz)
    return z_pred, Pzz, Pxz, X


def huber_ckf_coop_update(x: NDArray, P: NDArray, z: NDArray,
                          x_nb: NDArray, R_c: NDArray,
                          rule: CubatureRule, c: float = 1.345
                          ) -> tuple[NDArray, NDArray]:
    """Huber-robust CKF-CL (outlier-robust baseline, stateless).

    Diagonal-decoupled Huber M-estimator: each measurement component's noise is
    inflated by 1/w for large normalized innovations (|r| > c), so outlier links
    are downweighted instead of inflating a learned R.  This is the standard
    robust-Kalman alternative to residual-based noise adaptation (Karlgaard &
    Schaub 2007).
    """
    z_pred, Pzz, Pxz, _X = _ckf_moments(x, P, x_nb, rule)
    S = Pzz + R_c
    innov = wrap_innovation(z - z_pred)
    d = np.sqrt(np.maximum(np.diag(S), 1e-8))
    r = innov / d
    w = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
    R_tilde = np.diag(np.diag(R_c) / w)
    S_tilde = Pzz + R_tilde
    K = Pxz @ np.linalg.inv(S_tilde)
    x_upd = x + K @ innov
    P_upd = P - K @ S_tilde @ K.T
    return x_upd, P_upd


def iae_coop_update(x: NDArray, P: NDArray, z: NDArray, x_nb: NDArray,
                    R_hat: NDArray, fp: FilterParams, rule: CubatureRule,
                    lam: float = 0.1) -> tuple[NDArray, NDArray, NDArray]:
    """IAE (Mehra innovation-matching, Sage-Husa-type) adaptive-R baseline.

    ``R_hat`` is a *persistent* per-receiver (2x2) noise covariance carried
    across epochs; each received link updates it by innovation-covariance
    matching  R_hat <- (1-lam) R_hat + lam (nu nu^T - Pzz).  Crucially, Pzz is
    the *receiver's own* measurement covariance H P H^T, so -- exactly as in
    VB-CIF -- the *neighbor's* dead-reckoning error is NOT debiased: under
    residual dominance the learned R_hat is inflated and the update gain
    collapses.  This is the stateful counterpart of the per-epoch IW prior used
    by VB-CIF, so a comparison isolates whether the failure is specific to the
    inverse-Wishart prior or is fundamental to residual-based adaptation.
    """
    z_pred, Pzz, Pxz, _X = _ckf_moments(x, P, x_nb, rule)
    S = Pzz + R_hat
    K = Pxz @ np.linalg.inv(S)
    innov = wrap_innovation(z - z_pred)
    x_upd = x + K @ innov
    P_upd = P - K @ S @ K.T

    R_new = (1.0 - lam) * R_hat + lam * (np.outer(innov, innov) - Pzz)
    R_new = 0.5 * (R_new + R_new.T)
    w2, V = np.linalg.eigh(R_new)
    w2 = np.clip(w2, 1e-4, None)
    R_new = V @ np.diag(w2) @ V.T
    return x_upd, P_upd, R_new


# ---------------------------------------------------------------------------
# VB-CIF (adaptive R via inverse-Wishart prior)
# ---------------------------------------------------------------------------

def _iw_prior_params(alpha: float, fp: FilterParams) -> tuple[float, NDArray]:
    """CQM-driven IW prior (tau_prior, Psi_prior).

    Psi_nom is set so that at alpha = 1 (tau_prior = tau_max) the prior mean
    E[R] = Psi/(nu - d - 1) equals the nominal covariance r_nom.
    """
    r_nom = np.diag([fp.sigma_d_nominal ** 2, fp.sigma_theta_nominal ** 2])
    Psi_nom = (fp.tau_max - 2.0 - 1.0) * r_nom
    tau_prior = fp.tau_max - (fp.tau_max - fp.tau_min) * (1.0 - alpha)
    Psi_prior = (1.0 + fp.eta_inflation * (1.0 - alpha)) * Psi_nom
    return float(tau_prior), Psi_prior


def vb_cif_update(x: NDArray, P: NDArray, z: NDArray, x_nb: NDArray,
                  alpha: float, fp: FilterParams,
                  rule: CubatureRule) -> tuple[NDArray, NDArray, NDArray, int]:
    """Per-link VB-CIF: cubature information update + VB-M noise update.

    Returns (x_upd, P_upd, R_inv_post, n_iter).  ``alpha`` is the CQM link
    reliability (held at 0.5 for the baseline VB-CIF).
    """
    pts = rule.points
    wts = rule.weights
    n = rule.n

    tau_prior, Psi_prior = _iw_prior_params(alpha, fp)
    R_inv = tau_prior * np.linalg.inv(Psi_prior)

    Y_pred = np.linalg.inv(P)
    y_pred = Y_pred @ x

    x_upd = x.copy()
    P_upd = P.copy()
    n_iter = 0

    for t in range(fp.max_vb_iter):
        n_iter = t + 1
        H = measurement_jacobian(x_upd, x_nb)
        innov = wrap_innovation(z - predicted_measurement(x_upd, x_nb))

        Y = Y_pred + H.T @ R_inv @ H
        y = y_pred + H.T @ R_inv @ (innov + H @ x_upd)
        x_new = np.linalg.solve(Y, y)
        P_new = np.linalg.inv(Y)

        # VB-M: expected innovation outer product over cubature points.
        sqrt_P = np.linalg.cholesky(P_new)
        Xs = x_new[None, :] + (sqrt_P @ pts.T).T
        Zs = np.array([predicted_measurement(Xs[l], x_nb) for l in range(2 * n)])
        residuals = z[None, :] - Zs
        residuals[:, 1] = np.arctan2(np.sin(residuals[:, 1]),
                                     np.cos(residuals[:, 1]))
        outer_sum = np.sum(
            wts[:, None, None] * residuals[:, :, None] * residuals[:, None, :],
            axis=0)

        # Debias: subtract the *receiver's own* state contribution H P H^T.
        # NOTE (the failure mechanism): the neighbor's estimation error
        # H_j e_j e_j^T H_j^T is NOT subtracted here (x_nb is frozen), so when
        # the neighbor's dead-reckoning error dominates the residual the noise
        # estimate is biased upward and VB-CIF under-weights the measurement
        # (residual dominance).  The G-VB-CIF NIS gate is the guard against it.
        state_cov = H @ P_new @ H.T
        outer_corr = outer_sum - state_cov
        outer_corr = 0.5 * (outer_corr + outer_corr.T)
        w, V = np.linalg.eigh(outer_corr)
        w = np.clip(w, 1e-4, None)
        outer_corr = V @ np.diag(w) @ V.T

        tau_post = tau_prior + 1.0
        Psi_post = Psi_prior + outer_corr
        R_inv_new = tau_post * np.linalg.inv(Psi_post)

        x_upd = x_new
        P_upd = P_new
        if np.max(np.abs(R_inv_new - R_inv)) < fp.vb_tol and t >= 1:
            R_inv = R_inv_new
            break
        R_inv = R_inv_new

    return x_upd, P_upd, R_inv, n_iter


def gvb_cif_update(x: NDArray, P: NDArray, z: NDArray, x_nb: NDArray,
                   alpha: float, fp: FilterParams, R_c: NDArray,
                   rule: CubatureRule) -> tuple[NDArray, NDArray, bool]:
    """G-VB-CIF: residual-dominance NIS gate, else standard VB update.

    Returns (x_upd, P_upd, gate_disabled).  When the normalized innovation
    squared exceeds kappa the innovation is state-dominated (neighbor error +
    common-mode drift) and the guard falls back to a fixed-covariance CKF-CL
    update for that link.
    """
    H = measurement_jacobian(x, x_nb)
    innov = wrap_innovation(z - predicted_measurement(x, x_nb))
    S = H @ P @ H.T + R_c
    nis = float(innov @ np.linalg.solve(S, innov))

    if nis > fp.guard_nis_kappa:
        x_upd, P_upd = ckf_coop_update(x, P, z, x_nb, R_c, rule)
        return x_upd, P_upd, True

    x_upd, P_upd, _ri, _ni = vb_cif_update(x, P, z, x_nb, alpha, fp, rule)
    return x_upd, P_upd, False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHODS = ("ekf", "ckf", "iwcf", "vb", "gvb")


def coop_update(method: str, x: NDArray, P: NDArray, z: NDArray,
                x_nb: NDArray, R_c: NDArray, fp: FilterParams,
                rule: CubatureRule, alpha: float = 0.5
                ) -> tuple[NDArray, NDArray]:
    """Dispatch a cooperative update by method name."""
    if method == "ekf":
        return ekf_coop_update(x, P, z, x_nb, R_c)
    if method == "ckf":
        return ckf_coop_update(x, P, z, x_nb, R_c, rule)
    if method == "iwcf":
        return iwcf_coop_update(x, P, z, x_nb, R_c)
    if method == "vb":
        return vb_cif_update(x, P, z, x_nb, alpha, fp, rule)[:2]
    if method == "gvb":
        return gvb_cif_update(x, P, z, x_nb, alpha, fp, R_c, rule)[:2]
    raise ValueError(f"unknown method {method!r}")
