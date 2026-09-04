"""
M1 -- Shallow-water acoustic channel model (paper Appendix A / Sec. II-B).

Deterministic functions only: every stochastic draw (per-link log-normal
fading, reception flag, outlier flag, measurement noise) is the scenario
layer's job (see ``scenario.py``), so the channel realization can be drawn
*once per epoch* and shared verbatim by every filter (strictly paired
protocol).

Equations (manuscript):

    TL(d)   = 20 log10(d) + alpha_abs d / 1000          [dB]
    NL(w)   = 122 + 30 log10(w / 8)                     [dB]
    SNR     = SL - TL(d) - NL(w) + DI + xi,  xi ~ N(0, 6^2) dB
    p_succ  = 1 / (1 + exp(-beta (SNR - SNR_th)))
    p_out   = 0.2 exp(-SNR / 8)
    sigma_d = sqrt( c^2 / (8 pi^2 B_eff^2 SNR_lin) + (1.5 (1 + 4 e^{-SNR/8}))^2 )
    sigma_th = (3 + 5 e^{-SNR/8}) deg
    sigma_d, sigma_th  *= 2  when  wind > sea_state_threshold
    ToF jitter = sigma_d / c + 0.5 ms  (+ 4 ms if outlier)
"""
from __future__ import annotations

import numpy as np

from params import ChannelParams


def transmission_loss(d: float, alpha_abs: float = 6.5) -> float:
    """Thorp spreading-plus-absorption loss TL(d) in dB."""
    d = max(d, 1e-6)
    return 20.0 * np.log10(d) + alpha_abs * d / 1000.0


def noise_level(wind: float) -> float:
    """Knudsen in-band noise level NL(w) in dB (calibrated at 25 kHz)."""
    return 122.0 + 30.0 * np.log10(max(wind, 1e-3) / 8.0)


def mean_snr(d: float, wind: float, ch: ChannelParams) -> float:
    """Deterministic SNR (dB) *without* per-link fading."""
    return ch.sl - transmission_loss(d, ch.alpha_abs) - noise_level(wind) + ch.di


def snr_db(d: float, wind: float, fading_db: float, ch: ChannelParams) -> float:
    """Received SNR (dB) including a per-link log-normal fading draw."""
    return mean_snr(d, wind, ch) + fading_db


def snr_linear(snr: float) -> float:
    """Convert SNR in dB to a linear (power) ratio."""
    return float(10.0 ** (snr / 10.0))


def sea_state_factor(wind: float, ch: ChannelParams) -> float:
    """Multiplicative noise scale: x2 above the sea-state threshold."""
    return ch.sea_state_scale if wind > ch.sea_state_threshold else 1.0


def range_noise_std(snr: float, wind: float, ch: ChannelParams) -> float:
    """Range-noise standard deviation sigma_d (m), including sea-state x2.

    Matches the legacy channel model exactly: the SNR entering the CRLB term
    is floored at -10 dB and the multipath-gain exponent at 0 dB, so a
    severely faded link cannot blow the range noise up to tens of metres.
    """
    snr_lin = 10.0 ** (max(snr, -10.0) / 10.0)
    mp_gain = 1.0 + 4.0 * np.exp(-max(snr, 0.0) / 8.0)
    sd = np.sqrt(ch.c ** 2 / (8.0 * np.pi ** 2 * ch.b_eff ** 2
                              * max(snr_lin, 1.0))
                 + (1.5 * mp_gain * sea_state_factor(wind, ch)) ** 2)
    return float(sd)


def bearing_noise_std(snr: float, wind: float, ch: ChannelParams) -> float:
    """Bearing-noise standard deviation sigma_theta (rad), sea-state x2."""
    sigma_deg = ((3.0 + 5.0 * np.exp(-max(snr, 0.0) / 8.0))
                 * sea_state_factor(wind, ch))
    return float(np.deg2rad(sigma_deg))


def outlier_probability(snr: float, wind: float, ch: ChannelParams) -> float:
    """Multipath outlier probability p_out = p0 exp(-SNR/8), sea-state dependent.

    ``p0`` is 0.12 in calm sea state and 0.20 above the threshold (the exponent
    is floored at 0 dB so a faded link saturates, not explodes, the outlier
    rate).
    """
    p0 = 0.20 if wind > ch.sea_state_threshold else 0.12
    return float(p0 * np.exp(-max(snr, 0.0) / 8.0))


def packet_success(snr: float, ch: ChannelParams) -> float:
    """Sigmoid packet-success probability (before external PLR)."""
    return float(1.0 / (1.0 + np.exp(-ch.beta * (snr - ch.snr_th))))


def tof_jitter(sigma_d: float, is_outlier: bool, ch: ChannelParams) -> float:
    """Per-link ToF jitter (s): sigma_d/c + 0.5 ms, + 4 ms if outlier."""
    jitter = sigma_d / ch.c + 0.5e-3
    if is_outlier:
        jitter += 4e-3
    return float(jitter)
