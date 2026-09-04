"""
M8 -- Communication Quality Metric (paper Sec. Methods, Eq. 9-14).

For each in-range link the CQM produces a scalar reliability score
``alpha^{i,j} in (0,1)`` from three physical-layer indicators:

  I_snr  = 1 - exp(-SNR / SNR_ref)           (soft SNR indicator)
  I_prr  = sliding-window packet reception rate (window W)
  I_tofj = exp(-sigma_ToF / sigma_ref)       (ToF-jitter indicator)

Each indicator's Beta likelihood under reliable/degraded hypotheses is fitted
offline (Appendix); ``alpha`` is updated by a temporally-smoothed recursive
Bayes rule with forgetting factor lambda and smoothing gain gamma.  The PRR
likelihood ratio is bounded to [0.3, 3] so externally injected packet loss is
not double-counted as channel degradation.

The CQM is used *only* for link admission (scheduling).  The per-link VB-CIF
filter holds the link-reliability weighting fixed at alpha=0.5 (paper Sec. II),
so this module does not feed the VB prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaln

from params import CQMParams

# Offline-calibrated Beta likelihood parameters (MLE, paper Appendix).  These
# match simulation/outputs/data/cqm_beta_params.json in the legacy package.
BETA_PARAMS = {
    "snr": {"a_r": 21.83015046577027, "b_r": 6.26972334341673,
            "a_d": 0.6858704749095752, "b_d": 1.0403538569368882},
    "prr": {"a_r": 7.654431784098138, "b_r": 1.5983895482985047,
            "a_d": 0.40606783848675, "b_d": 1.1064094895339485},
    "tofj": {"a_r": 6.878989926128319, "b_r": 9.599403929410082,
             "a_d": 0.6757939366163536, "b_d": 4.353653416532615},
}


def _beta_pdf(x: float, a: float, b: float) -> float:
    """Beta density at x in (0,1), clamped to the open interval."""
    x = min(max(x, 1e-10), 1.0 - 1e-10)
    return float(np.exp((a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
                        - betaln(a, b)))


@dataclass
class LinkCQMState:
    """Per-link CQM state, keyed by (receiver i, transmitter j)."""
    alpha: float = 0.5
    prr_window: list = field(default_factory=list)
    last_tofj: float = 0.003


class CQM:
    """Maintains per-link CQM scores across epochs for a single trial."""

    def __init__(self, cfg: CQMParams):
        self.cfg = cfg
        self._states: dict[tuple[int, int], LinkCQMState] = {}

    def get_alpha(self, i: int, j: int) -> float:
        """Current CQM score for link i->j (0.5 if never observed)."""
        st = self._states.get((i, j))
        return st.alpha if st is not None else 0.5

    def update(self, i: int, j: int, snr_db: float, tof_jitter: float,
               received: bool, attempted: bool) -> float:
        """Update the CQM score for link i->j and return the new alpha.

        ``received`` should already be masked by ``attempted`` (a link that is
        not interrogated cannot register a packet loss), and the PRR window is
        advanced only for attempted links.
        """
        st = self._states.get((i, j))
        if st is None:
            st = LinkCQMState()
            self._states[(i, j)] = st
        cfg = self.cfg

        if attempted:
            st.prr_window.append(1.0 if received else 0.0)
            if len(st.prr_window) > cfg.window_size:
                st.prr_window.pop(0)
        prr = float(np.mean(st.prr_window)) if st.prr_window else 0.5

        i_snr = 1.0 - np.exp(-max(snr_db, 0.0) / cfg.snr_ref)
        i_prr = prr
        i_tofj = np.exp(-tof_jitter / cfg.sigma_ref)

        bp = BETA_PARAMS
        lr_snr = (_beta_pdf(i_snr, bp["snr"]["a_r"], bp["snr"]["b_r"])
                  / (_beta_pdf(i_snr, bp["snr"]["a_d"], bp["snr"]["b_d"]) + 1e-30))
        lr_prr = np.clip(
            _beta_pdf(i_prr, bp["prr"]["a_r"], bp["prr"]["b_r"])
            / (_beta_pdf(i_prr, bp["prr"]["a_d"], bp["prr"]["b_d"]) + 1e-30),
            cfg.prr_lr_lo, cfg.prr_lr_hi)
        lr_tofj = (_beta_pdf(i_tofj, bp["tofj"]["a_r"], bp["tofj"]["b_r"])
                   / (_beta_pdf(i_tofj, bp["tofj"]["a_d"], bp["tofj"]["b_d"]) + 1e-30))
        if not cfg.use_snr:
            lr_snr = 1.0
        if not cfg.use_prr:
            lr_prr = 1.0
        if not cfg.use_tofj:
            lr_tofj = 1.0
        lr = lr_snr * lr_prr * lr_tofj

        lam = cfg.forgetting_factor
        a_prev = st.alpha
        a_eff = a_prev ** lam
        a_neg_eff = (1.0 - a_prev) ** lam
        a_tilde = a_eff * lr / (a_eff * lr + a_neg_eff + 1e-12)

        alpha_new = (1.0 - cfg.smooth_gain) * a_prev + cfg.smooth_gain * a_tilde
        st.alpha = float(alpha_new)
        if received:
            st.last_tofj = tof_jitter
        return float(alpha_new)
