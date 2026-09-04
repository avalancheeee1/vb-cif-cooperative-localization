"""
M1 validation -- acoustic channel + range-bearing measurement model.

Checks:
  1. SNR spans the manuscript's 5--40 dB over 50--500 m at wind 8 m/s.
  2. Range/bearing noise stds decrease monotonically with SNR.
  3. Sea-state switching: noise stds x2 above 11 m/s.
  4. Outlier probability in [0, 0.2] and decreasing with SNR.
  5. Packet success sigmoid: 0.5 at SNR == SNR_th, monotone increasing.
  6. Measurement Jacobian matches finite differences.
  7. Prediction matches d = ||p_i - p_j|| and bearing = atan2 - psi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from params import DEFAULT_PARAMS  # noqa: E402
from src import channel as ch  # noqa: E402
from src.measurement import (  # noqa: E402
    measurement_jacobian, predicted_measurement)


def _fd_jacobian(x_i, x_j, eps=1e-6):
    h0 = predicted_measurement(x_i, x_j)
    H = np.zeros((2, 6))
    for j in range(6):
        xp = x_i.copy()
        xp[j] += eps
        hp = predicted_measurement(xp, x_j)
        H[:, j] = (hp - h0) / eps
    H[:, 2] = _wrap_col(H[:, 2], h0)
    return H


def _wrap_col(col, h0):
    # bearing row may cross the -pi/pi seam under finite differences; wrap
    col = col.copy()
    col[1] = np.arctan2(np.sin(col[1]), np.cos(col[1]))
    return col


def main() -> int:
    p = DEFAULT_PARAMS
    c = p.channel
    failures = []

    # 1. SNR range at wind 8 m/s (sea state 3-4, calm baseline).
    snr_lo = ch.mean_snr(500.0, 8.0, c)
    snr_hi = ch.mean_snr(50.0, 8.0, c)
    print(f"SNR @ 50 m  = {snr_hi:.1f} dB")
    print(f"SNR @ 500 m = {snr_lo:.1f} dB")
    if not (snr_hi <= 40.0 + 1.0 and snr_lo >= 5.0 - 1.0):
        failures.append(f"SNR out of 5-40 dB range: [{snr_lo:.1f}, {snr_hi:.1f}]")

    # 2. Noise stds monotone decreasing in SNR.
    snrs = [5, 10, 20, 30, 40]
    sd = [ch.range_noise_std(s, 8.0, c) for s in snrs]
    st = [ch.bearing_noise_std(s, 8.0, c) for s in snrs]
    if not all(sd[i] > sd[i + 1] for i in range(len(sd) - 1)):
        failures.append("range noise std not monotone decreasing in SNR")
    if not all(st[i] > st[i + 1] for i in range(len(st) - 1)):
        failures.append("bearing noise std not monotone decreasing in SNR")

    # 3. Sea-state switching x2 above 11 m/s.
    sd_calm = ch.range_noise_std(20.0, 8.0, c)
    sd_gust = ch.range_noise_std(20.0, 18.0, c)
    ratio = sd_gust / sd_calm
    print(f"sea-state ratio @ SNR=20 dB: {ratio:.3f}")
    if not (1.9 < ratio < 2.1):
        failures.append(f"sea-state scaling not x2 (ratio {ratio:.3f})")

    # 4. Outlier probability bounded and decreasing.
    pouts = [ch.outlier_probability(s) for s in snrs]
    if not all(0.0 <= po <= 0.2 for po in pouts):
        failures.append("outlier probability outside [0, 0.2]")
    if not all(pouts[i] > pouts[i + 1] for i in range(len(pouts) - 1)):
        failures.append("outlier probability not decreasing in SNR")

    # 5. Packet success sigmoid.
    ps_th = ch.packet_success(c.snr_th, c)
    ps_lo = ch.packet_success(c.snr_th - 10.0, c)
    ps_hi = ch.packet_success(c.snr_th + 10.0, c)
    print(f"p_succ @ SNR_th={c.snr_th} dB: {ps_th:.3f} "
          f"(lo {ps_lo:.3f}, hi {ps_hi:.3f})")
    if not (0.49 < ps_th < 0.51):
        failures.append("packet success not 0.5 at threshold")
    if not (ps_lo < ps_th < ps_hi):
        failures.append("packet success not monotone increasing")

    # 6. Measurement Jacobian vs finite differences.
    rng = np.random.default_rng(0)
    x_i = np.array([10.0, 5.0, 0.7, 1.0, 0.1, 0.02])
    x_j = np.array([30.0, 25.0, 1.2, 1.0, -0.1, -0.03])
    H_ana = measurement_jacobian(x_i, x_j)
    H_fd = _fd_jacobian(x_i, x_j)
    err = np.max(np.abs(H_ana - H_fd))
    print(f"Jacobian max abs err vs FD: {err:.2e}")
    if err > 1e-4:
        failures.append(f"Jacobian mismatch (err {err:.2e})")

    # 7. Prediction correctness.
    z = predicted_measurement(x_i, x_j)
    d_true = float(np.hypot(x_j[0] - x_i[0], x_j[1] - x_i[1]))
    th_true = float(np.arctan2(x_j[1] - x_i[1], x_j[0] - x_i[0]) - x_i[2])
    th_true = float(np.arctan2(np.sin(th_true), np.cos(th_true)))
    if abs(z[0] - d_true) > 1e-9 or abs(z[1] - th_true) > 1e-9:
        failures.append("prediction disagrees with analytic d/theta")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nM1 VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
