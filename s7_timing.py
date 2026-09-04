"""
M7.8 -- isolated cooperative-update timing (review recommended #3, timing side).

The paper's computation-cost claim ("VB-CIF costs Xx the CKF-CL") must rest on
a reproducible, single-baseline measurement.  This micro-benchmark times the
*cooperative update in isolation* -- no trajectory, channel, onboard, or
prediction overhead -- so the ratio is the pure filter cost per received link.

It times:

  * ckf_coop_update            (fixed R_c cubature, the baseline)
  * vb_cif_update              (adaptive R, max_iter in {2, 3, 4, 6})

and reports microseconds-per-update plus the VB/CKF ratio, together with the
mean fixed-point iteration count (which quantifies the early-exit behavior the
paper cites in Sec. "Computation").  The representative state is a settled
6-D pose with a 50 m / 0.1 rad range-bearing link.

Results -> outputs/s7_timing.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src.filters import CubatureRule, ckf_coop_update, vb_cif_update

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

N_REP = 5000


def representative_state():
    x = np.array([10.0, 20.0, 0.3, 1.0, 0.0, 0.0])
    P = np.diag([1.0, 1.0, 0.01, 0.05, 0.05, 0.01])   # settled covariance
    z = np.array([50.0, 0.1])                          # range 50 m, bearing 0.1 rad
    x_nb = np.array([58.0, 23.5, 0.3, 1.0, 0.0, 0.0])
    return x, P, z, x_nb


def time_fn(fn, n_rep):
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / n_rep                    # microseconds per call


def main():
    params = DEFAULT_PARAMS
    fp = params.filter
    rule = CubatureRule(6)
    R_c = fp.R_c
    x, P, z, x_nb = representative_state()

    # Warm-up (JIT/allocator settling).
    ckf_coop_update(x, P, z, x_nb, R_c, rule)
    vb_cif_update(x, P, z, x_nb, 0.5, fp, rule)

    ckf_us = time_fn(lambda: ckf_coop_update(x, P, z, x_nb, R_c, rule), N_REP)

    rows = []
    for mi in (2, 3, 4, 6):
        fpp = replace(fp, max_vb_iter=mi)
        # Deterministic fixed-point count for this state (single evaluation).
        _xu, _Pu, _ri, n_iter = vb_cif_update(x, P, z, x_nb, 0.5, fpp, rule)
        vb_us = time_fn(
            lambda: vb_cif_update(x, P, z, x_nb, 0.5, fpp, rule), N_REP)
        rows.append({
            "max_vb_iter": mi,
            "us_per_update": round(vb_us, 3),
            "mean_n_iter": int(n_iter),
            "ratio_vs_ckf": round(vb_us / ckf_us, 2),
        })

    out = {
        "ckf_us_per_update": round(ckf_us, 3),
        "n_rep": N_REP,
        "rows": rows,
    }
    (OUT / "s7_timing.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", OUT / "s7_timing.json")


if __name__ == "__main__":
    main()
