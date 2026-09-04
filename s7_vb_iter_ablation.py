"""
M7.8 -- VB-CIF iteration-count ablation (review recommended #3).

The review asked for a computation-vs-accuracy trade-off: how much accuracy is
lost when the VB-CIF's fixed-point iteration count ``max_vb_iter`` is reduced
from 6 toward 3--4, against the wall-clock saving.  This sweeps ``max_vb_iter``
in {2, 3, 4, 6} at the nominal 0.5 deg compass (where VB-CIF is well-posed and
must stay accurate), running VB-CIF only, and records the ARMSE plus the
per-epoch, per-vehicle wall-clock time (ms, single-threaded Python prototype).

Results -> outputs/s7_vb_iter_ablation.json.
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
from src.runner import run_trial

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

COMPASS_DEG = 0.5
MAX_ITER = (2, 3, 4, 6)


def sweep(seeds=10):
    rows = []
    n_epochs = DEFAULT_PARAMS.system.n_epochs
    n_usvs = DEFAULT_PARAMS.system.n_usvs
    for mi in MAX_ITER:
        params = replace(
            DEFAULT_PARAMS,
            sensors=replace(DEFAULT_PARAMS.sensors,
                            compass_sigma=float(np.deg2rad(COMPASS_DEG))),
            filter=replace(DEFAULT_PARAMS.filter, max_vb_iter=mi))

        armse = []
        t0 = time.perf_counter()
        for s in range(seeds):
            rng = np.random.default_rng(s)
            res = run_trial(params, rng, 0.0, ("vb",))
            armse.append(res.methods["vb"].armse)
        t1 = time.perf_counter()

        per_epoch_vehicle_ms = (t1 - t0) * 1e3 / (seeds * n_epochs * n_usvs)
        rows.append({
            "max_vb_iter": mi,
            "armse_mean": float(np.mean(armse)),
            "armse_std": float(np.std(armse, ddof=1)),
            "per_epoch_vehicle_ms": float(per_epoch_vehicle_ms),
        })
        print(f"max_iter={mi}: ARMSE {np.mean(armse):.3f} +- "
              f"{np.std(armse, ddof=1):.3f} m | "
              f"{per_epoch_vehicle_ms:.2f} ms/epoch/vehicle", flush=True)

    (OUT / "s7_vb_iter_ablation.json").write_text(
        json.dumps({"compass_deg": COMPASS_DEG, "n_trials": seeds,
                    "rows": rows}, indent=2))
    print("wrote", OUT / "s7_vb_iter_ablation.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    sweep(args.seeds)
