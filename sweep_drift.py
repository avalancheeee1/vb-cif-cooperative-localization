"""
Diagnostic sweep: how does the VB-vs-CKF penalty scale with the unobservable
drift (process noise sigma_pos and unmodeled disturbance unmodeled_sigma_pos)?

The residual-dominance mechanism predicts that VB-CIF's online R estimate is
biased upward by the *neighbor's* estimation error (which lives partly in the
unobservable common-mode subspace).  Larger drift -> larger neighbor error ->
larger VB penalty.  This script quantifies that scaling on the corrected
lawnmower framework so we can see whether the paper's mechanism has a
pronounced regime, or only a modest one.

Usage: python sweep_drift.py --seeds 10
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from s1_repro import METHODS, run_trial_metrics


def run_cfg(sigma_pos: float, unmodeled: float, seeds: int) -> dict:
    params = DEFAULT_PARAMS
    params = replace(params,
                     noise=replace(params.noise,
                                   sigma_pos=sigma_pos,
                                   unmodeled_sigma_pos=unmodeled))
    acc = {m: [] for m in METHODS}
    vb_gt_ckf = 0
    for s in range(seeds):
        r = run_trial_metrics(s, params)
        for m in METHODS:
            acc[m].append(r[m]["mae"])
        if r["vb"]["mae"] > r["ckf"]["mae"]:
            vb_gt_ckf += 1
    mean = {m: float(np.mean(acc[m])) for m in METHODS}
    gap = mean["vb"] - mean["ckf"]
    return {"gap": gap, "vb_gt_ckf": vb_gt_ckf, "mae": mean}


def main(seeds=10):
    print(f"{'sigma_pos':>10} {'unmodeled':>10} | {'ekf':>7} {'ckf':>7} "
          f"{'vb':>7} {'gvb':>7} | {'VB-CKF':>8} {'VB>CKF':>8}")
    for sigma_pos in (0.1, 0.2, 0.3, 0.5):
        for unmodeled in (0.0, 0.05, 0.1, 0.2):
            r = run_cfg(sigma_pos, unmodeled, seeds)
            m = r["mae"]
            print(f"{sigma_pos:>10.2f} {unmodeled:>10.2f} | "
                  f"{m['ekf']:>7.3f} {m['ckf']:>7.3f} "
                  f"{m['vb']:>7.3f} {m['gvb']:>7.3f} | "
                  f"{r['gap']:>8.3f} {r['vb_gt_ckf']:>7}/{seeds}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    main(args.seeds)
