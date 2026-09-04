"""
M7.5 -- Honest failure-mechanism experiment: residual dominance under a
degraded compass (dead-reckoning heading sensor).

This is the corrected successor to the paper's original "common-mode
unobservability" claim.  That claim does not reproduce: a *common-mode*
position drift cancels identically in the relative range/bearing measurements
(|p_i - p_j| and atan2(dy, dx) - psi are both invariant to a shared translation)
and is therefore invisible to the residual, so it cannot bias VB-CIF.

The reproducible mechanism is instead *residual dominance*: VB-CIF estimates
the measurement-noise covariance from the innovation residual, and the VB-M
debias only subtracts the receiver's own state contribution H P H^T -- the
*neighbor's* estimation error H_j e_j e_j^T H_j^T is not removed.  When the
neighbor's dead-reckoning error is large, that error dominates the residual,
VB-CIF's noise estimate is biased upward, VB-CIF under-weights the (still
informative) measurements, and it degrades below fixed-covariance CKF-CL.

The heading (compass) is the dominant source of dead-reckoning error (a heading
error acts as a lever arm that converts distance travelled into position
error).  This experiment sweeps the compass noise sigma_c over {0.5, 2, 5} deg
-- 0.5 deg is a high-end AHRS, 2-5 deg is a low-cost MEMS magnetometer -- and
shows the VB-CIF penalty appears and grows with sigma_c while CKF-CL is flat
and the residual-dominance guard (G-VB-CIF) recovers CKF-CL accuracy.

Results are written to outputs/s1_mechanism_results.json.  Every pairwise
contrast carries a two-sided paired t-test (true 95% CI), a TOST equivalence
test (margin 0.5 m), and, for the three headline contrasts, a Bonferroni
correction across the three compass levels (family size 3).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS, SensorParams
from src.runner import METHODS_S1, run_trial
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

COMPASS_DEG = (0.5, 2.0, 5.0)

# Headline contrasts the manuscript reports; each is Bonferroni-corrected
# across the three compass levels (family size 3).
HEADLINE = ("vb_vs_ckf", "gvb_vs_vb", "gvb_vs_ckf")


def sweep_level(compass_deg: float, seeds: int) -> dict:
    """Run ``seeds`` strictly paired trials at one compass-noise level."""
    params = replace(
        DEFAULT_PARAMS,
        sensors=replace(DEFAULT_PARAMS.sensors,
                        compass_sigma=float(np.deg2rad(compass_deg))))

    acc = {m: [] for m in METHODS_S1}
    guard_rate = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial(params, rng, 0.0, METHODS_S1)
        for m in METHODS_S1:
            acc[m].append(res.methods[m].armse)
        guard_rate.append(res.methods["gvb"].guard_disable_rate)

    arr = {m: np.array(acc[m]) for m in METHODS_S1}
    rep = report_matrix(arr, METHODS_S1)

    sign = {
        "vb_gt_ckf": int((arr["vb"] > arr["ckf"]).sum()),
        "gvb_lt_vb": int((arr["gvb"] < arr["vb"]).sum()),
    }
    # Readable aliases for the three headline contrasts (subset of ``pairwise``).
    paired = {k: rep["pairwise"][k] for k in HEADLINE}

    return {
        "compass_deg": compass_deg,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "paired": paired,
        "pairwise": rep["pairwise"],
        "sign": sign,
        "guard_disable_rate_mean": float(np.mean(guard_rate)),
        "guard_disable_rate_std": float(np.std(guard_rate, ddof=1)),
    }


def main(seeds=30):
    levels = [sweep_level(c, seeds) for c in COMPASS_DEG]

    # Bonferroni correction: each headline contrast is a family across the
    # three compass levels.
    for key in HEADLINE:
        ps = [lv["pairwise"][key]["p"] for lv in levels]
        for lv, p_corr in zip(levels, bonferroni(ps)):
            lv["pairwise"][key]["bonf_p"] = p_corr
            lv["pairwise"][key]["family_size"] = len(ps)
            lv["paired"][key] = lv["pairwise"][key]

    (OUT / "s1_mechanism_results.json").write_text(
        json.dumps({"levels": levels}, indent=2))

    for lv in levels:
        c = lv["compass_deg"]
        s = lv["summary"]
        print(f"\n=== compass sigma = {c:.1f} deg (n={lv['n_trials']}) ===")
        for m in METHODS_S1:
            print(f"  {m:>4}: ARMSE {s[m]['armse_mean']:.3f} +- "
                  f"{s[m]['armse_std']:.3f} m")
        for k, v in lv["paired"].items():
            tost = v["tost"]
            print(f"  {k:>12}: diff {v['mean_diff']:+.3f} m "
                  f"(t={v['t']:+.2f}, p={v['p']:.2e}, "
                  f"bonf_p={v.get('bonf_p', float('nan')):.2e}, "
                  f"d={v['cohens_d']:+.2f}, "
                  f"CI95 [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}], "
                  f"TOST p={tost['tost_p']:.2e} equiv={tost['equiv']})")
        print(f"  VB > CKF: {lv['sign']['vb_gt_ckf']}/{lv['n_trials']}; "
              f"GVB < VB: {lv['sign']['gvb_lt_vb']}/{lv['n_trials']}")
        print(f"  guard disable rate: {lv['guard_disable_rate_mean']:.3f} "
              f"+- {lv['guard_disable_rate_std']:.3f}")
    print(f"\nwrote {OUT / 's1_mechanism_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
