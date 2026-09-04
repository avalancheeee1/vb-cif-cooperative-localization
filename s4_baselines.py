"""
M16 -- Robust / adaptive baseline comparison.

Answers two questions the reviewers raised about the residual-dominance story:

  (Q1) Is the VB-CIF failure specific to its inverse-Wishart prior, or is it
       fundamental to *residual-based* noise adaptation?  IAE (Mehra /
       Sage-Husa-type innovation-covariance matching) also estimates R from the
       innovation residual and, exactly like VB-CIF, does NOT debias the
       *neighbor's* dead-reckoning error (it subtracts only the receiver's own
       H P H^T).  Under residual dominance it should therefore ALSO inflate its
       learned R and collapse the gain -- if so, the failure generalizes.

  (Q2) How does the guard compare with a standard *robust* (non-adaptive)
       baseline?  Huber-robust CKF-CL downweights large innovations instead of
       learning R, so it should be robust without any VB machinery.

Sweeps the compass (heading) noise -- the lever-arm knob that drives residual
dominance -- over {0.5, 2, 5} deg and runs five methods on identical
strictly-paired inputs per trial:

  ckf     fixed-covariance CKF-CL (reference)
  vb      VB-CIF (inverse-Wishart adaptive R; fails under dominance)
  gvb     G-VB-CIF (residual-dominance NIS guard -> CKF-CL fallback)
  iae     IAE adaptive R (persistent innovation-matching, lam=0.1)
  huber   Huber-robust CKF-CL (outlier downweighting, c=1.345)

Results -> outputs/s4_baselines_results.json (same schema as S1).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from params import DEFAULT_PARAMS
from src.runner import run_trial
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

COMPASS_DEG = (0.5, 2.0, 5.0)
METHODS = ("ckf", "vb", "gvb", "iae", "huber")
HEADLINE = ("vb_vs_ckf", "gvb_vs_ckf", "iae_vs_ckf", "huber_vs_ckf")


def sweep_level(compass_deg: float, seeds: int) -> dict:
    """Run ``seeds`` strictly paired trials at one compass-noise level."""
    params = replace(
        DEFAULT_PARAMS,
        sensors=replace(DEFAULT_PARAMS.sensors,
                        compass_sigma=float(np.deg2rad(compass_deg))))

    acc = {m: [] for m in METHODS}
    guard_rate = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial(params, rng, 0.0, METHODS)
        for m in METHODS:
            acc[m].append(res.methods[m].armse)
        guard_rate.append(res.methods["gvb"].guard_disable_rate)

    arr = {m: np.array(acc[m]) for m in METHODS}
    rep = report_matrix(arr, METHODS)
    paired = {k: rep["pairwise"][k] for k in HEADLINE}

    return {
        "compass_deg": compass_deg,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "paired": paired,
        "pairwise": rep["pairwise"],
        "guard_disable_rate_mean": float(np.mean(guard_rate)),
        "guard_disable_rate_std": float(np.std(guard_rate, ddof=1)),
    }


def main(seeds: int = 30) -> None:
    levels = [sweep_level(c, seeds) for c in COMPASS_DEG]

    # Bonferroni: each headline contrast is a family across the three levels.
    for key in HEADLINE:
        ps = [lv["pairwise"][key]["p"] for lv in levels]
        for lv, p_corr in zip(levels, bonferroni(ps)):
            lv["pairwise"][key]["bonf_p"] = p_corr
            lv["pairwise"][key]["family_size"] = len(ps)
            lv["paired"][key] = lv["pairwise"][key]

    (OUT / "s4_baselines_results.json").write_text(
        json.dumps({"levels": levels}, indent=2))

    for lv in levels:
        c = lv["compass_deg"]
        s = lv["summary"]
        print(f"\n=== compass sigma = {c:.1f} deg (n={lv['n_trials']}) ===")
        for m in METHODS:
            print(f"  {m:>6}: ARMSE {s[m]['armse_mean']:.3f} +- "
                  f"{s[m]['armse_std']:.3f} m")
        for k, v in lv["paired"].items():
            t = v["tost"]
            print(f"  {k:>14}: diff {v['mean_diff']:+.3f} m "
                  f"(p={v['p']:.2e}, "
                  f"bonf_p={v.get('bonf_p', float('nan')):.2e}, "
                  f"d={v['cohens_d']:+.2f}, "
                  f"TOST p={t['tost_p']:.2e} equiv={t['equiv']})")
        print(f"  guard disable rate: {lv['guard_disable_rate_mean']:.3f} "
              f"+- {lv['guard_disable_rate_std']:.3f}")
    print(f"\nwrote {OUT / 's4_baselines_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
