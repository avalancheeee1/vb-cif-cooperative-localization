"""
M17b -- Random-topology robustness: is the residual-dominance failure an
artifact of the collinear lawnmower geometry?

S1 swept the compass noise under the *collinear* boustrophedon (vehicles in a
line of parallel survey lanes).  A reviewer could reasonably worry that the
VB-CIF penalty is a peculiarity of that regular geometry (e.g. near-aligned
links).  Here we re-run the identical S1 sweep, same seeds, same compass levels
{0.5, 2, 5} deg, but with each vehicle scattered uniformly at random over the
same survey footprint and following an independent random waypoint sequence
(``dyn.generate_trajectory_random``).  Everything else -- the per-dt process
noise, the shared verbatim control, the five filters -- is unchanged.

If the VB-vs-CKF penalty reappears at 2-5 deg and the guard recovers CKF-CL,
the mechanism is a property of residual dominance (the lever-arm of the
heading error), not of the collinear geometry.

Results -> outputs/s6_random_topology_results.json (same schema as S1).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src.dynamics import generate_trajectory_random
from src.runner import METHODS_S1, run_trial
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

COMPASS_DEG = (0.5, 2.0, 5.0)
HEADLINE = ("vb_vs_ckf", "gvb_vs_vb", "gvb_vs_ckf")


def sweep_level(compass_deg: float, seeds: int) -> dict:
    params = replace(
        DEFAULT_PARAMS,
        sensors=replace(DEFAULT_PARAMS.sensors,
                        compass_sigma=float(np.deg2rad(compass_deg))))

    acc = {m: [] for m in METHODS_S1}
    guard_rate = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial(params, rng, 0.0, METHODS_S1,
                        trajectory_fn=generate_trajectory_random)
        for m in METHODS_S1:
            acc[m].append(res.methods[m].armse)
        guard_rate.append(res.methods["gvb"].guard_disable_rate)

    arr = {m: np.array(acc[m]) for m in METHODS_S1}
    rep = report_matrix(arr, METHODS_S1)

    return {
        "compass_deg": compass_deg,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "pairwise": rep["pairwise"],
        "paired": {k: rep["pairwise"][k] for k in HEADLINE},
        "sign": {
            "vb_gt_ckf": int((arr["vb"] > arr["ckf"]).sum()),
            "gvb_lt_vb": int((arr["gvb"] < arr["vb"]).sum()),
        },
        "guard_disable_rate_mean": float(np.mean(guard_rate)),
        "guard_disable_rate_std": float(np.std(guard_rate, ddof=1)),
    }


def main(seeds: int = 30) -> None:
    levels = [sweep_level(c, seeds) for c in COMPASS_DEG]

    for key in HEADLINE:
        ps = [lv["pairwise"][key]["p"] for lv in levels]
        for lv, p_corr in zip(levels, bonferroni(ps)):
            lv["pairwise"][key]["bonf_p"] = p_corr
            lv["pairwise"][key]["family_size"] = len(ps)
            lv["paired"][key] = lv["pairwise"][key]

    (OUT / "s6_random_topology_results.json").write_text(
        json.dumps({"topology": "random", "levels": levels}, indent=2))

    for lv in levels:
        c = lv["compass_deg"]
        s = lv["summary"]
        print(f"\n=== random topology, compass sigma = {c:.1f} deg "
              f"(n={lv['n_trials']}) ===")
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
    print(f"\nwrote {OUT / 's6_random_topology_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
