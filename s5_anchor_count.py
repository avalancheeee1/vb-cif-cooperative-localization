"""
M17a -- Anchor-count sensitivity: does the anchor-aware scheduling advantage
persist (and how does it scale) as the number of GNSS-aided anchors changes?

S3 used two anchors.  Here we sweep N_ANCHORS in {1, 2, 3} at the representative
lossy condition (B=2, PLR=0.4) and, at each count, compare the same five
scheduling policies on strictly paired inputs.  The question is whether the
anchor-aware policies (CQM with the anchor bonus, anchor-first, oracle) retain
their advantage over the anchor-blind nearest/random policies as the number of
absolute references changes, and how the magnitude of that advantage scales.

Results -> outputs/s5_anchor_count_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from params import DEFAULT_PARAMS
from src.runner_sched import S3_METHODS, run_trial_sched
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

ANCHOR_COUNTS = (1, 2, 3)
BUDGET = 2
PLR = 0.4
REF = "ckf_cqm"


def cell(n_anchors: int, seeds: int) -> dict:
    anchors = frozenset(range(n_anchors))
    acc = {m: [] for m in S3_METHODS}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial_sched(DEFAULT_PARAMS, rng, PLR, BUDGET, S3_METHODS,
                              anchors=anchors)
        for m in S3_METHODS:
            acc[m].append(res.methods[m].armse)

    arr = {m: np.array(acc[m]) for m in S3_METHODS}
    rep = report_matrix(arr, S3_METHODS)
    return {
        "n_anchors": n_anchors,
        "budget": BUDGET,
        "plr": PLR,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "pairwise": rep["pairwise"],
        "vs_cqm": {m: rep["pairwise"][f"{m}_vs_{REF}"]
                   for m in S3_METHODS if m != REF},
    }


def main(seeds: int = 30) -> None:
    cells = [cell(a, seeds) for a in ANCHOR_COUNTS]

    # Bonferroni: each non-reference contrast is a family across the three
    # anchor counts (nearest/random/anchor_first/oracle each vs cqm).
    vs_keys = [f"{m}_vs_{REF}" for m in S3_METHODS if m != REF]
    for key in vs_keys:
        ps = [c["pairwise"][key]["p"] for c in cells]
        for c, pc in zip(cells, bonferroni(ps)):
            c["pairwise"][key]["bonf_p"] = pc
            c["pairwise"][key]["family_size"] = len(ps)

    (OUT / "s5_anchor_count_results.json").write_text(
        json.dumps({"cells": cells}, indent=2))

    for c in cells:
        a = c["n_anchors"]
        print(f"\n=== {a} anchor(s), B={BUDGET}, PLR={PLR * 100:.0f}% "
              f"(n={c['n_trials']}) ===")
        for m in S3_METHODS:
            s = c["summary"][m]
            print(f"  {m:>18}: ARMSE {s['armse_mean']:.3f} +- "
                  f"{s['armse_std']:.3f} m")
        for m, v in c["vs_cqm"].items():
            imp = -100.0 * v["mean_diff"] / max(c["summary"][m]["armse_mean"], 1e-9)
            print(f"    {m:>18} vs cqm: {v['mean_diff']:+.3f} m ({imp:+.1f}%) "
                  f"p={v['p']:.2e} bonf_p={v.get('bonf_p', float('nan')):.2e} "
                  f"TOST_equiv={v['tost']['equiv']}")
    print(f"\nwrote {OUT / 's5_anchor_count_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
