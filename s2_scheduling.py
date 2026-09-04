"""
M8 -- S2 scheduling experiment: CQM-driven link selection under a budget.

Compares (filter x scheduling-policy) combinations across an interrogation
budget B in {1, 2, 7} and packet-loss rate PLR in {0, 0.2, 0.4, 0.6}:

  ckf_cl      CKF, nearest-B            (baseline filter + baseline schedule)
  ckf_random  CKF, random-B             (baseline filter + random schedule)
  vb_ckf      VB-CIF, nearest-B         (proposed filter + baseline schedule)
  cqa_vbcif   VB-CIF, CQM-driven B      (proposed: filter + scheduling)
  ckf_oracle  CKF, true-noise-B         (unachievable upper bound)

The hypothesis: under a tight budget (B=1,2) the CQM policy (``cqa_vbcif``)
recovers most of the oracle's information gain, beating the nearest/random
baselines, while at B=7 (unconstrained) all policies coincide and the residual
VB-vs-CKF difference reflects the filter alone.  Each cell is a strictly paired
Monte-Carlo run (shared truth/channel/noise/perturbation across methods).  Every
pairwise contrast carries a paired t-test (95% CI), TOST (0.5 m margin), and a
Bonferroni correction across the packet-loss levels at a fixed budget (family
size 4 for B=1,2).

Results -> outputs/s2_scheduling_results.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src.runner_sched import SCHED_METHODS, run_trial_sched
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

REF = "cqa_vbcif"          # the proposed method; all baselines compared to it

# Focused grid: budget scaling (B=1,2) across the PLR sweep, plus the
# unconstrained B=7 sanity check (all policies coincide) at one lossless cell.
CELLS = (
    (1, 0.0), (1, 0.2), (1, 0.4), (1, 0.6),
    (2, 0.0), (2, 0.2), (2, 0.4), (2, 0.6),
    (7, 0.0),
)


def cell(budget: int, plr: float, seeds: int) -> dict:
    """Run ``seeds`` strictly paired trials at one (budget, plr) cell."""
    acc = {m: [] for m in SCHED_METHODS}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial_sched(DEFAULT_PARAMS, rng, plr, budget, SCHED_METHODS)
        for m in SCHED_METHODS:
            acc[m].append(res.methods[m].armse)

    arr = {m: np.array(acc[m]) for m in SCHED_METHODS}
    rep = report_matrix(arr, SCHED_METHODS)

    # Backward-compatible alias: each baseline vs the proposed CQM method.
    paired_vs_cqa = {m: rep["pairwise"][f"{m}_vs_{REF}"]
                     for m in SCHED_METHODS if m != REF}

    return {
        "budget": budget,
        "plr": plr,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "pairwise": rep["pairwise"],
        "paired_vs_cqa": paired_vs_cqa,
    }


def main(seeds=30):
    cells = [cell(b, p, seeds) for b, p in CELLS]

    # Bonferroni: for each budget, correct each contrast across the PLR sweep.
    by_budget = defaultdict(list)
    for c in cells:
        by_budget[c["budget"]].append(c)
    for budget, cs in by_budget.items():
        m = len(cs)
        for key in cs[0]["pairwise"]:
            ps = [c["pairwise"][key]["p"] for c in cs]
            for c, pc in zip(cs, bonferroni(ps)):
                c["pairwise"][key]["bonf_p"] = pc
                c["pairwise"][key]["family_size"] = m

    (OUT / "s2_scheduling_results.json").write_text(
        json.dumps({"cells": cells}, indent=2))

    for c in cells:
        b, p = c["budget"], c["plr"]
        print(f"\n=== B={b}  PLR={p * 100:.0f}%  (n={c['n_trials']}) ===")
        for m in SCHED_METHODS:
            s = c["summary"][m]
            print(f"  {m:>12}: ARMSE {s['armse_mean']:.3f} +- "
                  f"{s['armse_std']:.3f} m")
        for m, v in c["paired_vs_cqa"].items():
            imp = -100.0 * v["mean_diff"] / max(c["summary"][m]["armse_mean"], 1e-9)
            tost = v["tost"]
            print(f"    vs {m:>12}: {v['mean_diff']:+.3f} m ({imp:+.1f}%) "
                  f"t={v['t']:+.2f} p={v['p']:.2e} bonf_p={v.get('bonf_p', float('nan')):.2e} "
                  f"d={v['cohens_d']:+.2f} TOST equiv={tost['equiv']}")
    print(f"\nwrote {OUT / 's2_scheduling_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
