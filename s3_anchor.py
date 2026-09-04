"""
M9 -- S3 anchor experiment: anchor observability under CQM link scheduling.

Adds ``n_anchors`` GNSS-aided anchor USVs to the swarm (the first ``n_anchors``
vehicles).  Anchors have absolute-position measurements, so a cooperative link
to an anchor carries the only absolute position information a GNSS-denied
follower can use to pin down its own position (resolving the otherwise
unobservable common-mode translation).  Under an interrogation budget B, each
follower must choose which links to interrogate -- anchors (absolute value, but
often far/lossy) vs followers (relative geometry, cheap and nearby).

All methods use the same CKF filter; only the scheduling policy varies:

  ckf_nearest       CKF, nearest-B            (baseline)
  ckf_random        CKF, random-B             (baseline)
  ckf_anchor_first  CKF, anchors-then-nearest (naive anchor-aware baseline)
  ckf_cqm           CKF, CQM + anchor bonus   (proposed)
  ckf_oracle        CKF, true-noise + anchor bonus (upper bound)

The CQM/oracle greedy multiplies the information weight of an anchor link by
``anchor_value_weight`` (4.0), so it preferentially selects the absolute
reference when the budget is tight.  Each cell is a strictly paired run; every
pairwise contrast carries a paired t-test (95% CI), TOST (0.5 m margin), and a
Bonferroni correction across the cells at a fixed budget (family size 4 for
B=2).

Results -> outputs/s3_anchor_results.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src.runner_sched import S3_METHODS, run_trial_sched
from s1_stats import bonferroni, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

N_ANCHORS = 2                       # first two vehicles are GNSS-aided anchors
ANCHORS = frozenset(range(N_ANCHORS))
REF = "ckf_cqm"                     # proposed anchor-aware scheduling

# (budget, plr) cells: budget scaling at the lossy PLR=0.4, and a PLR
# robustness sweep at the representative budget B=2.
CELLS = (
    (1, 0.4), (2, 0.4), (3, 0.4),
    (2, 0.0), (2, 0.2), (2, 0.6),
)


def cell(budget: int, plr: float, seeds: int) -> dict:
    """Run ``seeds`` strictly paired trials at one (budget, plr) cell."""
    acc = {m: [] for m in S3_METHODS}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial_sched(DEFAULT_PARAMS, rng, plr, budget, S3_METHODS,
                              anchors=ANCHORS)
        for m in S3_METHODS:
            acc[m].append(res.methods[m].armse)

    arr = {m: np.array(acc[m]) for m in S3_METHODS}
    rep = report_matrix(arr, S3_METHODS)

    # Backward-compatible alias: each policy vs the proposed CQM policy.
    paired_vs_cqm = {m: rep["pairwise"][f"{m}_vs_{REF}"]
                     for m in S3_METHODS if m != REF}

    return {
        "budget": budget,
        "plr": plr,
        "n_anchors": N_ANCHORS,
        "n_trials": seeds,
        "summary": rep["summary"],
        "per_seed": rep["per_seed"],
        "pairwise": rep["pairwise"],
        "paired_vs_cqm": paired_vs_cqm,
    }


def main(seeds=30):
    cells = [cell(b, p, seeds) for b, p in CELLS]

    # Bonferroni: for each budget, correct each contrast across the cells.
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

    (OUT / "s3_anchor_results.json").write_text(
        json.dumps({"cells": cells}, indent=2))

    for c in cells:
        b, p = c["budget"], c["plr"]
        print(f"\n=== B={b}  PLR={p * 100:.0f}%  (n={c['n_trials']}, "
              f"{c['n_anchors']} anchors) ===")
        for m in S3_METHODS:
            s = c["summary"][m]
            print(f"  {m:>16}: ARMSE {s['armse_mean']:.3f} +- "
                  f"{s['armse_std']:.3f} m")
        for m, v in c["paired_vs_cqm"].items():
            imp = -100.0 * v["mean_diff"] / max(c["summary"][m]["armse_mean"], 1e-9)
            tost = v["tost"]
            print(f"    vs {m:>16}: {v['mean_diff']:+.3f} m ({imp:+.1f}%) "
                  f"t={v['t']:+.2f} p={v['p']:.2e} bonf_p={v.get('bonf_p', float('nan')):.2e} "
                  f"d={v['cohens_d']:+.2f} TOST equiv={tost['equiv']}")
    print(f"\nwrote {OUT / 's3_anchor_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
