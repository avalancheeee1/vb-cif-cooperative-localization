"""
M11 -- VB-CIF prior-parameter sensitivity (reviewer major-revision item).

The paper's headline claim is that (a) the VB-CIF failure under a degraded
heading sensor and (b) the residual-dominance guard's recovery are properties of
the residual-dominance mechanism, not of the particular inverse-Wishart prior.
To support this, we sweep the two VB prior parameters that govern the fitted
covariance's damping and inflation---the degrees-of-freedom schedule ``tau_max``
and the inflation factor ``eta_inflation``---at the adverse 2 deg compass, where
the failure is clearest, and verify that

  * VB-CIF remains substantially worse than CKF-CL for every setting, and
  * G-VB-CIF remains TOST-equivalent to CKF-CL for every setting.

M = 20 strictly paired trials, matching the other M10 ablations.  Each sweep is
Bonferroni-corrected within its family of three cells.

Results -> outputs/m11_prior_sensitivity_results.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import DEFAULT_PARAMS
from src.runner import run_trial
from s1_stats import bonferroni, full_report

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

N = DEFAULT_PARAMS.stats.n_trials_small          # 20
COMPASS_DEG = 2.0                                # adverse level: failure clear
TAU_MAX = (15, 25, 35)                           # degrees-of-freedom schedule
ETA = (1.0, 2.0, 3.0)                            # inflation factor
METHODS = ("ckf", "vb", "gvb")


def run_cond(tau_max: int, eta: float, seeds: int) -> dict:
    """One strictly paired 2-deg-compass cell at a given (tau_max, eta)."""
    params = replace(
        DEFAULT_PARAMS,
        sensors=replace(DEFAULT_PARAMS.sensors,
                        compass_sigma=float(np.deg2rad(COMPASS_DEG))),
        filter=replace(DEFAULT_PARAMS.filter,
                       tau_max=int(tau_max),
                       eta_inflation=float(eta)))

    acc = {m: [] for m in METHODS}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        res = run_trial(params, rng, 0.0, METHODS)
        for m in METHODS:
            acc[m].append(res.methods[m].armse)

    arr = {m: np.array(acc[m]) for m in METHODS}
    return {
        "tau_max": int(tau_max),
        "eta": float(eta),
        "summary": {m: {"armse_mean": float(arr[m].mean()),
                        "armse_std": float(arr[m].std(ddof=1))}
                    for m in METHODS},
        "vb_vs_ckf": full_report(arr["vb"], arr["ckf"]),
        "gvb_vs_ckf": full_report(arr["gvb"], arr["ckf"]),
    }


def main(seeds: int = N) -> None:
    tau_cells = [run_cond(t, 2.0, seeds) for t in TAU_MAX]
    eta_cells = [run_cond(25, e, seeds) for e in ETA]

    for key in ("vb_vs_ckf", "gvb_vs_ckf"):
        for cells in (tau_cells, eta_cells):
            ps = [c[key]["p"] for c in cells]
            for c, pc in zip(cells, bonferroni(ps)):
                c[key]["bonf_p"] = pc
                c[key]["family_size"] = len(ps)

    out = {"compass_deg": COMPASS_DEG, "n_trials": seeds,
           "tau_sweep": tau_cells, "eta_sweep": eta_cells}
    (OUT / "m11_prior_sensitivity_results.json").write_text(
        json.dumps(out, indent=2))

    print(f"=== M11. VB prior sensitivity @ {COMPASS_DEG} deg (M={seeds}) ===")
    for label, cells in (("tau_max (eta=2.0)", tau_cells),
                         ("eta (tau_max=25)", eta_cells)):
        print(f"\n-- {label} --")
        for c in cells:
            v = c["vb_vs_ckf"]
            g = c["gvb_vs_ckf"]
            s = c["summary"]
            tag = f"tau_max={c['tau_max']}, eta={c['eta']}"
            print(f"  {tag:>24}: ckf={s['ckf']['armse_mean']:.2f} "
                  f"vb={s['vb']['armse_mean']:.2f} gvb={s['gvb']['armse_mean']:.2f}"
                  f" | vb-ckf={v['mean_diff']:+.2f} (p={v['p']:.1e}, "
                  f"bonf={v['bonf_p']:.1e}) | gvb-ckf={g['mean_diff']:+.2f} "
                  f"(p={g['p']:.2f}, bonf={g['bonf_p']:.2f}, "
                  f"TOST={g['tost']['equiv']})")

    print(f"\nwrote {OUT / 'm11_prior_sensitivity_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N)
    args = p.parse_args()
    main(args.seeds)
