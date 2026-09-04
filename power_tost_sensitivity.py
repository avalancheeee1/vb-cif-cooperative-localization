"""
M15 -- Statistical robustness for the revision: (a) TOST sensitivity to the
equivalence margin, and (b) a paired-test power analysis, computed from the
per-seed ARMSE arrays already saved in the S1 JSON (no re-run required).

Prints a compact table the manuscript can cite verbatim; writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from s1_stats import tost

OUT = Path(__file__).resolve().parent / "outputs" / "s1_mechanism_results.json"
ALPHA = 0.05
MARGINS = (0.3, 0.5, 0.7)
POWER = 0.8


def mdes_cohens_d(n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable |Cohen's d| for a two-sided paired t-test."""
    df = n - 1
    t_alpha = stats.t.ppf(1 - alpha / 2, df)     # two-sided critical value
    t_beta = stats.t.ppf(power, df)              # one-sided value for power
    return (t_alpha + t_beta) / np.sqrt(n)


def achieved_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Power of a two-sided paired t-test for effect size d and sample n."""
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = d * np.sqrt(n)                         # noncentrality parameter
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def main() -> None:
    data = json.loads(OUT.read_text())
    print("=" * 72)
    print("TASK M15 -- statistical robustness (from saved S1 per-seed data)")
    print("=" * 72)

    # ---- (a) TOST sensitivity ----
    print("\n(a) TOST equivalence sensitivity (margin 0.3 / 0.5 / 0.7 m)\n")
    for lv in data["levels"]:
        c = lv["compass_deg"]
        ps = lv["per_seed"]
        print(f"compass {c:.1f} deg (n={lv['n_trials']}):")
        for key, a_name, b_name in [
            ("vb_vs_ckf", "vb", "ckf"),
            ("gvb_vs_ckf", "gvb", "ckf"),
        ]:
            a = np.array(ps[a_name])
            b = np.array(ps[b_name])
            row = []
            for m in MARGINS:
                r = tost(a, b, margin=m, alpha=ALPHA)
                row.append(f"m={m}: p={r['tost_p']:.2e} {'equiv' if r['equiv'] else 'NOT-equiv'}")
            print(f"  {key:12s}  " + "  |  ".join(row))
        print()

    # ---- (b) power analysis ----
    print("(b) Power analysis (two-sided paired t-test, alpha=0.05)\n")
    print(f"{'n':>4s}  {'MDES (Cohen d)':>15s}")
    for n in (20, 30, 50, 100):
        print(f"{n:>4d}  {mdes_cohens_d(n):15.3f}")
    print()

    # MDES in meters using the observed within-pair SD of each contrast.
    print("MDES in metres (observed within-pair SD) and achieved power:\n")
    for lv in data["levels"]:
        c = lv["compass_deg"]
        ps = lv["per_seed"]
        n = lv["n_trials"]
        print(f"compass {c:.1f} deg (n={n}):")
        for key, a_name, b_name in [
            ("vb_vs_ckf", "vb", "ckf"),
            ("gvb_vs_ckf", "gvb", "ckf"),
            ("gvb_vs_vb", "gvb", "vb"),
        ]:
            d = np.array(ps[a_name]) - np.array(ps[b_name])
            sd = d.std(ddof=1)
            d_obs = d.mean() / sd
            mdes_m = mdes_cohens_d(n) * sd
            pow_obs = achieved_power(abs(d_obs), n)
            print(f"  {key:12s} d_obs={d_obs:+.2f} sd={sd:.3f} m "
                  f"-> MDES={mdes_m:.3f} m, power(d_obs)={pow_obs:.3f}")
        print()


if __name__ == "__main__":
    main()
