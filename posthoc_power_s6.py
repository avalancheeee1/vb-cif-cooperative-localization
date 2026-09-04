"""
R3 -- Post-hoc TOST power / MDES for the S6 random-topology equivalence claims.

The reviewer asked that the M=30 random-topology TOST verdicts be backed by a
power statement.  For each compass level we report, from the saved per-seed
ARMSE arrays:

  * the within-pair standard deviation of the VB-CIF - CKF-CL difference;
  * the minimum detectable effect size (MDES) in metres at 80% power, M=30;
  * the achieved power of a two-sided paired t-test to detect a difference of
    exactly the 0.5 m equivalence margin.

If the MDES is a small fraction of the 0.5 m margin, the sample is adequate for
the equivalence verdict (an effect large enough to matter would have been
detected with high probability).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "outputs"
ALPHA = 0.05
POWER = 0.8
MARGIN = 0.5


def mdes_d(n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    df = n - 1
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) / np.sqrt(n)


def power_at(delta: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = (delta / sd) * np.sqrt(n)
    if ncp > 12.0:            # noncentral-t saturates numerically; power ~ 1
        return 1.0
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def main() -> None:
    s6 = json.loads((OUT / "s6_random_topology_results.json").read_text())

    print("=" * 74)
    print("R3 post-hoc power / MDES (S6 random topology, M=30, alpha=0.05)")
    print("=" * 74)
    for lv in s6["levels"]:
        ps = lv["per_seed"]
        vb = np.array(ps["vb"])
        ckf = np.array(ps["ckf"])
        d = vb - ckf
        n = len(d)
        sd = d.std(ddof=1)
        mdes_m = mdes_d(n) * sd
        pw = power_at(MARGIN, sd, n)
        print(f"  {lv['compass_deg']:>5} deg: VB-CKF = {d.mean():+.3f} m, "
              f"sd = {sd:.3f} m, MDES(80%, M={n}) = {mdes_m:.3f} m, "
              f"power(detect 0.5 m) = {pw:.4f}")


if __name__ == "__main__":
    main()
