"""
R2 -- Post-hoc power / MDES for the S2/S3 "no-difference" (equivalence)
contrasts, computed from the saved per-seed ARMSE arrays (no re-run needed).

For each equivalence-relevant contrast we report:
  * the observed within-pair standard deviation of the difference;
  * the minimum detectable effect size (MDES) in metres at 80% power, M=30;
  * the observed mean difference (for scale).

This lets the manuscript state that the equivalence claims are backed by TOST
(not by a failed null), and that the MDES is a small fraction of the 0.5 m
equivalence margin, so the sample is adequate for the equivalence verdicts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "outputs"
ALPHA = 0.05
POWER = 0.8


def mdes_d(n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    df = n - 1
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) / np.sqrt(n)


def achieved_power(d: float, n: int, alpha: float = ALPHA) -> float:
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = d * np.sqrt(n)
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def contrast(a, b):
    d = a - b
    sd = d.std(ddof=1)
    d_obs = d.mean() / sd if sd > 0 else 0.0
    return d.mean(), sd, d_obs


def main() -> None:
    s2 = json.loads((OUT / "s2_scheduling_results.json").read_text())
    s3 = json.loads((OUT / "s3_anchor_results.json").read_text())

    print("=" * 78)
    print("R2 post-hoc power / MDES (M=30, two-sided paired t, alpha=0.05)")
    print("=" * 78)

    # --- S2: CQM vs nearest (the key no-difference claims), B=2 ---
    print("\nS2 (B=2): CQM vs nearest equivalence contrasts")
    for c in s2["cells"]:
        if c["budget"] != 2:
            continue
        ps = c["per_seed"]
        n = c["n_trials"]
        plr = c["plr"]
        # VB filter: CQM scheduler vs nearest
        m, sd, d_obs = contrast(np.array(ps["cqa_vbcif"]),
                                np.array(ps["vb_ckf"]))
        mdes_m = mdes_d(n) * sd
        print(f"  PLR={plr*100:>3.0f}%  vb_cqm-vb_near: diff={m:+.3f} m  "
              f"sd={sd:.3f}  MDES={mdes_m:.3f} m  power(d_obs={d_obs:+.2f})="
              f"{achieved_power(abs(d_obs), n):.3f}")
        # CKF filter: oracle vs nearest
        m, sd, d_obs = contrast(np.array(ps["ckf_oracle"]),
                                np.array(ps["ckf_cl"]))
        mdes_m = mdes_d(n) * sd
        print(f"  PLR={plr*100:>3.0f}%  ckf_oracle-vs_near: diff={m:+.3f} m  "
              f"sd={sd:.3f}  MDES={mdes_m:.3f} m  power(d_obs={d_obs:+.2f})="
              f"{achieved_power(abs(d_obs), n):.3f}")

    # --- S3: anchor-aware mutual equivalence, B=2 ---
    print("\nS3 (B=2): anchor-aware mutual-equivalence contrasts (vs CQM)")
    for c in s3["cells"]:
        if c["budget"] != 2 or c.get("n_anchors") not in (None, 2):
            if c["budget"] != 2:
                continue
        ps = c["per_seed"]
        n = c["n_trials"]
        plr = c["plr"]
        for m_key in ("ckf_anchor_first", "ckf_oracle"):
            m, sd, d_obs = contrast(np.array(ps[m_key]),
                                    np.array(ps["ckf_cqm"]))
            mdes_m = mdes_d(n) * sd
            print(f"  PLR={plr*100:>3.0f}%  {m_key}-vs_cqm: diff={m:+.3f} m  "
                  f"sd={sd:.3f}  MDES={mdes_m:.3f} m  power(d_obs={d_obs:+.2f})="
                  f"{achieved_power(abs(d_obs), n):.3f}")


if __name__ == "__main__":
    main()
