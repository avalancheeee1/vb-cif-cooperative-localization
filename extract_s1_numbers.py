"""
Extract all S1 numbers needed for the manuscript prose + Table S1 in a
paste-ready form, from a s1_mechanism_results.json (n=30 or n=100).
Usage: python extract_s1_numbers.py [path-to-json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

METHODS = ("ekf", "ckf", "iwcf", "vb", "gvb")
LABEL = {"ekf": "EKF-CL", "ckf": "CKF-CL", "iwcf": "EIF", "vb": "VB-CIF",
         "gvb": "G-VB-CIF"}


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.2f}\\pm{s:.2f}"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "outputs" / "s1_mechanism_results.json")
    data = json.loads(path.read_text())
    print(f"# source: {path.name}\n")

    print("## Table S1 rows (mean \\pm std, ARMSE m)")
    for lv in data["levels"]:
        c = lv["compass_deg"]
        cells = []
        for m in METHODS:
            s = lv["summary"][m]
            cells.append(fmt_mean_std(s["armse_mean"], s["armse_std"]))
        print(f"${c:.1f}^\\circ$ & " + " & ".join(cells) + "\\\\")
    print()

    print("## Headline paired contrasts (per level)")
    for lv in data["levels"]:
        c = lv["compass_deg"]
        n = lv["n_trials"]
        print(f"--- compass {c:.1f} deg (n={n}) ---")
        for key in ("vb_vs_ckf", "gvb_vs_vb", "gvb_vs_ckf"):
            v = lv["pairwise"][key]
            t = v["tost"]
            bonf = v.get("bonf_p")
            bonf_s = f"{bonf:.3g}" if bonf is not None else "n/a"
            print(f"  {key}: diff={v['mean_diff']:+.3f} m, "
                  f"CI95 [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], "
                  f"p={v['p']:.3g}, bonf_p={bonf_s}, d={v['cohens_d']:+.2f}, "
                  f"TOST p={t['tost_p']:.2e} equiv={t['equiv']}")
        print(f"  sign: VB>CKF {lv['sign']['vb_gt_ckf']}/{n}, "
              f"GVB<VB {lv['sign']['gvb_lt_vb']}/{n}")
        print(f"  guard disable rate: {lv['guard_disable_rate_mean']:.3f} "
              f"+- {lv['guard_disable_rate_std']:.3f}")
    print()

    # CKF-CL / EKF-CL ranges across the sweep
    ckf = [lv["summary"]["ckf"]["armse_mean"] for lv in data["levels"]]
    ekf = [lv["summary"]["ekf"]["armse_mean"] for lv in data["levels"]]
    print(f"CKF-CL range across sweep: {min(ckf):.2f}--{max(ckf):.2f} m")
    print(f"EKF-CL range across sweep: {min(ekf):.2f}--{max(ekf):.2f} m")


if __name__ == "__main__":
    main()
