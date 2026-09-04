"""
M10 -- Ablation and robustness experiments (paper Sec. IV-D / V).

Three strictly paired sub-experiments:

  A. Guard-threshold (kappa) ablation  -- the residual-dominance guard is the
     core contribution; sweep kappa over a wide band under the adverse 2-deg
     compass and show G-VB-CIF recovers CKF-CL for every reasonable kappa while
     only an extreme (never-trigger) kappa lets the VB-CIF failure leak through.

  B. Process-noise robustness  -- sweep sigma_pos at the nominal 0.5-deg compass
     and show the VB-vs-CKF gap does NOT open (process noise is not the failure
     mechanism; the heading-sensor lever arm is).

  C. CQM indicator ablation  -- drop each CQM indicator (SNR / PRR / ToF) in turn
     and show the full three-indicator CQM gives no advantage over any single
     indicator or over nearest selection.

All runs are strictly paired (identical seed -> identical truth/channel/noise);
each ablation uses the paper's ``n_trials_small`` = 20.  Every pairwise contrast
carries a paired t-test (95% CI), TOST (0.5 m margin), and a Bonferroni
correction across its swept condition (family size 7 for A's kappa, 4 for B
and C).

Results -> outputs/m10_ablation_results.json.
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
from src.runner_sched import run_trial_sched
from s1_stats import bonferroni, full_report, report_matrix

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

N = DEFAULT_PARAMS.stats.n_trials_small          # 20 for ablation / robustness


# ---------------------------------------------------------------------------
# A. Guard-threshold ablation
# ---------------------------------------------------------------------------

KAPPAS = (1.0, 2.0, 3.0, 5.99, 9.0, 15.0, 30.0)
ABL_METHODS = ("ckf", "vb", "gvb")


def guard_kappa_ablation(seeds: int = N, compass_deg: float = 2.0) -> dict:
    params = replace(DEFAULT_PARAMS, sensors=replace(
        DEFAULT_PARAMS.sensors, compass_sigma=float(np.deg2rad(compass_deg))))

    acc = {f"kappa_{k}": [] for k in KAPPAS}
    rate = {f"kappa_{k}": [] for k in KAPPAS}
    ckf, vb = [], []
    for s in range(seeds):
        for k in KAPPAS:
            pk = replace(params, filter=replace(
                params.filter, guard_nis_kappa=float(k)))
            rng = np.random.default_rng(s)
            res = run_trial(pk, rng, 0.0, ABL_METHODS)
            acc[f"kappa_{k}"].append(res.methods["gvb"].armse)
            rate[f"kappa_{k}"].append(res.methods["gvb"].guard_disable_rate)
            # ckf and vb are kappa-independent: grab them once per seed.
            if k == KAPPAS[0]:
                ckf.append(res.methods["ckf"].armse)
                vb.append(res.methods["vb"].armse)

    ckf_arr = np.array(ckf)
    vb_arr = np.array(vb)
    out = {"compass_deg": compass_deg, "n_trials": seeds,
           "ckf_armse_mean": float(ckf_arr.mean()),
           "vb_armse_mean": float(vb_arr.mean()),
           "kappa": []}
    for k in KAPPAS:
        g = np.array(acc[f"kappa_{k}"])
        out["kappa"].append({
            "kappa": k,
            "gvb_armse_mean": float(g.mean()),
            "gvb_armse_std": float(g.std(ddof=1)),
            "guard_disable_rate": float(np.mean(rate[f"kappa_{k}"])),
            "gvb_vs_ckf": full_report(g, ckf_arr),
            "gvb_vs_vb": full_report(g, vb_arr),
        })
    return out


# ---------------------------------------------------------------------------
# B. Process-noise robustness
# ---------------------------------------------------------------------------

SIGMA_POS = (0.05, 0.1, 0.2, 0.4)


def process_noise_robustness(seeds: int = N, compass_deg: float = 0.5) -> dict:
    params = replace(DEFAULT_PARAMS, sensors=replace(
        DEFAULT_PARAMS.sensors, compass_sigma=float(np.deg2rad(compass_deg))))

    out = {"compass_deg": compass_deg, "n_trials": seeds, "levels": []}
    for sp in SIGMA_POS:
        psp = replace(params, noise=replace(params.noise, sigma_pos=float(sp)))
        acc = {m: [] for m in ABL_METHODS}
        for s in range(seeds):
            rng = np.random.default_rng(s)
            res = run_trial(psp, rng, 0.0, ABL_METHODS)
            for m in ABL_METHODS:
                acc[m].append(res.methods[m].armse)
        arr = {m: np.array(acc[m]) for m in ABL_METHODS}
        out["levels"].append({
            "sigma_pos": sp,
            "summary": {m: {"armse_mean": float(arr[m].mean()),
                            "armse_std": float(arr[m].std(ddof=1))}
                        for m in ABL_METHODS},
            "vb_vs_ckf": full_report(arr["vb"], arr["ckf"]),
            "gvb_vs_ckf": full_report(arr["gvb"], arr["ckf"]),
        })
    return out


# ---------------------------------------------------------------------------
# C. CQM indicator ablation
# ---------------------------------------------------------------------------

CQM_VARIANTS = {
    "full": dict(use_snr=True, use_prr=True, use_tofj=True),
    "snr_only": dict(use_snr=True, use_prr=False, use_tofj=False),
    "prr_only": dict(use_snr=False, use_prr=True, use_tofj=False),
    "tofj_only": dict(use_snr=False, use_prr=False, use_tofj=True),
}
VARIANT_ORDER = ("full", "snr_only", "prr_only", "tofj_only", "vb_nearest")


def cqm_indicator_ablation(seeds: int = N, budget: int = 2,
                           plr: float = 0.4) -> dict:
    acc = {k: [] for k in CQM_VARIANTS}
    acc["vb_nearest"] = []
    for s in range(seeds):
        for k, mask in CQM_VARIANTS.items():
            pk = replace(DEFAULT_PARAMS,
                         cqm=replace(DEFAULT_PARAMS.cqm, **mask))
            rng = np.random.default_rng(s)
            res = run_trial_sched(pk, rng, plr, budget, ("cqa_vbcif",))
            acc[k].append(res.methods["cqa_vbcif"].armse)
        rng = np.random.default_rng(s)
        res = run_trial_sched(DEFAULT_PARAMS, rng, plr, budget, ("vb_ckf",))
        acc["vb_nearest"].append(res.methods["vb_ckf"].armse)

    arr = {k: np.array(v) for k, v in acc.items()}
    rep = report_matrix(arr, VARIANT_ORDER)

    out = {"budget": budget, "plr": plr, "n_trials": seeds,
           "variants": {k: {"armse_mean": float(arr[k].mean()),
                            "armse_std": float(arr[k].std(ddof=1))}
                        for k in VARIANT_ORDER},
           "per_seed": rep["per_seed"],
           "pairwise": rep["pairwise"],
           "full_vs_nearest": rep["pairwise"]["full_vs_vb_nearest"]}
    return out


def main(seeds=N):
    results = {
        "A_guard_kappa": guard_kappa_ablation(seeds),
        "B_process_noise": process_noise_robustness(seeds),
        "C_cqm_indicator": cqm_indicator_ablation(seeds),
    }

    # Bonferroni within each ablation family.
    # A: gvb_vs_ckf / gvb_vs_vb across the 5 kappa thresholds.
    a = results["A_guard_kappa"]
    for key in ("gvb_vs_ckf", "gvb_vs_vb"):
        ps = [row[key]["p"] for row in a["kappa"]]
        for row, pc in zip(a["kappa"], bonferroni(ps)):
            row[key]["bonf_p"] = pc
            row[key]["family_size"] = len(ps)
    # B: vb_vs_ckf / gvb_vs_ckf across the 4 sigma_pos levels.
    b = results["B_process_noise"]
    for key in ("vb_vs_ckf", "gvb_vs_ckf"):
        ps = [lv[key]["p"] for lv in b["levels"]]
        for lv, pc in zip(b["levels"], bonferroni(ps)):
            lv[key]["bonf_p"] = pc
            lv[key]["family_size"] = len(ps)
    # C: each variant vs nearest across the 4 CQM variants.
    c = results["C_cqm_indicator"]
    vkeys = ["full", "snr_only", "prr_only", "tofj_only"]
    ps = [c["pairwise"][f"{k}_vs_vb_nearest"]["p"] for k in vkeys]
    for k, pc in zip(vkeys, bonferroni(ps)):
        c["pairwise"][f"{k}_vs_vb_nearest"]["bonf_p"] = pc
        c["pairwise"][f"{k}_vs_vb_nearest"]["family_size"] = len(vkeys)
    c["full_vs_nearest"] = c["pairwise"]["full_vs_vb_nearest"]

    (OUT / "m10_ablation_results.json").write_text(
        json.dumps(results, indent=2))

    # A
    print(f"\n=== A. Guard-kappa ablation @ {a['compass_deg']} deg "
          f"(ckf={a['ckf_armse_mean']:.3f}, vb={a['vb_armse_mean']:.3f}) ===")
    for row in a["kappa"]:
        g = row["gvb_vs_ckf"]
        print(f"  kappa={row['kappa']:>5}: gvb={row['gvb_armse_mean']:.3f} "
              f"+-{row['gvb_armse_std']:.3f}  "
              f"rate={row['guard_disable_rate']:.3f}  "
              f"vs ckf p={g['p']:.2e} bonf_p={g['bonf_p']:.2e} "
              f"TOST_equiv={g['tost']['equiv']}")

    # B
    print(f"\n=== B. Process-noise robustness @ {b['compass_deg']} deg ===")
    for lv in b["levels"]:
        v = lv["vb_vs_ckf"]
        print(f"  sigma_pos={lv['sigma_pos']:.2f}: vb={lv['summary']['vb']['armse_mean']:.3f} "
              f"ckf={lv['summary']['ckf']['armse_mean']:.3f} "
              f"vb-ckf={v['mean_diff']:+.3f} (p={v['p']:.2e}, bonf_p={v['bonf_p']:.2e}, "
              f"TOST_equiv={v['tost']['equiv']})")

    # C
    print(f"\n=== C. CQM indicator ablation (B={c['budget']}, PLR={c['plr']}) ===")
    for k in VARIANT_ORDER:
        s = c["variants"][k]
        print(f"  {k:>12}: ARMSE {s['armse_mean']:.3f} +- {s['armse_std']:.3f}")
    for k in vkeys:
        v = c["pairwise"][f"{k}_vs_vb_nearest"]
        print(f"  {k:>12} vs nearest: {v['mean_diff']:+.3f} m "
              f"(p={v['p']:.2e}, bonf_p={v['bonf_p']:.2e}, TOST_equiv={v['tost']['equiv']})")

    print(f"\nwrote {OUT / 'm10_ablation_results.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N)
    args = p.parse_args()
    main(args.seeds)
