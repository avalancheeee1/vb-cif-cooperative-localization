# Cooperative Localization for GNSS-Denied USV Swarms

Reproducible, strictly-paired Monte-Carlo simulation code for a manuscript under
review. It implements cooperative localization for a swarm of `N = 8`
Cybership-II unmanned surface vehicles (USVs) operating **without GNSS**, using
acoustic range/bearing measurements between vehicles and (in the anchor
experiments) a small number of GNSS anchors.

The code is intentionally self-contained and deterministic: every experiment is
seeded and the statistical protocol (paired *t*-tests, TOST equivalence tests,
Cohen's *d*, Bonferroni correction, minimum-detectable-difference power
analysis) is implemented here rather than delegated to a black box.

---

## The headline result: *residual dominance* in VB-CIF

The variational Bayesian cubature information filter (VB-CIF) fits its
measurement-noise covariance from the innovation residual. The central finding
of this work is the mechanism by which that adaptation can go wrong, and a
guard that fixes it.

* **Nominal compass (0.5°, high-end AHRS):** VB-CIF ≈ CKF-CL — no failure.
* **Low-cost compass (2–5°, MEMS magnetometer):** a heading error acts as a
  **lever arm** — over a dead-reckoning path of length `L = v·t` it produces a
  cross-track neighbor position error `≈ L·δψ` (unit vector `n̂ = (−sin ψ, cos ψ)`).
  That error **dominates the cooperative innovation residual**. The VB-M step
  subtracts only the *receiver's own* state contribution `H P Hᵀ`, not the
  *neighbor's* error `Hⱼ eⱼ eⱼᵀ Hⱼᵀ`, so the fitted noise variance is biased
  upward, the still-informative measurements are under-weighted, and VB-CIF
  degrades below fixed-covariance CKF-CL.
* **G-VB-CIF guard:** a χ² NIS gate (`κ = χ²₀.₉₅(2) = 5.99`) detects residual
  dominance and falls back to fixed covariance, recovering CKF-CL accuracy.

An earlier "common-mode unobservability" hypothesis **does not reproduce** and is
superseded: a common-mode drift cancels identically in the relative range and
bearing (both `|pᵢ − pⱼ|` and `atan2(dy, dx) − ψ` are invariant to a shared
translation), so it is invisible to the residual and cannot bias VB-CIF. The
reproducible mechanism is residual dominance by the neighbor's
compass-driven dead-reckoning error — see `s1_mechanism.py`.

Two honest negative controls sharpen the claim:

* VB-CIF is **not** degraded by generic process-noise mismatch — as `σ_pos`
  grows it actually *improves* relative to CKF-CL (`m10_ablation.py`, part B).
* The CQM scheduler's multi-metric weighting contributes little on its own; the
  real scheduling gain comes from the **anchor-aware selection structure**
  (`s2_scheduling.py`, `s3_anchor.py`, `m10_ablation.py`, part C).

---

## Requirements

* Python 3.10+ (developed and tested on 3.13)
* `numpy`, `scipy`

```bash
pip install -r requirements.txt
```

No GPU, no third-party solver, no network access is required. `matplotlib` is
only needed if you re-generate publication figures (not included here).

---

## Quick start

```bash
# 1. Core failure-mechanism sweep (0.5 / 2 / 5 deg compass, 30 seeds each)
python s1_mechanism.py --seeds 30

# 2. Isolated cooperative-update timing (VB/CKF cost ratio)
python s7_timing.py
```

Each script writes its results to `outputs/<name>.json`. Run any script with
`--help` for options; all experiment scripts accept `--seeds`.

| Script | Experiment | Writes |
|---|---|---|
| `s1_mechanism.py` | compass-noise sweep (0.5/2/5°) — the core mechanism | `outputs/s1_mechanism_results.json` |
| `s1_repro.py` | MAE vs RMS metric comparison across filters | (stdout) |
| `s1_noise_covariance.py` | fitted-noise-covariance diagnostics | `outputs/s1_noise_covariance.json` |
| `s2_scheduling.py` | link scheduling, budget × packet-loss sweep | `outputs/s2_scheduling_results.json` |
| `s3_anchor.py` | GNSS-anchor-aware scheduling | `outputs/s3_anchor_results.json` |
| `s4_baselines.py` | baseline filter comparison | `outputs/s4_baselines_results.json` |
| `s5_anchor_count.py` | anchor-count sweep | `outputs/s5_anchor_count_results.json` |
| `s6_random_topology.py` | random-topology robustness | `outputs/s6_random_topology_results.json` |
| `s7_timing.py` | micro-benchmark of cooperative updates | `outputs/s7_timing.json` |
| `s7_vb_iter_ablation.py` | VB fixed-point iteration ablation | `outputs/s7_vb_iter_ablation.json` |
| `m10_ablation.py` | guard-κ / process-noise / CQM-metric ablations | `outputs/m10_ablation_results.json` |
| `m11_prior_sensitivity.py` | inverse-Wishart prior sensitivity | `outputs/m11_prior_sensitivity_results.json` |
| `sweep_drift.py` | drift sweep | (stdout) |

`validate/` contains standalone sanity checks of the measurement model,
dynamics, and filter update (`validate_m0.py`, `validate_m1.py`, `validate_m2.py`).
`posthoc_power_s2s3.py`, `posthoc_power_s6.py`, and `power_tost_sensitivity.py`
implement the post-hoc power / TOST-sensitivity analyses behind the
"minimum-detectable-difference" statements.

---

## Reference results

The `outputs/` directory contains the reference JSON produced by a full
30-seed run. Headline numbers (ARMSE, metres):

| compass | CKF-CL | VB-CIF | VB−CKF (paired) | G-VB-CIF | G-VB−CKF (paired) | guard rate |
|---|---|---|---|---|---|---|
| 0.5° | 3.296 | 3.264 | −0.031 | 3.181 | −0.115 | 0.128 |
| 2.0° | 3.254 | 4.817 | +1.562 | 3.099 | −0.155 | 0.218 |
| 5.0° | 3.449 | 5.665 | +2.216 | 3.306 | −0.143 | 0.254 |

VB > CKF in 30/30 seeds at 2° and 5° (14/30 at 0.5°), the guard-disable rate
rises monotonically with compass noise (0.128 → 0.218 → 0.254, the "mechanism
switch"), and G-VB-CIF recovers or beats CKF-CL at every level. Re-running the
script with the same seeds reproduces these values exactly.

---

## Layout

```
params.py            single source of truth for every parameter (frozen dataclasses)
s1_stats.py          paired t-test, TOST, Cohen's d, Bonferroni, MDD helpers
s1_…–s7_…, m10/m11   experiment drivers (see table above)
src/
  dynamics.py        Cybership-II 3-DOF Fossen model + truth generation
  sensors.py         onboard DVL / compass / GNSS measurements
  channel.py         shallow-water acoustic channel (SNR, fading, outliers)
  measurement.py     range/bearing measurement model + Jacobian
  filters.py         EKF-CL / CKF-CL / IWCF / VB-CIF / G-VB-CIF updates
  cqm.py             communication quality metric (SNR/PRR/ToF-jitter)
  scheduler.py       nearest / random / anchor-first / CQM / oracle selection
  scenario.py        per-epoch shared draws (strictly paired protocol)
  runner.py          outer epoch loop, inner method loop
  runner_sched.py    scheduler-aware runner
validate/            model/filter sanity checks
outputs/             reference results (regenerable)
```

`params.py` is the single source of truth and is transcribed directly from the
manuscript's experimental setup; the dataclasses are frozen so a run can never
silently mutate its own configuration.

---

## Reproducibility

* Fixed RNG: every experiment draws from `np.random.default_rng(seed)` with
  seed `0, 1, …, N−1`, so results are bit-for-bit reproducible.
* Strictly paired protocol: all filters share the same truth trajectory,
  perturbations, and measurement draws per seed (`src/scenario.py`).
* Headline contrasts carry a two-sided paired *t*-test (95% CI), a TOST
  equivalence test (margin 0.5 m), and Bonferroni correction across the swept
  factor levels.

---

## Data availability

No external datasets are required: trajectories, perturbations, channel
realizations, and measurements are all generated synthetically from the
parameters in `params.py`. The reference outputs are included in `outputs/`.

## License

[MIT](LICENSE). (Add the author name to `LICENSE` at publication time.)

## Citation

This repository accompanies a manuscript under review. A full citation and
author list will be added here upon acceptance.
