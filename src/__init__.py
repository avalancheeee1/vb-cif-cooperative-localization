"""Clean re-implementation of the cooperative-localization simulation.

Layers (see README):

  1. dynamics     -- Cybership-II 3-DOF Fossen model + truth generation
  2. sensors      -- onboard DVL / compass / GNSS measurements
  3. channel      -- shallow-water acoustic channel (SNR, fading, outliers)
  4. measurement  -- range-bearing measurement model + Jacobian
  5. filters      -- CKF-CL / EKF-CL / IWCF / VB-CIF / G-VB-CIF
  6. cqm          -- communication quality metric
  7. scheduling   -- nearest / random / anchor-first / CQM / oracle
  8. scenario     -- per-epoch shared draws (strictly paired protocol)
  9. runner       -- outer epoch loop, inner method loop

Statistics helpers (paired t-tests, TOST, Cohen's d, Bonferroni, MDD) live in
``s1_stats.py`` at the repository root. Publication figures are not included in
this repository.
"""
