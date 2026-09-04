"""
M0 validation -- dynamics + sensors + truth generation.

Checks:
  1. Reproducibility: same seed -> identical truth.
  2. Different seeds -> distinct trajectories.
  3. Physical sensibility: speeds near 1 m/s, finite, heading wrapped, no NaN.
  4. 2-D spatial diversity: the 8 vehicles are never collinear (so the
     relative range/bearing geometry is well-posed).
  5. Onboard sensors: follower (dim 3) vs anchor (dim 5), sane covariances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from params import DEFAULT_PARAMS, Params  # noqa: E402
from src.dynamics import generate_trajectory  # noqa: E402
from src.sensors import onboard_measurement  # noqa: E402


def _collinearity_rank(truth: NDArray, k: int) -> int:
    """Rank of the centered 2-D position matrix at epoch k (2 = full 2-D)."""
    pts = truth[:, k, :2]
    return int(np.linalg.matrix_rank(pts - pts.mean(axis=0)))


def main() -> int:
    params: Params = DEFAULT_PARAMS
    rng = np.random.default_rng(20260826)

    truth_a, ctrl_a = generate_trajectory(params, np.random.default_rng(20260826))
    truth_b, ctrl_b = generate_trajectory(params, np.random.default_rng(20260826))
    truth_c, ctrl_c = generate_trajectory(params, np.random.default_rng(7))

    failures = []

    # 1. Reproducibility
    if not np.allclose(truth_a, truth_b):
        failures.append("same seed produced different truth")

    # 2. Seed sensitivity
    if np.allclose(truth_a, truth_c):
        failures.append("different seeds produced identical truth")

    # 3. Physical sensibility
    if not np.all(np.isfinite(truth_a)):
        failures.append("non-finite states")
    if np.any(np.abs(truth_a[:, :, 2]) > np.pi + 1e-9):
        failures.append("heading outside [-pi, pi]")

    speeds = np.linalg.norm(truth_a[:, :, 3:5], axis=2)  # (n_usvs, n_steps)
    mean_speed = float(speeds.mean())
    max_speed = float(speeds.max())
    if not (0.3 < mean_speed < 2.0):
        failures.append(f"implausible mean speed {mean_speed:.3f} m/s")
    if max_speed > 5.0:
        failures.append(f"implausible max speed {max_speed:.3f} m/s")

    # 4. 2-D spatial diversity at every epoch
    min_rank = min(_collinearity_rank(truth_a, k)
                   for k in range(0, params.system.n_steps + 1,
                                  params.system.acoustic_period))
    if min_rank < 2:
        failures.append(f"fleet became collinear (min rank {min_rank})")

    # 5. Onboard sensors
    s = params.sensors
    x = truth_a[0, 100]
    z_f, R_f = onboard_measurement(x, s, np.random.default_rng(0), is_anchor=False)
    z_a, R_a = onboard_measurement(x, s, np.random.default_rng(0), is_anchor=True)
    if z_f.shape != (3,) or R_f.shape != (3, 3):
        failures.append(f"follower onboard shape wrong: {z_f.shape}")
    if z_a.shape != (5,) or R_a.shape != (5, 5):
        failures.append(f"anchor onboard shape wrong: {z_a.shape}")
    if abs(z_f[0] - z_a[2]) > 1e-9:
        failures.append("follower/anchor heading measurements disagree")

    print(f"n_steps      = {params.system.n_steps}")
    print(f"n_epochs     = {params.system.n_epochs}")
    print(f"acoustic_period = {params.system.acoustic_period}")
    print(f"truth shape  = {truth_a.shape}")
    print(f"ctrl shape   = {ctrl_a.shape}")
    print(f"mean speed   = {mean_speed:.3f} m/s (max {max_speed:.3f})")
    print(f"min 2-D rank = {min_rank}")
    print(f"fleet bbox x = [{truth_a[:, :, 0].min():.1f}, "
          f"{truth_a[:, :, 0].max():.1f}] m")
    print(f"fleet bbox y = [{truth_a[:, :, 1].min():.1f}, "
          f"{truth_a[:, :, 1].max():.1f}] m")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nM0 VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
