"""
M8 -- Link scheduling policies (paper Sec. Methods / Sec. III-C).

Given a receiver's candidate links (the in-range links that *would* be received
this epoch), a policy selects up to ``budget`` of them to interrogate.  Four
policies are implemented:

  nearest  select the B links with smallest *estimated* range (baseline)
  random   select B links uniformly at random (baseline)
  cqm      greedy max log-det Fisher information, weighted by the CQM score
           alpha and with the measurement noise inflated by
           sqrt(1 + eta*(1-alpha)) to discount low-confidence links
  oracle   greedy max log-det Fisher information using the *true* reception
           probability p_succ and the *true* noise (an unachievable upper bound)

The greedy rule builds the 2x2 position information matrix incrementally:
  Y_{22} += alpha_j * H_j^T R_j^{-1} H_j
and at each step admits the candidate maximizing log det(Y_{22}) (equivalently
log(det(M)), where M = Y + J is the 2x2 information block after adding a link).

After selection, the chosen links are returned in a *canonical* order (sorted by
true range), so that policy differences reflect link *selection* only and not
the order dependence of sequential nonlinear updates.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from params import FilterParams
from src.cqm import CQM
from src.measurement import measurement_jacobian
from src.scenario import LinkDraw


def _pos_block(x_i: NDArray, x_j: NDArray) -> NDArray:
    """2x2 position block of the range-bearing Jacobian dh/dp_i."""
    return measurement_jacobian(x_i, x_j)[:, :2]


def _noise_inv_oracle(link: LinkDraw) -> NDArray:
    sd = max(link.sigma_d, 1e-6)
    st = max(link.sigma_theta, 1e-6)
    return np.diag([1.0 / sd ** 2, 1.0 / st ** 2])


def _noise_inv_cqm(alpha: float, fp: FilterParams) -> NDArray:
    infl = np.sqrt(1.0 + fp.eta_inflation * (1.0 - alpha))
    sd = fp.sigma_d_nominal * infl
    st = fp.sigma_theta_nominal * infl           # already in radians
    return np.diag([1.0 / sd ** 2, 1.0 / st ** 2])


def _greedy(policy: str, i: int, candidates: list[LinkDraw], x_est: NDArray,
            P_est: NDArray, x_nb: dict, cqm: CQM, fp: FilterParams,
            budget: int, anchors: frozenset[int] = frozenset(),
            anchor_value: float = 4.0) -> list[LinkDraw]:
    """Greedy max log-det Fisher-information selection (cqm / oracle).

    ``anchors`` is the set of anchor (GNSS-aided) vehicle ids; a link to an
    anchor carries absolute-position information, so its information weight is
    multiplied by ``anchor_value`` (S3 only; empty in S2).
    """
    Y = np.linalg.inv(P_est)[:2, :2]
    selected: list[LinkDraw] = []
    remaining = list(candidates)

    for _ in range(min(budget, len(candidates))):
        best_link = None
        best_gain = -np.inf
        for link in remaining:
            H = _pos_block(x_est, x_nb[link.j])
            if policy == "oracle":
                w = max(link.p_succ, 1e-3)
                R_inv = _noise_inv_oracle(link)
            else:  # cqm
                alpha = cqm.get_alpha(i, link.j)
                w = alpha
                R_inv = _noise_inv_cqm(alpha, fp)
            if link.j in anchors:
                w *= anchor_value
            J = w * (H.T @ R_inv @ H)
            M = Y + J
            gain = np.log(max(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0], 1e-12))
            if gain > best_gain:
                best_gain = gain
                best_link = link

        if best_link is None:
            break
        selected.append(best_link)
        remaining.remove(best_link)
        H = _pos_block(x_est, x_nb[best_link.j])
        if policy == "oracle":
            w = max(best_link.p_succ, 1e-3)
            R_inv = _noise_inv_oracle(best_link)
        else:
            alpha = cqm.get_alpha(i, best_link.j)
            w = alpha
            R_inv = _noise_inv_cqm(alpha, fp)
        if best_link.j in anchors:
            w *= anchor_value
        Y = Y + w * (H.T @ R_inv @ H)

    return selected


def select_links(policy: str, i: int, candidates: list[LinkDraw],
                 x_est: NDArray, P_est: NDArray, x_nb: dict, cqm: CQM,
                 fp: FilterParams, rng: np.random.Generator,
                 budget: int, anchors: frozenset[int] = frozenset(),
                 anchor_value: float = 4.0) -> list[LinkDraw]:
    """Select <= budget links for receiver i and return them in canonical order.

    ``candidates`` are the received in-range links (LinkDraw) for receiver i.
    ``x_nb`` maps neighbor id -> predicted state at the epoch boundary.
    ``anchors`` (S3) is the set of anchor vehicle ids; ``anchor_value`` is the
    information bonus applied to anchor links by the cqm/oracle greedy.
    """
    def _range(lk: LinkDraw) -> float:
        return float(np.hypot(x_est[0] - x_nb[lk.j][0],
                              x_est[1] - x_nb[lk.j][1]))

    if budget >= len(candidates):
        sel = list(candidates)
    elif policy == "nearest":
        sel = sorted(candidates, key=_range)[:budget]
    elif policy == "random":
        idx = rng.choice(len(candidates), size=budget, replace=False)
        sel = [candidates[k] for k in idx]
    elif policy == "anchor_first":
        anchors_links = [lk for lk in candidates if lk.j in anchors]
        follower_links = [lk for lk in candidates if lk.j not in anchors]
        anchors_links.sort(key=_range)
        follower_links.sort(key=_range)
        sel = (anchors_links + follower_links)[:budget]
    elif policy in ("cqm", "oracle"):
        sel = _greedy(policy, i, candidates, x_est, P_est, x_nb, cqm, fp,
                      budget, anchors, anchor_value)
    else:
        raise ValueError(f"unknown policy {policy!r}")

    # Canonical order (true range) -- shared across methods, removes order
    # dependence from the sequential nonlinear update.
    sel.sort(key=lambda lk: lk.d)
    return sel
