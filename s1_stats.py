"""
Strictly-paired statistical protocol (paper Sec. ``Strictly paired evaluation
protocol``).  Single source of truth for every comparison the manuscript
reports, so that the paper's statistics are exactly what the code computes.

Each helper operates on the per-seed scalar arrays ``a`` and ``b`` for two
methods run on identical inputs (strictly paired).  All significance decisions
use a two-sided level ``alpha`` (default 0.05, matching the paper's "95% CI"
and its use of p=0.02/0.01 as significant).

Functions
---------
paired_test(a, b, alpha)   two-sided paired t-test + true (1-alpha) CI + d.
tost(a, b, margin, alpha)  two one-sided equivalence tests (Schuirmann 1987).
full_report(a, b, ...)     paired_test + TOST in one dict (used in the JSONs).
bonferroni(ps)             Bonferroni correction for a family of p-values.
report_matrix(arr, order)  full pairwise contrast matrix over >=2 methods,
                           plus per-seed arrays (for sign counts / audit).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

DEFAULT_ALPHA = 0.05          # significance level (two-sided t-test + TOST)
DEFAULT_MARGIN = 0.5          # m, equivalence margin (20% of 2.5 m range std)


def paired_test(a: np.ndarray, b: np.ndarray,
                alpha: float = DEFAULT_ALPHA) -> dict:
    """Two-sided paired t-test on the per-seed differences (a - b).

    Returns the mean/std of the difference, the t statistic, the two-sided
    p-value, a true ``(1 - alpha)`` confidence interval (labeled ``ci``; for the
    default alpha=0.05 this is the 95% CI), Cohen's d, and n.
    """
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1) if n > 1 else float("nan")
    ci = (float(d.mean() - tcrit * se), float(d.mean() + tcrit * se))
    cohens_d = float(d.mean() / sd) if sd > 0 else 0.0
    return {
        "mean_diff": float(d.mean()),
        "std_diff": sd,
        "t": float(t),
        "p": float(p),
        "ci": [ci[0], ci[1]],
        "ci_level": 1 - alpha,
        "cohens_d": cohens_d,
        "n": n,
    }


def tost(a: np.ndarray, b: np.ndarray, margin: float = DEFAULT_MARGIN,
         alpha: float = DEFAULT_ALPHA) -> dict:
    """Two one-sided equivalence tests (TOST) for ``|mean(a-b)| < margin``.

    Rejects the null "the true difference is outside +/-margin" only when both
    one-sided tests are significant; ``equiv`` is True in that case and the
    methods may be declared practically equivalent at level ``alpha``.  The
    overall TOST p-value is the larger of the two one-sided p-values.
    """
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    df = n - 1
    # H0_lower: mu_d <= -margin  (reject -> mu_d > -margin)
    t_lower = (d.mean() + margin) / se if se > 0 else float("inf")
    # H0_upper: mu_d >= +margin  (reject -> mu_d < +margin)
    t_upper = (d.mean() - margin) / se if se > 0 else float("-inf")
    p_lower = float(stats.t.sf(t_lower, df)) if n > 1 else 1.0
    p_upper = float(stats.t.cdf(t_upper, df)) if n > 1 else 1.0
    p_tost = max(p_lower, p_upper)
    return {
        "margin": margin,
        "tost_p": p_tost,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equiv": bool(p_tost < alpha),
        "alpha": alpha,
        "n": n,
    }


def full_report(a: np.ndarray, b: np.ndarray, margin: float = DEFAULT_MARGIN,
                alpha: float = DEFAULT_ALPHA) -> dict:
    """Paired t-test + TOST in a single dict (the JSON schema for a contrast)."""
    rep = paired_test(a, b, alpha)
    rep["tost"] = tost(a, b, margin, alpha)
    return rep


def bonferroni(ps: list[float]) -> list[float]:
    """Bonferroni correction for a family of raw p-values (min(1, p*m))."""
    m = len(ps)
    return [min(1.0, float(p) * m) for p in ps]


def report_matrix(arr: dict, order: tuple[str, ...],
                  margin: float = DEFAULT_MARGIN,
                  alpha: float = DEFAULT_ALPHA) -> dict:
    """Full pairwise contrast matrix over the methods in ``arr``.

    Returns
    -------
    summary   {method: {armse_mean, armse_std}}
    per_seed  {method: [armse, ...]}            (raw, for sign counts / audit)
    pairwise  {"a_vs_b": full_report(...), ...} for every ordered pair a!=b
    """
    summary = {m: {"armse_mean": float(arr[m].mean()),
                   "armse_std": float(arr[m].std(ddof=1))}
               for m in order}
    per_seed = {m: [float(x) for x in arr[m]] for m in order}
    pairwise = {}
    for a in order:
        for b in order:
            if a == b:
                continue
            pairwise[f"{a}_vs_{b}"] = full_report(arr[a], arr[b], margin, alpha)
    return {"summary": summary, "per_seed": per_seed, "pairwise": pairwise}
