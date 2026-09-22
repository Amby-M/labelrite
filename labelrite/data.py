"""Synthetic data generation.

Mirrors the paper's illustrative worked example (Section 4): a workforce of
40 moderators, error rates roughly 3-6%, with three outlier annotators.
All data here is synthetic and for demonstration/testing only.
"""

from __future__ import annotations

import numpy as np


def synthetic_workforce(
    n_annotators: int = 40,
    items_per_annotator: int = 2000,
    workforce_mean: float = 0.042,
    workforce_sd: float = 0.009,
    low: float = 0.03,
    high: float = 0.06,
    outliers: tuple[tuple[str, float], ...] = (
        ("A-17", 0.066),
        ("A-29", 0.070),
        ("A-33", 0.061),
    ),
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Generate per-annotator binary outcomes (1 = disagreed with reference).

    Returns a mapping of annotator id -> array of 0/1 outcomes, ready for
    :meth:`labelrite.LabelRiteDetector.fit`.
    """
    if n_annotators < len(outliers) + 1:
        raise ValueError("n_annotators must exceed the number of outliers")
    rng = np.random.default_rng(seed)
    n_normal = n_annotators - len(outliers)
    true_rates = np.clip(
        rng.normal(workforce_mean, workforce_sd, n_normal), low, high
    )
    outcomes: dict[str, np.ndarray] = {}
    for i, p in enumerate(true_rates):
        outcomes[f"worker-{i:02d}"] = rng.binomial(1, p, items_per_annotator)
    for aid, p in outliers:
        if aid in outcomes:
            raise ValueError(f"outlier id {aid!r} collides with a generated id")
        # Exact error count (shuffled): the worked example stipulates precise
        # rates, so construction — not binomial sampling noise — sets them.
        n_errors = int(round(p * items_per_annotator))
        arr = np.zeros(items_per_annotator, dtype=int)
        arr[:n_errors] = 1
        outcomes[aid] = rng.permutation(arr)
    return outcomes
