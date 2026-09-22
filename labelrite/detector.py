"""The four-step LabelRite detection procedure (paper, Section 3).

Step 1: Per-annotator error rates — for each annotator, average the error
        rate over at least ``n_min`` items scored against a trusted reference.
Step 2: Reference distribution — model the population of per-annotator error
        rates; the reference mean ``mu`` and standard deviation ``sigma`` are
        estimated here, with a bootstrap confidence interval on ``mu``.
Step 3: Outlier identification — flag annotators with ``|e_m - mu| > 2*sigma``.
Step 4: Correct and retrain — route flags through the confidence-graded
        response in :mod:`labelrite.actions`.

Reference-implementation note: the paper's worked example (Section 4) reports
z-scores against the population standard deviation of per-annotator error
rates (e.g. 6.6% vs mu=4.2%, sigma=0.9% -> 2.7 SD). This implementation follows
that observable behavior: ``sigma`` is the standard deviation of the
per-annotator error rates. The bootstrap procedure described in the paper is
used to report a 95% confidence interval on the reference mean. The worked
example in ``examples/worked_example.py`` is the conformance test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from .actions import recommended_action


def _normal_sf(z: float) -> float:
    """Survival function of the standard normal, P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class AnnotatorResult:
    """Outcome for a single annotator."""

    annotator_id: str
    n_items: int
    error_rate: float
    z_score: float  # signed: (error_rate - mu) / sigma
    p_value_two_sided: float
    flagged: bool
    confidence: str  # "high" | "moderate" | "none"
    explanation: str  # human-readable, per the paper's interpretability requirement
    action: dict = field(default_factory=dict)


class LabelRiteDetector:
    """Real-time annotator anomaly detector.

    Parameters
    ----------
    n_min:
        Minimum items per annotator before an error rate is trusted
        (paper recommends 100).
    sigma_threshold:
        Flag annotators beyond this many standard deviations (paper: 2.0).
    high_confidence_z:
        Flags at or beyond this z-score get the high-confidence response
        (automatic retraining + label quarantine). Flags between
        ``sigma_threshold`` and this value get the moderate response
        (notification only). Tunable; the paper's example puts 2.1 SD in the
        moderate band and 2.7/3.1 SD in the high band.
    n_bootstrap:
        Bootstrap resamples for the reference-mean confidence interval.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_min: int = 100,
        sigma_threshold: float = 2.0,
        high_confidence_z: float = 2.5,
        n_bootstrap: int = 10_000,
        random_state=None,
    ):
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        if sigma_threshold <= 0:
            raise ValueError("sigma_threshold must be positive")
        if not sigma_threshold < high_confidence_z:
            raise ValueError("need sigma_threshold < high_confidence_z")
        self.n_min = int(n_min)
        self.sigma_threshold = float(sigma_threshold)
        self.high_confidence_z = float(high_confidence_z)
        self.n_bootstrap = int(n_bootstrap)
        self._rng = np.random.default_rng(random_state)

        # Filled by fit().
        self.reference_mean_: float | None = None
        self.reference_sigma_: float | None = None
        self.reference_mean_ci95_: tuple[float, float] | None = None
        self.results_: list[AnnotatorResult] | None = None

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _coerce_outcomes(
        self, error_outcomes: Mapping[str, Sequence[int]]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        ids, ns, means = [], [], []
        for aid, outcomes in error_outcomes.items():
            arr = np.asarray(outcomes, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError(f"annotator {aid!r}: empty outcome array")
            if not np.all((arr == 0.0) | (arr == 1.0)):
                raise ValueError(
                    f"annotator {aid!r}: outcomes must be binary "
                    "(1 = disagreed with reference, 0 = agreed)"
                )
            ids.append(str(aid))
            ns.append(arr.size)
            means.append(arr.mean())
        return ids, np.asarray(ns, dtype=int), np.asarray(means, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, error_outcomes: Mapping[str, Sequence[int]]) -> "LabelRiteDetector":
        """Fit on per-item binary outcomes.

        ``error_outcomes`` maps annotator id -> array of 0/1, where 1 means
        the annotator's label disagreed with the trusted reference.
        """
        ids, ns, means = self._coerce_outcomes(error_outcomes)
        return self._analyze(ids, ns, means)

    def fit_from_averages(
        self,
        avg_error_rates: Mapping[str, float],
        item_counts: Mapping[str, int],
    ) -> "LabelRiteDetector":
        """Fit on precomputed per-annotator error rates and item counts."""
        ids = [str(a) for a in avg_error_rates]
        if set(ids) != {str(a) for a in item_counts}:
            raise ValueError("avg_error_rates and item_counts must cover the same annotators")
        means = np.array([float(avg_error_rates[a]) for a in avg_error_rates], dtype=float)
        ns = np.array([int(item_counts[a]) for a in avg_error_rates], dtype=int)
        if np.any((means < 0.0) | (means > 1.0)):
            raise ValueError("error rates must be in [0, 1]")
        return self._analyze(ids, ns, means)

    # ------------------------------------------------------------------
    # Core procedure
    # ------------------------------------------------------------------
    def _analyze(
        self, ids: list[str], ns: np.ndarray, means: np.ndarray
    ) -> "LabelRiteDetector":
        if len(ids) < 3:
            raise ValueError("LabelRite needs at least 3 annotators for a reference distribution")
        short = sorted(i for i, n in zip(ids, ns) if n < self.n_min)
        if short:
            raise ValueError(
                f"{len(short)} annotator(s) below n_min={self.n_min}: "
                + ", ".join(short[:10])
                + (" ..." if len(short) > 10 else "")
            )

        # Step 2: reference distribution.
        mu = float(means.mean())
        sigma = float(means.std(ddof=0))
        boot = self._rng.choice(means, size=(self.n_bootstrap, len(means)), replace=True).mean(axis=1)
        ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        self.reference_mean_ = mu
        self.reference_sigma_ = sigma
        self.reference_mean_ci95_ = ci95

        # Steps 3+4: flag and route.
        results: list[AnnotatorResult] = []
        for aid, n, e in zip(ids, ns, means):
            z = (float(e) - mu) / sigma if sigma > 0 else 0.0
            az = abs(z)
            flagged = bool(az > self.sigma_threshold)
            if not flagged:
                confidence = "none"
            elif az >= self.high_confidence_z:
                confidence = "high"
            else:
                confidence = "moderate"
            p = 2.0 * _normal_sf(az)
            direction = "above" if z > 0 else "below"
            if flagged:
                explanation = (
                    f"Annotator {aid}: error rate {e:.1%} lies {az:.1f} standard "
                    f"deviations {direction} the workforce reference mean of {mu:.1%} "
                    f"(sigma={sigma:.1%}), based on {int(n)} reviewed items "
                    f"(two-sided p={p:.4f}). Confidence: {confidence}."
                )
            else:
                explanation = (
                    f"Annotator {aid}: error rate {e:.1%} within "
                    f"{self.sigma_threshold:g} standard deviations of the workforce "
                    f"reference mean of {mu:.1%} ({az:.1f} SD, {int(n)} items). No flag."
                )
            results.append(
                AnnotatorResult(
                    annotator_id=aid,
                    n_items=int(n),
                    error_rate=float(e),
                    z_score=float(z),
                    p_value_two_sided=float(p),
                    flagged=flagged,
                    confidence=confidence,
                    explanation=explanation,
                    action=recommended_action(confidence),
                )
            )
        results.sort(key=lambda r: (r.flagged, abs(r.z_score)), reverse=True)
        self.results_ = results
        return self

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def flagged(self) -> list[AnnotatorResult]:
        """Flagged annotators, most extreme first."""
        if self.results_ is None:
            raise RuntimeError("call fit() first")
        return [r for r in self.results_ if r.flagged]

    def quarantine_impact(self, items_per_week: Mapping[str, int] | int) -> dict:
        """Estimated mislabeled items per week removed by quarantining flags.

        ``items_per_week`` is either a per-annotator mapping or a single
        weekly volume applied to every quarantined annotator.
        """
        if self.results_ is None:
            raise RuntimeError("call fit() first")
        quarantined = [r for r in self.results_ if r.action.get("quarantine_labels")]
        if isinstance(items_per_week, Mapping):
            weekly = sum(items_per_week.get(r.annotator_id, 0) * r.error_rate for r in quarantined)
        else:
            weekly = sum(r.error_rate * int(items_per_week) for r in quarantined)
        return {
            "n_quarantined": len(quarantined),
            "annotators": [r.annotator_id for r in quarantined],
            "est_mislabeled_per_week": weekly,
        }
