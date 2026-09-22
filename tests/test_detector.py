"""Tests for the LabelRite reference implementation."""

import numpy as np
import pytest

from labelrite import LabelRiteDetector, recommended_action, synthetic_workforce


def _clean_workforce(seed=0, n=20, p=0.04, items=500):
    # Deterministic: every worker has exactly the same error rate, so the
    # reference sigma is 0 and nothing can be flagged. Tests the no-flag path
    # without depending on sampling luck.
    n_errors = int(round(p * items))
    base = np.zeros(items, dtype=int)
    base[:n_errors] = 1
    rng = np.random.default_rng(seed)
    return {f"w-{i:02d}": rng.permutation(base) for i in range(n)}


def test_flags_known_outliers():
    det = LabelRiteDetector(random_state=0).fit(synthetic_workforce())
    flagged = {r.annotator_id: r for r in det.flagged()}
    assert set(flagged) == {"A-17", "A-29", "A-33"}
    assert flagged["A-29"].confidence == "high"
    assert flagged["A-17"].confidence == "high"
    assert flagged["A-33"].confidence == "moderate"
    # paper's worked example: ~2.7 SD and ~3.1 SD for the two high-confidence flags
    assert flagged["A-17"].z_score == pytest.approx(2.7, abs=0.3)
    assert flagged["A-29"].z_score == pytest.approx(3.1, abs=0.3)
    assert flagged["A-33"].z_score == pytest.approx(2.1, abs=0.3)
    # high-confidence flags quarantine; moderate only notifies
    assert flagged["A-29"].action["quarantine_labels"] is True
    assert flagged["A-29"].action["trigger_retraining"] is True
    assert flagged["A-33"].action["quarantine_labels"] is False
    assert flagged["A-33"].action["notify"] == ["annotator", "management"]


def test_no_flags_on_clean_workforce():
    det = LabelRiteDetector(random_state=1).fit(_clean_workforce())
    assert det.flagged() == []


def test_n_min_enforced():
    outcomes = _clean_workforce(items=500)
    outcomes["w-00"] = outcomes["w-00"][:50]  # below default n_min=100
    with pytest.raises(ValueError, match="n_min"):
        LabelRiteDetector().fit(outcomes)


def test_binary_outcomes_required():
    outcomes = _clean_workforce()
    outcomes["w-00"] = np.array([0.0, 0.5, 1.0] * 200)
    with pytest.raises(ValueError, match="binary"):
        LabelRiteDetector().fit(outcomes)


def test_fit_from_averages_matches_fit():
    outcomes = synthetic_workforce(seed=11)
    avgs = {k: float(v.mean()) for k, v in outcomes.items()}
    counts = {k: len(v) for k, v in outcomes.items()}
    a = LabelRiteDetector(random_state=3).fit(outcomes)
    b = LabelRiteDetector(random_state=3).fit_from_averages(avgs, counts)
    assert [r.annotator_id for r in a.flagged()] == [r.annotator_id for r in b.flagged()]


def test_reproducible_with_seed():
    outcomes = synthetic_workforce(seed=5)
    a = LabelRiteDetector(random_state=42).fit(outcomes)
    b = LabelRiteDetector(random_state=42).fit(outcomes)
    assert [(r.annotator_id, r.z_score) for r in a.results_] == [
        (r.annotator_id, r.z_score) for r in b.results_
    ]


def test_explanation_is_human_readable():
    det = LabelRiteDetector(random_state=0).fit(synthetic_workforce())
    r = next(x for x in det.results_ if x.annotator_id == "A-29")
    assert "A-29" in r.explanation
    assert "standard deviations" in r.explanation
    assert "reference mean" in r.explanation
    assert "reviewed items" in r.explanation


def test_quarantine_impact_estimate():
    det = LabelRiteDetector(random_state=0).fit(synthetic_workforce())
    impact = det.quarantine_impact(items_per_week=500)
    assert impact["n_quarantined"] == 2
    assert set(impact["annotators"]) == {"A-17", "A-29"}
    # (6.6% + 7.0%) of 500 weekly items each ~= 68 mislabeled items/week
    assert impact["est_mislabeled_per_week"] == pytest.approx(68, abs=10)


def test_recommended_action_routing():
    assert recommended_action("high")["automatic"] is True
    assert recommended_action("moderate")["automatic"] is False
    assert recommended_action("moderate")["quarantine_labels"] is False
    assert recommended_action("none")["notify"] == []
    with pytest.raises(ValueError):
        recommended_action("bogus")


def test_reference_stats_populated():
    det = LabelRiteDetector(random_state=0).fit(synthetic_workforce())
    assert det.reference_mean_ == pytest.approx(0.042, abs=0.01)
    assert det.reference_sigma_ == pytest.approx(0.009, abs=0.005)
    lo, hi = det.reference_mean_ci95_
    assert lo < det.reference_mean_ < hi
