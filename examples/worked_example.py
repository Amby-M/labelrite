"""Reproduces the paper's illustrative worked example (Section 4).

Workforce of 40 moderators, ~500 items/week each over four weeks, scored
against a 200-item expert-adjudicated gold reference. Two annotators are
flagged with high confidence (retraining + label quarantine); a third falls
in the low-confidence band (management notification only).

All data is synthetic.
"""

from labelrite import LabelRiteDetector, synthetic_workforce

ITEMS_PER_WEEK = 500


def main() -> None:
    outcomes = synthetic_workforce(
        n_annotators=40,
        items_per_annotator=4 * ITEMS_PER_WEEK,
        seed=7,
    )
    det = LabelRiteDetector(random_state=0).fit(outcomes)

    print(f"Workforce reference: mu={det.reference_mean_:.1%}, "
          f"sigma={det.reference_sigma_:.1%}")
    lo, hi = det.reference_mean_ci95_
    print(f"95% bootstrap CI on reference mean: [{lo:.1%}, {hi:.1%}]\n")

    for r in det.flagged():
        print(f"[{r.confidence.upper()}] {r.explanation}")
        print(f"          action: {r.action}\n")

    impact = det.quarantine_impact(items_per_week=ITEMS_PER_WEEK)
    print(f"Quarantining {impact['n_quarantined']} annotator(s) "
          f"{impact['annotators']} removes an estimated "
          f"{impact['est_mislabeled_per_week']:.0f} mislabeled items per week "
          f"from training input.")

    # Conformance with the paper's worked example.
    flagged = {r.annotator_id: r.confidence for r in det.flagged()}
    assert flagged == {"A-29": "high", "A-17": "high", "A-33": "moderate"}, flagged
    print("\nConformance check passed: matches the paper's worked example.")


if __name__ == "__main__":
    main()
