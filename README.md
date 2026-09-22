# LabelRite

Reference implementation of **LabelRite**: real-time detection of anomalous annotators in human labeling pipelines via reference-distribution modeling.

Instead of treating human labels as ground truth, LabelRite treats them as *signals carrying uncertainty*: it models each annotator's error rate against a reference distribution built from the annotator population, flags outliers beyond two standard deviations, and triggers graduated corrective actions — retraining and label quarantine — proportional to detection confidence. Every flag is explainable in human-readable terms (interpretability by construction).

**Method paper:** Majumdar, A. (2026). *LabelRite: Real-Time Detection of Anomalous Annotators in Human Labeling Pipelines via Reference-Distribution Modeling.* Zenodo. https://doi.org/10.5281/zenodo.22887754

## Installation

```bash
pip install labelrite        # once published
# or from source:
pip install -e .
```

Requires Python ≥ 3.9 and `numpy` only.

## Quickstart

```python
from labelrite import LabelRiteDetector

# outcomes: annotator_id -> array of 0/1, where 1 = label disagreed
# with the trusted reference (expert re-labels, gold set, outcome signal)
outcomes = {
    "annotator-01": [0, 1, 0, 0, ...],   # >= 100 items each (n_min)
    ...
}

det = LabelRiteDetector().fit(outcomes)

print(f"reference: mu={det.reference_mean_:.1%}, sigma={det.reference_sigma_:.1%}")
for r in det.flagged():
    print(r.explanation)
    print(r.action)   # quarantine / retrain / notify routing
```

## The four-step procedure

1. **Per-annotator error rates** — average each annotator's error rate over at least `n_min` items (default 100).
2. **Reference distribution** — model the population of per-annotator error rates; report the reference mean μ, standard deviation σ, and a bootstrap 95% CI on μ.
3. **Outlier identification** — flag annotators with |e − μ| > 2σ.
4. **Correct and retrain** — route each flag through a confidence-graded response:
   - **high** (≥ 2.5 SD): automatic retraining workflow + label quarantine;
   - **moderate** (2–2.5 SD): metrics surfaced to the annotator and management, no automatic action.

## Worked example (paper, Section 4)

```bash
python examples/worked_example.py
```

Runs the synthetic 40-moderator workforce from the paper: two annotators flagged with high confidence (quarantined + retrained), one in the low-confidence band (notification only). This script is the conformance test — its assertions encode the paper's reported outcomes.

## Tests

```bash
pytest
```

## Citation

```bibtex
@misc{majumdar2026labelrite,
  author       = {Ambarish Majumdar},
  title        = {LabelRite: Real-Time Detection of Anomalous Annotators in Human Labeling Pipelines via Reference-Distribution Modeling},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22887754},
  url          = {https://doi.org/10.5281/zenodo.22887754}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable record.

## License

MIT — see [LICENSE](LICENSE). The method paper is separately archived on Zenodo under CC-BY-4.0.
