"""Confidence-graded corrective actions (paper, Section 3).

Automatic intervention on weak evidence would itself introduce error, so the
response is proportional to detection confidence:

- high:     automatic retraining workflow + label quarantine.
- moderate: surface aggregate metrics to the annotator and their management;
            no automatic action.
- none:     no action.
"""


def recommended_action(confidence: str) -> dict:
    """Return the corrective-action routing for a confidence level."""
    if confidence == "high":
        return {
            "quarantine_labels": True,
            "trigger_retraining": True,
            "notify": ["annotator", "management"],
            "automatic": True,
        }
    if confidence == "moderate":
        return {
            "quarantine_labels": False,
            "trigger_retraining": False,
            "notify": ["annotator", "management"],
            "automatic": False,
        }
    if confidence == "none":
        return {
            "quarantine_labels": False,
            "trigger_retraining": False,
            "notify": [],
            "automatic": False,
        }
    raise ValueError(f"unknown confidence level: {confidence!r}")
