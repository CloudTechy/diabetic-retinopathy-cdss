"""
What a stored model output may be called in the report.

The trained engine applies temperature scaling fitted on the validation
partition and records the temperature in the persisted model version. Only an
output with that record whose five values form a valid probability
distribution is labelled a class probability. Anything else (a result
stored before calibration, a malformed vector) keeps the
neutral "model-generated class score" wording, so an uncalibrated value is
never relabelled as a probability.

Grad-CAM wording (activation, attribution, heatmap) is deliberately not here:
it describes where the model looked, not how likely a class is.

The interface applies the same rule in frontend/src/utils/scoreTerminology.ts.
"""
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

CALIBRATION_EXPLANATION = (
    "Class probabilities are produced using temperature scaling fitted on the validation partition. "
    "They support professional review and do not establish a diagnosis."
)
UNCALIBRATED_EXPLANATION = (
    "This record carries no verified calibration metadata, so its values are shown as model-generated class "
    "scores, not probabilities. They support professional review and do not establish a diagnosis."
)

_TEMPERATURE_IN_VERSION = re.compile(r"temperature-scaled T=([0-9]+(?:\.[0-9]+)?)")
_SUM_TOLERANCE = 1e-3


def recorded_temperature(model_version: Optional[str]) -> Optional[float]:
    """The T the trained engine recorded in model_version, or None if none was applied."""
    match = _TEMPERATURE_IN_VERSION.search(model_version or "")
    return float(match.group(1)) if match else None


def is_probability_vector(scores: Iterable[float]) -> bool:
    values = list(scores)
    if not values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return False
    return all(0.0 <= v <= 1.0 for v in values) and abs(sum(values) - 1.0) <= _SUM_TOLERANCE


@dataclass(frozen=True)
class ScoreTerms:
    calibrated: bool
    temperature: Optional[float]
    highest: str
    per_class: str
    distribution: str
    explanation: str


def score_terms(model_version: Optional[str], class_scores: Optional[list]) -> ScoreTerms:
    t = recorded_temperature(model_version)
    grades = sorted(int(c.get("grade", -1)) for c in (class_scores or []))
    calibrated = (
        t is not None and math.isfinite(t) and t > 0
        and grades == [0, 1, 2, 3, 4]
        and is_probability_vector(c.get("score") for c in class_scores)
    )
    if calibrated:
        return ScoreTerms(True, t, "Highest Class Probability", "Temperature-Scaled Class Probability",
                          "Five-Class Probability Distribution", CALIBRATION_EXPLANATION)
    return ScoreTerms(False, None, "Highest Model-Generated Class Score", "Model-Generated Class Score",
                      "Five-Class Model-Generated Score Distribution", UNCALIBRATED_EXPLANATION)
