/**
 * What a model output may be called on screen.
 *
 * The trained engine applies temperature scaling fitted on the validation
 * partition and records the temperature in the persisted model version, which
 * the API returns as calibrationTemperature. Only an output that carries that
 * record, and whose five values form a valid probability distribution, is
 * labelled a class probability. Anything else (a record stored before
 * calibration, a malformed vector) keeps the neutral
 * "model-generated class score" wording, so an uncalibrated value is never
 * relabelled as a probability.
 *
 * Grad-CAM wording (activation, attribution, heatmap) is deliberately absent
 * here: it describes where the model looked, not how likely a class is.
 */
import type { ModelObservation } from '../types/clinical';

export const CALIBRATION_EXPLANATION =
  'Class probabilities are produced using temperature scaling fitted on the validation partition. ' +
  'They support professional review and do not establish a diagnosis.';

export const UNCALIBRATED_EXPLANATION =
  'This record carries no verified calibration metadata, so its values are shown as model-generated class ' +
  'scores, not probabilities. They support professional review and do not establish a diagnosis.';

// Tolerance on the sum of the five stored values (they are float64 softmax outputs).
const SUM_TOLERANCE = 1e-3;

export interface ProbabilityVectorCheck {
  count: number;
  allFinite: boolean;
  allInUnitInterval: boolean;
  sum: number;
  sumsToOne: boolean;
}

export function checkProbabilityVector(scores: readonly number[]): ProbabilityVectorCheck {
  const allFinite = scores.every((s) => typeof s === 'number' && Number.isFinite(s));
  const allInUnitInterval = allFinite && scores.every((s) => s >= 0 && s <= 1);
  const sum = allFinite ? scores.reduce((a, b) => a + b, 0) : Number.NaN;
  return {
    count: scores.length,
    allFinite,
    allInUnitInterval,
    sum,
    sumsToOne: Number.isFinite(sum) && Math.abs(sum - 1) <= SUM_TOLERANCE,
  };
}

/** True only for an output with recorded calibration and a valid five-class probability vector. */
export function hasVerifiedCalibration(observation: ModelObservation | null | undefined): boolean {
  if (!observation) return false;
  const t = observation.calibrationTemperature;
  if (typeof t !== 'number' || !Number.isFinite(t) || t <= 0) return false;
  const grades = observation.classScores.map((c) => c.grade).sort((a, b) => a - b);
  if (grades.join(',') !== '0,1,2,3,4') return false;
  const check = checkProbabilityVector(observation.classScores.map((c) => c.score));
  return check.allInUnitInterval && check.sumsToOne;
}

export interface ScoreTerms {
  calibrated: boolean;
  /** Label for the predicted grade's value. */
  highest: string;
  /** Label for one class's value. */
  perClass: string;
  /** Title of the five-class list. */
  distribution: string;
  /** Short lower-case tag used in lists and history rows. */
  short: string;
  /** The one-sentence explanation shown once per view. */
  explanation: string;
}

export function scoreTerms(observation: ModelObservation | null | undefined): ScoreTerms {
  if (hasVerifiedCalibration(observation)) {
    return {
      calibrated: true,
      highest: 'Highest Class Probability',
      perClass: 'Temperature-Scaled Class Probability',
      distribution: 'Five-Class Probability Distribution',
      short: 'temperature-scaled class probability',
      explanation: CALIBRATION_EXPLANATION,
    };
  }
  return {
    calibrated: false,
    highest: 'Highest Model-Generated Class Score',
    perClass: 'Model-Generated Class Score',
    distribution: 'Five-Class Model-Generated Score Distribution',
    short: 'model-generated class score',
    explanation: UNCALIBRATED_EXPLANATION,
  };
}
