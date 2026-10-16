import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import type { ModelObservation } from '../types/clinical';
import {
  CALIBRATION_EXPLANATION,
  checkProbabilityVector,
  hasVerifiedCalibration,
  scoreTerms,
} from './scoreTerminology';

// The calibrated vector the API served for REC-2026-6954 (held-out image c6c2bad91f23).
const SERVED = [0.000693969914342396, 0.007727745597431056, 0.8843014501550295, 0.05967577356239384, 0.04760106077080314];

const observation = (overrides: Partial<ModelObservation> = {}): ModelObservation => ({
  primaryClassGrade: 2,
  primaryClassLabel: 'Moderate NPDR',
  primaryScore: SERVED[2],
  classScores: SERVED.map((score, grade) => ({ grade, label: `Grade ${grade}`, score })),
  targetLayer: 'features.8 (Conv2d Bottleneck Residual)',
  topActivationRegion: 'central cell',
  modelVersion: 'EfficientNet-B0-DR-v1 (fixed weights; temperature-scaled T=1.093921, ts-v2)',
  inferenceTimestamp: '2026-10-15T23:48:49Z',
  calibrationTemperature: 1.093921,
  ...overrides,
});

describe('checkProbabilityVector', () => {
  it('accepts the served calibrated vector: finite, within 0-1, summing to 1', () => {
    const c = checkProbabilityVector(SERVED);
    expect(c.allFinite && c.allInUnitInterval && c.sumsToOne).toBe(true);
    expect(Math.abs(c.sum - 1)).toBeLessThan(1e-9);
  });
  it('rejects non-finite, out-of-range and non-normalised vectors', () => {
    expect(checkProbabilityVector([0.5, Number.NaN, 0.5]).allFinite).toBe(false);
    expect(checkProbabilityVector([1.2, -0.2]).allInUnitInterval).toBe(false);
    expect(checkProbabilityVector([0.2, 0.2, 0.2]).sumsToOne).toBe(false);
  });
});

describe('scoreTerms', () => {
  it('labels a calibrated output as class probabilities', () => {
    const t = scoreTerms(observation());
    expect(t.calibrated).toBe(true);
    expect(t.highest).toBe('Highest Class Probability');
    expect(t.perClass).toBe('Temperature-Scaled Class Probability');
    expect(t.distribution).toBe('Five-Class Probability Distribution');
    expect(t.explanation).toBe(CALIBRATION_EXPLANATION);
  });

  it('keeps model-generated class score wording when no calibration is recorded', () => {
    for (const t of [scoreTerms(observation({ calibrationTemperature: null })),
                     scoreTerms(observation({ calibrationTemperature: undefined })),
                     scoreTerms(null)]) {
      expect(t.calibrated).toBe(false);
      expect(t.perClass).toBe('Model-Generated Class Score');
      expect(`${t.highest} ${t.perClass} ${t.distribution} ${t.short}`).not.toMatch(/probabilit/i);
    }
  });

  it('does not call a malformed vector a probability even with a recorded temperature', () => {
    const bad = observation({ classScores: SERVED.map((s, grade) => ({ grade, label: `G${grade}`, score: s * 2 })) });
    expect(hasVerifiedCalibration(bad)).toBe(false);
    const missing = observation({ classScores: SERVED.slice(0, 4).map((score, grade) => ({ grade, label: `G${grade}`, score })) });
    expect(hasVerifiedCalibration(missing)).toBe(false);
    expect(hasVerifiedCalibration(observation({ calibrationTemperature: 0 }))).toBe(false);
  });

  it('never uses the obsolete wording or describes probabilities as feature activations', () => {
    for (const t of [scoreTerms(observation()), scoreTerms(null)]) {
      const text = Object.values(t).join(' ');
      expect(text).not.toMatch(/Relative Score|Score Breakdown|feature activation/i);
    }
  });

  it('displays each value as the served value rounded to two decimals', () => {
    expect(SERVED.map((s) => s.toFixed(2))).toEqual(['0.00', '0.01', '0.88', '0.06', '0.05']);
  });
});

describe('active calibrated-output views', () => {
  // The views that display model output. None may keep the wording that
  // described calibrated probabilities as relative scores or feature activations.
  const VIEWS = [
    '../components/ScoreDistributionCard.tsx',
    '../screens/DecisionSupportScreen.tsx',
    '../screens/ProfessionalReviewModal.tsx',
    '../screens/CompletedAssessmentScreen.tsx',
    '../screens/RecordHistoryScreen.tsx',
    '../screens/DashboardScreen.tsx',
  ];
  it.each(VIEWS)('%s uses no obsolete score wording', (view) => {
    const source = readFileSync(new URL(view, import.meta.url), 'utf8');
    expect(source).not.toMatch(/Relative Score|Score Breakdown|feature activations|algorithmic association|do not represent clinical probability/i);
  });
  it('keeps the Grad-CAM labels distinct from probability labels', () => {
    const card = readFileSync(new URL('../components/ScoreDistributionCard.tsx', import.meta.url), 'utf8');
    expect(card).toContain('Explainability Attribution Insight (Grad-CAM)');
    expect(card).toContain('Peak Activation Region');
    expect(card).not.toMatch(/Activation[^<\n]*Probability|Probability[^<\n]*Activation/);
  });
});
