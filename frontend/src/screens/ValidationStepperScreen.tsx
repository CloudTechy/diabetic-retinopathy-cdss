import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  RotateCcw,
  ArrowRight,
  Loader2
} from 'lucide-react';
import { AssessmentRecord, GateResult } from '../types/clinical';
import { ValidationStepper } from '../components/ValidationStepper';

interface ValidationStepperScreenProps {
  assessment: AssessmentRecord;
  onValidationComplete: (passed: boolean) => void;
  onRetry: () => void;
}

export const ValidationStepperScreen: React.FC<ValidationStepperScreenProps> = ({
  assessment,
  onValidationComplete,
  onRetry,
}) => {
  // Stepper Animation State: 1 -> 2 -> 3
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(1);
  const [stepperGates, setStepperGates] = useState<GateResult[]>([
    {
      gateIndex: 1,
      name: 'Gate 1',
      title: 'File Integrity & Safe Decode',
      status: 'in_progress',
      metric: 'Checking file signature, size and decodability...',
    },
    {
      gateIndex: 2,
      name: 'Gate 2',
      title: 'Technical retinal-image relevance',
      status: 'pending',
      metric: 'Pending Gate 1 completion...',
    },
    {
      gateIndex: 3,
      name: 'Gate 3',
      title: 'Technical Quality & Sharpness',
      status: 'pending',
      metric: 'Pending Gate 2 completion...',
    },
  ]);

  const [hasCompleted, setHasCompleted] = useState<boolean>(false);
  const [finalStatus, setFinalStatus] = useState<'passed' | 'failed' | 'incomplete'>('passed');
  const [failedGate, setFailedGate] = useState<GateResult | null>(null);

  useEffect(() => {
    // Replay the SERVER'S stored gate results one gate at a time. Pacing is
    // the only thing added here: each gate is shown with the status the API
    // returned ('passed', 'failed' or 'pending'). An earlier revision rewrote
    // every gate after the first failure to 'passed', so a gate the server
    // reported as 'pending' / "Not evaluated" got a green tick and the screen
    // announced "All 3 Validation Gates Successfully Passed".
    const server = assessment.validationGates;
    const inProgress = [
      'Checking file signature, size and decodability...',
      'Checking geometry and colour-profile thresholds...',
      'Calculating Laplacian blur variance and contrast...',
    ];
    const timers: ReturnType<typeof setTimeout>[] = [];

    const frame = (revealed: number, running: number | null) =>
      server.map((g, i) => {
        if (i < revealed) return g;                                   // the server's status, verbatim
        if (i === running) return { ...g, status: 'in_progress' as const, metric: inProgress[i] };
        return { ...g, status: 'pending' as const, metric: `Pending Gate ${i} completion...` };
      });

    if (server.length === 0) {
      // no gate list at all: nothing can be announced as passed
      setFailedGate(null);
      setFinalStatus('incomplete');
      setHasCompleted(true);
      return () => timers.forEach(clearTimeout);
    }

    const finish = (revealed: number) => {
      setStepperGates(frame(revealed, null));
      const stopped = server.slice(0, revealed).find((g) => g.status !== 'passed');
      if (!stopped) {
        setFinalStatus(revealed === server.length ? 'passed' : 'incomplete');
      } else if (stopped.status === 'failed') {
        setFailedGate(stopped);
        setFinalStatus('failed');
      } else {
        setFailedGate(stopped);
        setFinalStatus('incomplete');
      }
      setHasCompleted(true);
    };

    const step = (k: number) => {
      if (k >= server.length) { finish(server.length); return; }
      setStepperGates(frame(k, k));
      setCurrentStepIndex(k + 1);
      timers.push(setTimeout(() => {
        const g = server[k];
        if (g.status === 'passed') step(k + 1);
        else finish(k + 1);
      }, k === 0 ? 800 : 900));
    };

    step(0);
    return () => timers.forEach(clearTimeout);
  }, [assessment]);

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Page Header */}
      <div className="border-b border-slate-200 pb-4 flex items-start justify-between">
        <div>
          <h2 className="text-xl font-black text-slate-900 leading-tight">
            Fail-Closed 3-Gate Validation Pipeline
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Validating input parameters prior to automated model inference. Failures strictly abort inference to preserve data integrity and prevent artifact analysis.
          </p>
        </div>
        <span className="inline-flex items-center px-2.5 py-1 rounded bg-blue-50 text-blue-800 text-xs font-mono font-semibold border border-blue-200">
          Screen 4: Validation
        </span>
      </div>

      {/* Stepper Component */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <ValidationStepper gates={stepperGates} activeGateIndex={currentStepIndex} />
      </div>

      {/* Main Validation Diagnostic Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
        {/* Header Preview of Ingested Image */}
        <div className="flex flex-col sm:flex-row items-center gap-6 pb-6 border-b border-slate-100">
          <div className="relative w-28 h-28 rounded-full overflow-hidden border-2 border-slate-300 shadow-md bg-black flex-shrink-0">
            <img
              src={assessment.imageUrl}
              alt="Ingested retinal fundus"
              className="w-full h-full object-cover"
            />
          </div>

          <div className="space-y-1 text-xs text-left w-full">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-900 text-sm">
                Study ID: {assessment.patientId}
              </span>
              <span className="font-mono text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200 font-semibold">
                {assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
              </span>
            </div>
            <p className="text-slate-500">Camera: {assessment.cameraModel || 'Not recorded'}</p>
            <p className="text-slate-500 font-mono text-[11px]">
              SHA-256: {assessment.qualityMetrics.sha256Hash.substring(0, 24)}...
            </p>
          </div>
        </div>

        {/* Live Status Message */}
        {!hasCompleted && (
          <div className="p-4 bg-blue-50/60 border border-blue-200 rounded-xl flex items-center space-x-3 text-xs text-blue-900">
            <Loader2 className="w-5 h-5 text-blue-600 animate-spin flex-shrink-0" />
            <div>
              <p className="font-bold">Automated Technical Screening in Progress</p>
              <p className="text-[11px] text-blue-700">
                Checking file integrity, geometry and colour profile, and Laplacian sharpness...
              </p>
            </div>
          </div>
        )}

        {/* Gate Failure Clinical Guidance Display */}
        {hasCompleted && finalStatus === 'failed' && failedGate && (
          <div className="space-y-4">
            <div className="p-4 bg-rose-50 border-2 border-rose-300 rounded-xl space-y-3">
              <div className="flex items-start space-x-3">
                <ShieldAlert className="w-6 h-6 text-rose-600 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <h4 className="font-bold text-rose-900 text-sm">
                    Fail-Closed Rejection: {failedGate.title} (Gate {failedGate.gateIndex})
                  </h4>
                  <p className="text-xs text-rose-800 leading-relaxed font-medium">
                    {failedGate.rejectionReason ||
                      'The submitted image failed the automated technical validation criteria.'}
                  </p>
                </div>
              </div>

              {/* Actionable Next Step */}
              <div className="p-3 bg-white rounded-lg border border-rose-200 text-xs text-slate-800 space-y-1">
                <strong className="text-rose-900 block font-bold">Actionable Clinical Next Step:</strong>
                <p className="text-[11px] leading-relaxed">
                  {failedGate.clinicalAction ||
                    'Please recapture the retinal photograph ensuring steady patient fixation, correct camera working distance, and clean lens surface.'}
                </p>
              </div>

              {/* Quality Safeguard Banner */}
              <div className="p-2.5 bg-rose-100/70 rounded text-[11px] text-rose-900 border border-rose-200">
                <strong>Quality Safeguard (Fail-Closed Image Technical Acceptance):</strong> Image does not meet technical quality requirements. To prevent artifact processing and model misclassification, automated evaluation has been aborted.
              </div>
            </div>

            <div className="flex justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={onRetry}
                className="px-4 py-2 text-xs font-bold text-slate-700 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg shadow-xs flex items-center gap-1.5"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Recapture & Re-Upload Image
              </button>
            </div>
          </div>
        )}

        {/* Not evaluated: the server holds no verdict for a gate */}
        {hasCompleted && finalStatus === 'incomplete' && (
          <div className="space-y-4">
            <div className="p-4 bg-amber-50 border-2 border-amber-300 rounded-xl flex items-start space-x-3 text-xs">
              <ShieldAlert className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="font-bold text-amber-900 text-sm">Validation not completed for this record</h4>
                <p className="text-amber-800 leading-relaxed">
                  {failedGate
                    ? `Gate ${failedGate.gateIndex} (${failedGate.title}) is "${failedGate.status}": ${failedGate.metric || 'no result was recorded'}.`
                    : 'Not every gate has a recorded result.'}{' '}
                  No decision support is available until an image has been validated.
                </p>
              </div>
            </div>
            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={onRetry}
                className="px-4 py-2 text-xs font-bold text-slate-700 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg shadow-xs flex items-center gap-1.5"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Upload an image for this assessment
              </button>
            </div>
          </div>
        )}

        {/* Gate Passed Success Display */}
        {hasCompleted && finalStatus === 'passed' && (
          <div className="space-y-4">
            <div className="p-4 bg-emerald-50 border border-emerald-300 rounded-xl flex items-start space-x-3">
              <CheckCircle2 className="w-6 h-6 text-emerald-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-1 text-xs">
                <h4 className="font-bold text-emerald-950 text-sm">
                  All 3 Validation Gates Successfully Passed
                </h4>
                <p className="text-emerald-800 leading-relaxed">
                  The file decoded, and its geometry, colour profile, sharpness (Laplacian variance:{' '}
                  {assessment.qualityMetrics.laplacianVariance.toFixed(1)}) and contrast met the configured thresholds.
                </p>
              </div>
            </div>

            {/* Non-Diagnostic Microcopy Mandatory Notice */}
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600 leading-relaxed">
              <strong>Boundary notice:</strong> passing these heuristics confirms only that the input met the configured geometry, colour-profile, sharpness and contrast thresholds. It does not confirm retinal identity, anatomical correctness or clinical gradability. The grading decision is solely the responsibility of the reviewing medical professional.
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={() => onValidationComplete(true)}
                className="px-6 py-2.5 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm flex items-center gap-2 focus-visible:ring-2 focus-visible:ring-clinical-primary transition"
              >
                <span>Open Decision-Support Result Workspace</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
