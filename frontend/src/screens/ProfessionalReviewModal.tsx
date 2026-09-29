import React, { useState } from 'react';
import {
  X,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  UserCheck,
  ShieldCheck,
  FileCheck
} from 'lucide-react';
import {
  AssessmentRecord,
  AgreementType,
  ICDR_GRADES,
  ClinicianUser
} from '../types/clinical';
import { clinicalApi } from '../services/api';

interface ProfessionalReviewModalProps {
  assessment: AssessmentRecord;
  currentUser: ClinicianUser;
  onClose: () => void;
  onReviewSubmitted: (updated: AssessmentRecord) => void;
}

export const ProfessionalReviewModal: React.FC<ProfessionalReviewModalProps> = ({
  assessment,
  currentUser,
  onClose,
  onReviewSubmitted,
}) => {
  // Friction engineering: NO pre-selected default
  const [agreement, setAgreement] = useState<AgreementType | null>(null);
  const [certifiedGrade, setCertifiedGrade] = useState<number | null>(null);
  const [justificationNotes, setJustificationNotes] = useState<string>('');
  const [inconclusiveReason, setInconclusiveReason] = useState<string>('Media Opacity / Cataract');
  const [isConfirmed, setIsConfirmed] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const modelGrade = assessment.modelObservation?.primaryClassGrade ?? 2;

  // Validation: Clinician must select Agree/Disagree/Unable to determine and check confirmation
  const canProceed = agreement !== null && isConfirmed;

  const handleSelectAgreement = (type: AgreementType) => {
    setAgreement(type);
    if (type === 'agree') {
      setCertifiedGrade(modelGrade);
    } else if (type === 'inconclusive') {
      setCertifiedGrade(modelGrade);
    } else if (type === 'disagree') {
      if (certifiedGrade === modelGrade) {
        setCertifiedGrade(null);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canProceed) return;
    setIsSubmitting(true);

    try {
      const finalGrade = certifiedGrade ?? modelGrade;
      const updated = await clinicalApi.submitReview(assessment.id, {
        agreement,
        certifiedGrade: finalGrade,
        certifiedGradeLabel: ICDR_GRADES[finalGrade]?.label || 'Clinical Observation Recorded',
        justificationNotes: justificationNotes.trim() || undefined,
        inconclusiveReason: agreement === 'inconclusive' ? inconclusiveReason : undefined,
        referralPlan: 'Referral decisions are outside the scope of this research prototype.',
      });

      onReviewSubmitted(updated);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="review-modal-title"
      className="fixed inset-0 z-50 overflow-y-auto bg-black/60 backdrop-blur-xs flex items-center justify-center p-3 sm:p-6"
    >
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-300 max-w-2xl w-full overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <header className="px-6 py-4 bg-teal-800 text-white flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-teal-900/60 rounded-lg text-teal-200">
              <UserCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 id="review-modal-title" className="text-base font-bold leading-tight">
                Professional Review & Concurrence
              </h2>
              <p className="text-xs text-teal-200 font-mono">
                {assessment.id} • Patient {assessment.patientId} ({assessment.laterality})
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-teal-200 hover:text-white hover:bg-teal-700/50 transition focus-visible:ring-2 focus-visible:ring-white"
            aria-label="Close review dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        {/* Modal Scrollable Body */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6 space-y-6 text-xs">
          {/* AI Model Baseline Observation for Comparison */}
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Preliminary Model Attribution & Classification:
            </span>
            <div className="flex items-center justify-between text-slate-800">
              <span className="font-bold text-sm">
                {assessment.modelObservation?.primaryClassLabel} (Grade {modelGrade})
              </span>
              <span className="font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200">
                Score: {assessment.modelObservation?.primaryScore.toFixed(2)}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 italic bg-white p-2 rounded border border-slate-100 leading-relaxed">
              Note: The visual attribution highlights image regions that influenced the model output. It does not constitute verified lesion localisation or clinical interpretation.
            </p>
          </div>

          {/* Section 1: Professional Review Response */}
          <div className="space-y-2">
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider">
              1. Professional Assessment Response *
            </label>
            <p className="text-[11px] text-slate-500">
              Record your independent evaluation of the automated finding.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1" role="radiogroup">
              {/* Option A: Agree */}
              <button
                type="button"
                onClick={() => handleSelectAgreement('agree')}
                className={`p-3.5 rounded-xl border text-left transition flex flex-col justify-between space-y-2 focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                  agreement === 'agree'
                    ? 'border-emerald-500 bg-emerald-50/70 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
                role="radio"
                aria-checked={agreement === 'agree'}
              >
                <div className="flex items-center justify-between">
                  <CheckCircle2
                    className={`w-4 h-4 ${
                      agreement === 'agree' ? 'text-emerald-600' : 'text-slate-400'
                    }`}
                  />
                  <span className="text-[10px] uppercase font-mono font-bold text-emerald-800">
                    Concur
                  </span>
                </div>
                <div className="font-bold text-slate-900">Agree</div>
                <p className="text-[11px] text-slate-500">
                  Retinal findings concur with model candidate class.
                </p>
              </button>

              {/* Option B: Disagree */}
              <button
                type="button"
                onClick={() => handleSelectAgreement('disagree')}
                className={`p-3.5 rounded-xl border text-left transition flex flex-col justify-between space-y-2 focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                  agreement === 'disagree'
                    ? 'border-amber-500 bg-amber-50/70 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
                role="radio"
                aria-checked={agreement === 'disagree'}
              >
                <div className="flex items-center justify-between">
                  <AlertTriangle
                    className={`w-4 h-4 ${
                      agreement === 'disagree' ? 'text-amber-600' : 'text-slate-400'
                    }`}
                  />
                  <span className="text-[10px] uppercase font-mono font-bold text-amber-800">
                    Override
                  </span>
                </div>
                <div className="font-bold text-slate-900">Disagree</div>
                <p className="text-[11px] text-slate-500">
                  Clinician identifies alternate retinal findings.
                </p>
              </button>

              {/* Option C: Inconclusive */}
              <button
                type="button"
                onClick={() => handleSelectAgreement('inconclusive')}
                className={`p-3.5 rounded-xl border text-left transition flex flex-col justify-between space-y-2 focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                  agreement === 'inconclusive'
                    ? 'border-blue-500 bg-blue-50/70 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
                role="radio"
                aria-checked={agreement === 'inconclusive'}
              >
                <div className="flex items-center justify-between">
                  <HelpCircle
                    className={`w-4 h-4 ${
                      agreement === 'inconclusive' ? 'text-blue-600' : 'text-slate-400'
                    }`}
                  />
                  <span className="text-[10px] uppercase font-mono font-bold text-blue-800">
                    Indeterminate
                  </span>
                </div>
                <div className="font-bold text-slate-900">Unable to Determine</div>
                <p className="text-[11px] text-slate-500">
                  Image artifacts, opacity, or ungradable ambiguity.
                </p>
              </button>
            </div>
          </div>

          {/* Inconclusive Category Dropdown (if Inconclusive selected) */}
          {agreement === 'inconclusive' && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-xl space-y-2">
              <label htmlFor="inconclusive-reason" className="block text-xs font-bold text-blue-950">
                Inconclusive Ambiguity Category *
              </label>
              <select
                id="inconclusive-reason"
                value={inconclusiveReason}
                onChange={(e) => setInconclusiveReason(e.target.value)}
                className="w-full text-xs p-2 border border-blue-300 rounded-lg bg-white text-slate-800"
              >
                <option value="Media Opacity / Cataract">Media Opacity / Advanced Cataract</option>
                <option value="Non-DR Retinal Pathology (e.g. RVO, AMD)">Non-DR Retinal Pathology (e.g. Vein Occlusion, AMD)</option>
                <option value="Uncertain Peripheral Lesion">Uncertain Peripheral Lesion Bordering Field Edge</option>
                <option value="Severe Reflection Artifact">Severe Corneal / Lens Reflection Artifact</option>
              </select>
            </div>
          )}

          {/* Clinical Observation Notes (Optional per supervisor directive) */}
          <div className="space-y-2 p-3.5 bg-slate-50 border border-slate-200 rounded-xl">
            <div className="flex items-center justify-between">
              <label htmlFor="justification-text" className="block text-xs font-bold text-slate-800">
                Professional Observation Notes (Optional)
              </label>
              <span className="text-[10px] text-slate-400 font-mono">
                Optional clinical commentary
              </span>
            </div>
            <textarea
              id="justification-text"
              rows={2}
              value={justificationNotes}
              onChange={(e) => setJustificationNotes(e.target.value)}
              placeholder="Enter optional clinical findings or observations (e.g., Microaneurysm cluster noted in temporal macula, or artifact on lens)..."
              className="w-full text-xs p-2.5 border border-slate-300 rounded-lg bg-white shadow-xs focus:ring-2 focus:ring-teal-500 focus:outline-none"
            />
          </div>

          {/* Review Confirmation Checkbox (Anti-Automation Bias Guardrail) */}
          <div className="p-3 bg-teal-50/60 rounded-xl border border-teal-200 flex items-start gap-2.5">
            <input
              id="confirm-checkbox"
              type="checkbox"
              checked={isConfirmed}
              onChange={(e) => setIsConfirmed(e.target.checked)}
              className="mt-0.5 w-4 h-4 text-clinical-primary rounded border-slate-300 focus:ring-clinical-primary"
            />
            <label htmlFor="confirm-checkbox" className="text-xs text-teal-950 font-medium cursor-pointer leading-relaxed">
              I confirm that I have evaluated the preliminary model observation and visual attribution, and this submission reflects my independent professional assessment.
            </label>
          </div>

          {/* Reviewer Information Preview */}
          <div className="p-3 bg-slate-100 rounded-xl border border-slate-200 text-slate-600 flex items-center justify-between text-[11px]">
            <div>
              <span className="font-bold text-slate-800 block">Reviewer: {currentUser.name}</span>
              <span className="font-mono text-slate-500">
                {currentUser.facility}
              </span>
            </div>
            <div className="flex items-center text-teal-700 font-mono font-semibold">
              <ShieldCheck className="w-4 h-4 mr-1 text-teal-600" />
              Independent Professional Review
            </div>
          </div>

          {/* Modal Footer Actions */}
          <div className="flex items-center justify-between pt-3 border-t border-slate-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-lg border border-slate-300 transition"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={!canProceed || isSubmitting}
              className="px-6 py-2.5 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm transition disabled:opacity-40 flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-clinical-primary"
            >
              <FileCheck className="w-4 h-4" />
              <span>{isSubmitting ? 'Recording Review...' : 'Submit Professional Review'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
