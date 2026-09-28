import React, { useState } from 'react';
import {
  X,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Lock,
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
  const [referralPlan, setReferralPlan] = useState<string>('Routine 12-month diabetic eye screening recall.');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Anti-Automation Bias Verification Dialog State
  const [showConfirmDialog, setShowConfirmDialog] = useState<boolean>(false);

  const modelGrade = assessment.modelObservation?.primaryClassGrade ?? 2;

  // Validation rules: Per supervisor directive, observation rationale is optional and arbitrary 15-char restriction is removed
  const isGradeSelected = certifiedGrade !== null;
  const canProceed = agreement !== null && isGradeSelected;

  const handleSelectAgreement = (type: AgreementType) => {
    setAgreement(type);
    if (type === 'agree') {
      // Pre-fill with the candidate grade when agreeing
      setCertifiedGrade(modelGrade);
    } else if (type === 'disagree') {
      // If disagreeing, clear if it matches model grade to force conscious human selection
      if (certifiedGrade === modelGrade) {
        setCertifiedGrade(null);
      }
    }
  };

  const handleOpenConfirm = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canProceed) return;
    setShowConfirmDialog(true);
  };

  const handleConfirmAndSign = async () => {
    if (certifiedGrade === null || agreement === null) return;
    setIsSubmitting(true);

    try {
      const updated = await clinicalApi.submitReview(assessment.id, {
        agreement,
        certifiedGrade,
        certifiedGradeLabel: ICDR_GRADES[certifiedGrade].label,
        justificationNotes: justificationNotes.trim() || undefined,
        inconclusiveReason: agreement === 'inconclusive' ? inconclusiveReason : undefined,
        referralPlan,
      });

      onReviewSubmitted(updated);
    } finally {
      setIsSubmitting(false);
      setShowConfirmDialog(false);
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
                Official Clinical Evaluation & Sign-off
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
        <form onSubmit={handleOpenConfirm} className="flex-1 overflow-y-auto p-6 space-y-6 text-xs">
          {/* AI Model Baseline Observation for Comparison */}
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Reference AI Computational Observation:
            </span>
            <div className="flex items-center justify-between text-slate-800">
              <span className="font-bold text-sm">
                {assessment.modelObservation?.primaryClassLabel} (Grade {modelGrade})
              </span>
              <span className="font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200">
                Score: {assessment.modelObservation?.primaryScore.toFixed(2)}
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              Target region: {assessment.modelObservation?.topActivationRegion}
            </p>
          </div>

          {/* Section 1: Clinical Agreement Tri-State Selector */}
          <div className="space-y-2">
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider">
              1. Clinician Agreement with Preliminary Model Observation *
            </label>
            <p className="text-[11px] text-slate-500">
              Select your clinical assessment agreement. (Anti-automation bias: no pre-selected default).
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
                <div className="font-bold text-slate-900">Agree with AI Observation</div>
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
                <div className="font-bold text-slate-900">Disagree / Override</div>
                <p className="text-[11px] text-slate-500">
                  Clinician identifies a different ICDR clinical stage.
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
                  Clinical artifacts, opacity, or ungradable ambiguity.
                </p>
              </button>
            </div>
          </div>

          {/* Section 2: Certified Classification (ICDR 5-Grade) */}
          <div className="space-y-2">
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider">
              2. Clinician Certified Classification (ICDR 5-Grade) *
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-5 gap-2" role="radiogroup">
              {Object.values(ICDR_GRADES).map((item) => {
                const isSelected = certifiedGrade === item.grade;
                return (
                  <button
                    key={item.grade}
                    type="button"
                    onClick={() => setCertifiedGrade(item.grade)}
                    className={`p-3 rounded-lg border text-left transition focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                      isSelected
                        ? 'border-teal-600 bg-teal-50 ring-2 ring-teal-500 font-bold'
                        : 'border-slate-200 hover:bg-slate-50 text-slate-700'
                    }`}
                    role="radio"
                    aria-checked={isSelected}
                  >
                    <span className="block text-xs font-bold text-slate-900">Grade {item.grade}</span>
                    <span className="block text-[11px] text-slate-600 truncate mt-0.5">
                      {item.label.split(':')[1]?.trim() || item.label}
                    </span>
                  </button>
                );
              })}
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

          {/* Clinical Observation Rationale (Optional per supervisor directive) */}
          <div className="space-y-2 p-3.5 bg-slate-50 border border-slate-200 rounded-xl">
            <div className="flex items-center justify-between">
              <label htmlFor="justification-text" className="block text-xs font-bold text-slate-800">
                Clinical Observation Rationale (Optional)
              </label>
              <span className="text-[10px] text-slate-400 font-mono">
                Optional observation notes
              </span>
            </div>
            <textarea
              id="justification-text"
              rows={2}
              value={justificationNotes}
              onChange={(e) => setJustificationNotes(e.target.value)}
              placeholder="Enter optional clinical findings or override rationale (e.g., Focal microaneurysms noted in macular zone, or 4-2-1 rule verification)..."
              className="w-full text-xs p-2.5 border border-slate-300 rounded-lg bg-white shadow-xs focus:ring-2 focus:ring-teal-500 focus:outline-none"
            />
          </div>

          {/* Section 3: Management & Referral Protocol */}
          <div className="space-y-2">
            <label htmlFor="referral-plan" className="block text-xs font-bold text-slate-900 uppercase tracking-wider">
              3. Management & Referral Recommendation Protocol *
            </label>
            <select
              id="referral-plan"
              value={referralPlan}
              onChange={(e) => setReferralPlan(e.target.value)}
              className="w-full text-xs p-2.5 border border-slate-300 rounded-lg bg-white text-slate-800 focus:ring-2 focus:ring-clinical-primary"
            >
              <option value="Routine 12-month diabetic eye screening recall.">
                Routine 12-month recall (No or mild DR without macular edema)
              </option>
              <option value="Repeat photograph & clinical assessment in 3–6 months.">
                Repeat photograph & assessment in 3–6 months (Stable moderate NPDR)
              </option>
              <option value="Referral to secondary care / hospital medical retina clinic.">
                Referral to medical retina clinic (Moderate to severe NPDR / DMO)
              </option>
              <option value="Urgent vitreoretinal / anti-VEGF referral within 2 weeks.">
                Urgent referral within 2 weeks (PDR, vitreous hemorrhage, active NVD)
              </option>
            </select>
          </div>

          {/* Clinician Digital Signature Preview */}
          <div className="p-3 bg-slate-100 rounded-xl border border-slate-200 text-slate-600 flex items-center justify-between text-[11px]">
            <div>
              <span className="font-bold text-slate-800 block">Signatory: {currentUser.name}</span>
              <span className="font-mono text-slate-500">
                Lic: {currentUser.licenseNumber} • {currentUser.facility}
              </span>
            </div>
            <div className="flex items-center text-teal-700 font-mono font-semibold">
              <Lock className="w-3.5 h-3.5 mr-1 text-teal-600" />
              Professional Review Sign-off
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
              <ShieldCheck className="w-4 h-4" />
              <span>Review and Confirm Sign-off</span>
            </button>
          </div>
        </form>
      </div>

      {/* Explicit Confirmation Dialog (Anti-Automation Bias Guardrail) */}
      {showConfirmDialog && certifiedGrade !== null && (
        <div
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="confirm-dialog-title"
          className="fixed inset-0 z-60 bg-black/70 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div className="bg-white rounded-2xl shadow-2xl border border-slate-300 max-w-lg w-full p-6 space-y-4 animate-in zoom-in-95 duration-150">
            <div className="flex items-start space-x-3 text-teal-800">
              <FileCheck className="w-6 h-6 text-clinical-primary flex-shrink-0 mt-0.5" />
              <div>
                <h3 id="confirm-dialog-title" className="text-base font-bold text-slate-900">
                  Confirm Official Clinical Certification
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Mandatory explicit verification checkpoint prior to signing.
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-500">Patient Reference:</span>
                <span className="font-bold text-slate-900">{assessment.patientId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Eye Laterality:</span>
                <span className="font-bold text-slate-900">
                  {assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Model Candidate Class:</span>
                <span className="font-mono text-slate-700">
                  {assessment.modelObservation?.primaryClassLabel} ({assessment.modelObservation?.primaryScore.toFixed(2)})
                </span>
              </div>
              <div className="flex justify-between pt-2 border-t border-slate-200">
                <span className="font-bold text-slate-800">Certified Human Diagnosis:</span>
                <span className="font-bold text-teal-800 font-mono">
                  {ICDR_GRADES[certifiedGrade].label}
                </span>
              </div>
            </div>

            <div className="p-3 bg-amber-50 rounded-lg border border-amber-200 text-[11px] text-amber-900 leading-relaxed">
              <strong>Medical-Legal Traceability Notice:</strong> Once signed, this record becomes an immutable clinical document tied to your GMC/professional registration. It cannot be altered without an addendum audit entry.
            </div>

            <div className="flex justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={() => setShowConfirmDialog(false)}
                className="px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100 rounded-lg border border-slate-300"
              >
                Return to Edit
              </button>
              <button
                type="button"
                disabled={isSubmitting}
                onClick={handleConfirmAndSign}
                className="px-5 py-2 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-clinical-primary"
              >
                <Lock className="w-3.5 h-3.5" />
                <span>Confirm and Sign Record</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
