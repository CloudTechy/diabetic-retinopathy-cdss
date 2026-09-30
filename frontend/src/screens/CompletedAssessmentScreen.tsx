import React, { useState } from 'react';
import {
  ShieldCheck,
  Lock,
  Download,
  Printer,
  Clock,
  Eye,
  Cpu,
  UserCheck,
  Layers,
  ArrowLeft
} from 'lucide-react';
import { AssessmentRecord, ICDR_GRADES } from '../types/clinical';
import { FundusViewer } from '../components/FundusViewer';
import { AuditDrawer } from '../components/AuditDrawer';
import { clinicalApi } from '../services/api';

interface CompletedAssessmentScreenProps {
  assessment: AssessmentRecord;
  onBackToDashboard: () => void;
}

export const CompletedAssessmentScreen: React.FC<CompletedAssessmentScreenProps> = ({
  assessment,
  onBackToDashboard,
}) => {
  const [showAuditDrawer, setShowAuditDrawer] = useState<boolean>(false);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  const review = assessment.clinicianReview;
  const reviewerGradeInfo = review ? ICDR_GRADES[review.reviewerAssessedGrade] : null;

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      await clinicalApi.downloadReportPdf(assessment);
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Top Navigation & Status Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <button
          onClick={onBackToDashboard}
          className="inline-flex items-center text-xs font-semibold text-slate-600 hover:text-slate-900 transition focus-visible:ring-2 focus-visible:ring-clinical-primary rounded-md p-1"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" />
          Return to Assessment Worklist
        </button>

        {/* Lock Status Indicator */}
        <div className="flex items-center space-x-2 bg-teal-50 border border-teal-200 px-3 py-1.5 rounded-lg text-xs text-teal-900 font-semibold shadow-xs">
          <ShieldCheck className="w-4 h-4 text-teal-600 flex-shrink-0" />
          <span>Assessment Finalized & Signed — Record Immutable</span>
          <Lock className="w-3.5 h-3.5 text-teal-700 ml-1" />
        </div>
      </div>

      {/* Main Encounter Summary Card */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 border-b border-slate-100 pb-4">
          <div>
            <span className="text-[10px] font-mono uppercase text-slate-400 font-bold">
              Finalized Clinical Consultation Report
            </span>
            <h2 className="text-xl font-black text-slate-900 leading-tight">
              {assessment.id} • Patient {assessment.patientId}
            </h2>
          </div>

          <div className="flex items-center space-x-2">
            <span
              className={`inline-flex items-center px-3 py-1 rounded text-xs font-mono font-bold ${
                assessment.laterality === 'OD'
                  ? 'bg-teal-50 text-teal-800 border border-teal-200'
                  : 'bg-blue-50 text-blue-800 border border-blue-200'
              }`}
            >
              <Eye className="w-3.5 h-3.5 mr-1" />
              {assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
            </span>

            <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded border border-slate-200">
              Acquired: {new Date(assessment.acquisitionDate).toLocaleDateString()}
            </span>
          </div>
        </div>

        {/* Assessment Integrity Hash */}
        <div className="p-3 bg-slate-900 rounded-xl text-slate-300 text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Lock className="w-4 h-4 text-teal-400 flex-shrink-0" />
            <span className="font-semibold text-slate-200">Assessment Integrity Hash:</span>
          </div>
          <span className="font-mono text-[11px] text-teal-300 bg-slate-800 px-2 py-0.5 rounded border border-slate-700 break-all">
            {assessment.qualityMetrics.sha256Hash}
          </span>
        </div>
      </div>

      {/* Side-by-Side Verification Summary (Strict Demarcation: AI Output vs Certified Clinician) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
        {/* Box A: AI Preliminary Domain (Slate / Neutral Gray Container) */}
        <div
          role="region"
          aria-labelledby="ai-domain-heading"
          className="bg-slate-50 rounded-2xl border-2 border-slate-300 p-6 shadow-xs flex flex-col justify-between space-y-4"
        >
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-slate-200 text-slate-700">
                <Cpu className="w-3 h-3 mr-1 text-slate-600" />
                [AI Assistive Engine]
              </span>
              <span className="text-[10px] text-slate-500 font-mono">
                {assessment.modelObservation?.modelVersion || 'EfficientNet-B0'}
              </span>
            </div>

            <h3 id="ai-domain-heading" className="text-base font-bold text-slate-900">
              Preliminary Model Observation
            </h3>

            <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-semibold">
                Candidate Classification & Score
              </span>
              <div className="flex justify-between items-center">
                <span className="font-bold text-slate-900 text-sm">
                  {assessment.modelObservation?.primaryClassLabel}
                </span>
                <span className="text-base font-mono font-bold text-slate-700">
                  {assessment.modelObservation?.primaryScore.toFixed(2)}
                </span>
              </div>
              <p className="text-[10px] text-slate-400">model-generated class score</p>
            </div>

            <div className="space-y-1 text-xs text-slate-600">
              <div>
                <strong>Explainability Target:</strong>{' '}
                <span className="font-mono text-slate-700">
                  {assessment.modelObservation?.targetLayer}
                </span>
              </div>
              <div>
                <strong>Top Saliency Zone:</strong>{' '}
                <span>{assessment.modelObservation?.topActivationRegion}</span>
              </div>
            </div>
          </div>

          <div className="p-2.5 bg-slate-100 rounded text-[11px] text-slate-500 leading-relaxed border border-slate-200">
            <strong>Non-Diagnostic Notice:</strong> Model score reflects algorithmic association only. Not for independent diagnosis.
          </div>
        </div>

        {/* Box B: Clinician Review Response */}
        <div
          role="region"
          aria-labelledby="clinician-domain-heading"
          className="bg-teal-50/40 rounded-2xl border-2 border-teal-500 p-6 shadow-md flex flex-col justify-between space-y-4"
        >
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-teal-200 pb-3">
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-teal-700 text-white">
                <UserCheck className="w-3 h-3 mr-1" />
                [Professional Review Response]
              </span>
              <span className="text-[10px] text-teal-800 font-mono font-bold">
                SIGNED & IMMUTABLE
              </span>
            </div>

            <h3 id="clinician-domain-heading" className="text-base font-bold text-slate-900">
              Professional Review Response
            </h3>

            {review && reviewerGradeInfo ? (
              <div className="space-y-3">
                {/* Professional Review Concurrence Card */}
                <div className="p-3.5 bg-white rounded-xl border border-teal-300 shadow-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-teal-800 uppercase font-bold">
                      Professional Evaluation
                    </span>
                    <span
                      className={`text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded ${
                        review.agreement === 'agree'
                          ? 'bg-emerald-100 text-emerald-800'
                          : review.agreement === 'disagree'
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-blue-100 text-blue-800'
                      }`}
                    >
                      {review.agreement.toUpperCase()}
                    </span>
                  </div>
                  <div className="font-bold text-slate-900 text-sm">
                    {review.agreement === 'agree' && 'Concurred with Model Observation'}
                    {review.agreement === 'disagree' && 'Clinician Disagreed with Automated Finding'}
                    {review.agreement === 'inconclusive' && `Indeterminate (${review.inconclusiveReason || 'Quality Ambiguity'})`}
                  </div>
                  <p className="text-[11px] text-slate-500">
                    {reviewerGradeInfo.label}
                  </p>
                </div>

                {/* Justification if provided */}
                {review.justificationNotes && (
                  <div className="p-3 bg-white rounded-xl border border-slate-200 text-xs text-slate-700 space-y-1">
                    <strong className="text-slate-900 block text-[11px]">
                      Clinician Observations:
                    </strong>
                    <p className="text-[11px] leading-relaxed italic">
                      "{review.justificationNotes}"
                    </p>
                  </div>
                )}

                {/* Scope Notice */}
                <div className="text-[11px] text-slate-500 p-2.5 bg-slate-50 rounded-lg border border-slate-200 leading-relaxed">
                  <strong className="text-slate-700 block text-[11px]">Clinical Scope Boundary:</strong>
                  <span>Independent professional assessment recorded. Decision-support findings do not constitute clinical diagnosis.</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">No professional review recorded.</p>
            )}
          </div>

          {/* Clinician Signature Strip */}
          {review && (
            <div className="pt-3 border-t border-teal-200 text-[11px] space-y-1 text-teal-950 font-mono">
              <div className="flex justify-between">
                <span>Reviewer: {review.clinicianName}</span>
                <span>{review.facility}</span>
              </div>
              <div className="flex justify-between text-slate-500 text-[10px]">
                <span>Recorded: {new Date(review.signedAt).toLocaleString()}</span>
                <span className="truncate max-w-[160px]">Integrity ID: {review.signatureHash?.substring(0, 16)}...</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Retinal Fundus & Grad-CAM Image Review Strip */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-teal-600" />
            Retinal Photographic & Attribution Record
          </h3>
          <span className="text-xs text-slate-400 font-mono">Interactive Viewer</span>
        </div>

        <FundusViewer
          imageUrl={assessment.imageUrl}
          gradcamUrl={assessment.gradcamUrl}
          laterality={assessment.laterality}
          nativeResolution={assessment.qualityMetrics.nativeResolution}
          grade={review?.reviewerAssessedGrade ?? assessment.modelObservation?.primaryClassGrade ?? 2}
        />
      </div>

      {/* Action Footer Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <button
          type="button"
          onClick={() => setShowAuditDrawer(true)}
          className="px-4 py-2.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 rounded-xl transition shadow-xs flex items-center gap-1.5"
        >
          <Clock className="w-4 h-4 text-slate-500" />
          <span>Inspect Immutable Audit Trail</span>
        </button>

        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={handlePrint}
            className="px-4 py-2.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 rounded-xl transition shadow-xs flex items-center gap-1.5"
          >
            <Printer className="w-4 h-4 text-slate-500" />
            <span>Print Record</span>
          </button>

          <button
            type="button"
            onClick={handleDownload}
            disabled={isDownloading}
            className="px-6 py-2.5 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-xl transition shadow-sm flex items-center gap-2 focus-visible:ring-2 focus-visible:ring-clinical-primary"
          >
            <Download className="w-4 h-4" />
            <span>Download Tamper-Evident Report</span>
          </button>
        </div>
      </div>

      {/* Audit Drawer Modal */}
      {showAuditDrawer && (
        <AuditDrawer assessment={assessment} onClose={() => setShowAuditDrawer(false)} />
      )}
    </div>
  );
};
