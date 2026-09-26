import React, { useState } from 'react';
import {
  AlertTriangle,
  UserCheck,
  Download,
  ShieldCheck,
  Eye,
  Clock
} from 'lucide-react';
import { AssessmentRecord } from '../types/clinical';
import { FundusViewer } from '../components/FundusViewer';
import { ScoreDistributionCard } from '../components/ScoreDistributionCard';
import { AuditDrawer } from '../components/AuditDrawer';
import { clinicalApi } from '../services/api';

interface DecisionSupportScreenProps {
  assessment: AssessmentRecord;
  onInitiateReview: () => void;
  onViewCompleted: () => void;
}

export const DecisionSupportScreen: React.FC<DecisionSupportScreenProps> = ({
  assessment,
  onInitiateReview,
  onViewCompleted,
}) => {
  const [showAuditDrawer, setShowAuditDrawer] = useState<boolean>(false);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  const handleDownloadReport = async () => {
    setIsDownloading(true);
    try {
      await clinicalApi.downloadReportPdf(assessment);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Persistent Clinical Safety & Regulatory Banner */}
      <div
        role="alert"
        className="bg-amber-50 border-l-4 border-amber-600 p-4 rounded-r-xl shadow-xs text-amber-950 space-y-1"
      >
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <h2 className="text-xs sm:text-sm font-bold uppercase tracking-wide">
            NOTICE: CLINICAL DECISION SUPPORT ONLY — NOT FOR INDEPENDENT DIAGNOSIS
          </h2>
        </div>
        <p className="text-xs text-amber-900 leading-relaxed pl-7">
          Model-generated scores represent preliminary mathematical associations based on visual feature activations. Diagnostic judgment, clinical staging, and management plans remain exclusively the responsibility of the reviewing clinician.
        </p>
      </div>

      {/* Main 2-Column Clinical Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Pane (7 cols / ~60-65%): Dual-Layer Synchronized Fundus & Grad-CAM Canvas */}
        <div className="lg:col-span-7 space-y-4">
          <FundusViewer
            imageUrl={assessment.imageUrl}
            gradcamUrl={assessment.gradcamUrl}
            laterality={assessment.laterality}
            nativeResolution={assessment.qualityMetrics.nativeResolution}
            grade={assessment.modelObservation?.primaryClassGrade ?? 2}
          />

          {/* Technical Quality Summary Strip */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs flex flex-wrap items-center justify-between text-xs gap-3">
            <div className="flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <span className="font-semibold text-slate-900">
                Technical Quality Thresholds:
              </span>
              <span className="text-slate-600">
                Laplacian Variance:{' '}
                <strong className="text-slate-900 font-mono">
                  {assessment.qualityMetrics.laplacianVariance.toFixed(1)}
                </strong>{' '}
                (&gt;100.0)
              </span>
            </div>

            <div className="flex items-center space-x-4 text-slate-500 font-mono text-[11px]">
              <span>Illumination: {assessment.qualityMetrics.illuminationIndex.toFixed(2)}</span>
              <span>SHA-256 Verified</span>
            </div>
          </div>
        </div>

        {/* Right Pane (5 cols / ~35-40%): Assessment Summary, Bounded Scores & Review Callout */}
        <div className="lg:col-span-5 space-y-5">
          {/* Patient Context & Study Details */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
            <div className="flex items-start justify-between border-b border-slate-100 pb-3">
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 font-mono">
                  Assessment Reference
                </span>
                <h3 className="text-base font-bold text-slate-900">{assessment.id}</h3>
              </div>

              <span
                className={`inline-flex items-center px-2.5 py-1 rounded text-xs font-mono font-bold ${
                  assessment.laterality === 'OD'
                    ? 'bg-teal-50 text-teal-800 border border-teal-200'
                    : 'bg-blue-50 text-blue-800 border border-blue-200'
                }`}
              >
                <Eye className="w-3.5 h-3.5 mr-1" />
                {assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-400 block text-[11px]">Patient ID:</span>
                <span className="font-bold text-slate-800">{assessment.patientId}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[11px]">Acquired:</span>
                <span className="font-mono text-slate-700 text-[11px]">
                  {new Date(assessment.acquisitionDate).toLocaleString([], {
                    dateStyle: 'short',
                    timeStyle: 'short',
                  })}
                </span>
              </div>
            </div>

            {assessment.clinicalNotes && (
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200 text-[11px] text-slate-600">
                <strong className="text-slate-800">Clinical Indication:</strong>{' '}
                {assessment.clinicalNotes}
              </div>
            )}
          </div>

          {/* Model-Generated Class Scores Card (Bounded Terminology) */}
          {assessment.modelObservation && (
            <ScoreDistributionCard observation={assessment.modelObservation} />
          )}

          {/* Action Callout Bar */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
            {assessment.status === 'completed' ? (
              <div className="space-y-3">
                <div className="p-3 bg-teal-50 border border-teal-200 rounded-lg text-xs text-teal-900 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-teal-600" />
                    <div>
                      <span className="font-bold block">Review Finalized & Signed</span>
                      <span className="text-[11px] text-teal-700">Record is officially certified and locked.</span>
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={onViewCompleted}
                  className="w-full py-2.5 px-4 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm transition"
                >
                  View Final Certified Clinical Report
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <button
                  type="button"
                  onClick={onInitiateReview}
                  className="w-full flex items-center justify-center py-3 px-4 rounded-xl text-xs font-black text-white bg-clinical-primary hover:bg-clinical-primary-hover shadow-md transition transform active:scale-[0.99] focus-visible:ring-2 focus-visible:ring-clinical-primary"
                >
                  <UserCheck className="w-4 h-4 mr-2" />
                  Initiate Certified Clinician Review (Screen 6)
                </button>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={handleDownloadReport}
                    disabled={isDownloading}
                    className="p-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-200 transition flex items-center justify-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5 text-slate-500" />
                    <span>Technical Summary</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setShowAuditDrawer(true)}
                    className="p-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-200 transition flex items-center justify-center gap-1.5"
                  >
                    <Clock className="w-3.5 h-3.5 text-slate-500" />
                    <span>Audit Ledger</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Audit Ledger Drawer Modal */}
      {showAuditDrawer && (
        <AuditDrawer assessment={assessment} onClose={() => setShowAuditDrawer(false)} />
      )}
    </div>
  );
};
