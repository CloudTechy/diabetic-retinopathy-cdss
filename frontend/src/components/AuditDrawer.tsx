import React from 'react';
import { X, ShieldCheck, Clock, FileText, CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';
import { AssessmentRecord } from '../types/clinical';

interface AuditDrawerProps {
  assessment: AssessmentRecord | null;
  onClose: () => void;
}

export const AuditDrawer: React.FC<AuditDrawerProps> = ({ assessment, onClose }) => {
  if (!assessment) return null;

  const getBadgeIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-amber-500" />;
      case 'error':
        return <AlertOctagon className="w-4 h-4 text-rose-500" />;
      default:
        return <Clock className="w-4 h-4 text-blue-500" />;
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="audit-drawer-title"
      className="fixed inset-0 z-50 overflow-hidden bg-black/50 backdrop-blur-xs flex justify-end"
    >
      <div className="w-full max-w-lg bg-white h-full shadow-2xl flex flex-col animate-in slide-in-from-right duration-200">
        {/* Header */}
        <header className="p-4 sm:p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-teal-50 text-teal-700 rounded-lg border border-teal-200">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 id="audit-drawer-title" className="text-base font-bold text-slate-900 leading-tight">
                Audit trail (application-level append-only)
              </h2>
              <p className="text-xs text-slate-500 font-mono">
                {assessment.id} • Patient {assessment.patientId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition focus-visible:ring-2 focus-visible:ring-teal-500"
            aria-label="Close audit ledger drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        {/* Image hash card */}
        <div className="p-4 bg-slate-900 text-slate-200 text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-teal-300 uppercase tracking-wider text-[10px]">
              SHA-256 Retinal Image Hash
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Hash-anchored</span>
          </div>
          <p className="font-mono text-[11px] break-all bg-slate-800 p-2 rounded border border-slate-700 text-teal-200">
            {assessment.qualityMetrics.sha256Hash}
          </p>
          <div className="flex justify-between text-[11px] text-slate-400 pt-1">
            <span>Laterality: {assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}</span>
            <span>Resolution: {assessment.qualityMetrics.nativeResolution}</span>
          </div>
        </div>

        {/* Audit Event Timeline */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-6">
          <div className="space-y-4">
            {assessment.auditTrail.map((event, idx) => (
              <div key={event.id || idx} className="relative flex items-start space-x-3 text-xs">
                {/* Timeline connector */}
                {idx !== assessment.auditTrail.length - 1 && (
                  <div
                    className="absolute left-2.5 top-6 bottom-0 w-0.5 bg-slate-200"
                    aria-hidden="true"
                  />
                )}

                <div className="mt-0.5 flex-shrink-0 z-10 bg-white p-0.5 rounded-full">
                  {getBadgeIcon(event.badgeType)}
                </div>

                <div className="flex-1 bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-900">{event.action}</span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {new Date(event.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </span>
                  </div>
                  <p className="text-slate-600 text-[11px]">{event.details}</p>
                  <div className="text-[10px] text-slate-400 font-medium pt-1">
                    Actor: <span className="text-slate-700">{event.actor}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <footer className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
          <span className="flex items-center gap-1 font-mono text-[11px]">
            <FileText className="w-3.5 h-3.5 text-slate-400" />
            Audit Ledger Spec v1.0
          </span>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 font-medium rounded-lg transition"
          >
            Close
          </button>
        </footer>
      </div>
    </div>
  );
};
