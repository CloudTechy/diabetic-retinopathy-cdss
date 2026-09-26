import React from 'react';
import { Check, X, Loader2, ShieldCheck, Eye, Sparkles } from 'lucide-react';
import { GateResult } from '../types/clinical';

interface ValidationStepperProps {
  gates: GateResult[];
  activeGateIndex?: number; // 1, 2, or 3
}

export const ValidationStepper: React.FC<ValidationStepperProps> = ({ gates }) => {
  const getGateIcon = (index: number) => {
    switch (index) {
      case 1:
        return <ShieldCheck className="w-4 h-4" />;
      case 2:
        return <Eye className="w-4 h-4" />;
      case 3:
        return <Sparkles className="w-4 h-4" />;
      default:
        return <span>{index}</span>;
    }
  };

  return (
    <nav aria-label="Image Validation Pipeline Progress" className="w-full py-4">
      <ol className="flex items-center justify-between w-full" role="list">
        {gates.map((gate, idx) => {
          const isLast = idx === gates.length - 1;
          const isPassed = gate.status === 'passed';
          const isFailed = gate.status === 'failed';
          const isInProgress = gate.status === 'in_progress';

          return (
            <React.Fragment key={gate.gateIndex}>
              <li
                className="flex items-center space-x-3 flex-1"
                aria-current={isInProgress ? 'step' : 'false'}
              >
                {/* Circle Icon Badge */}
                <div
                  className={`flex items-center justify-center w-9 h-9 rounded-full transition-all duration-300 font-bold text-xs flex-shrink-0 shadow-sm ${
                    isPassed
                      ? 'bg-status-pass text-white'
                      : isFailed
                      ? 'bg-status-fail text-white ring-4 ring-rose-100'
                      : isInProgress
                      ? 'border-2 border-clinical-primary bg-clinical-primary-light text-clinical-primary'
                      : 'border-2 border-slate-300 bg-white text-slate-400'
                  }`}
                  aria-hidden="true"
                >
                  {isPassed ? (
                    <Check className="w-4 h-4 stroke-[3]" />
                  ) : isFailed ? (
                    <X className="w-4 h-4 stroke-[3]" />
                  ) : isInProgress ? (
                    <Loader2 className="w-4 h-4 animate-spin text-clinical-primary" />
                  ) : (
                    getGateIcon(gate.gateIndex)
                  )}
                </div>

                {/* Gate Label & Detail */}
                <div className="min-w-0 pr-2">
                  <p
                    className={`text-[10px] font-semibold uppercase tracking-wider ${
                      isPassed
                        ? 'text-status-pass'
                        : isFailed
                        ? 'text-status-fail'
                        : isInProgress
                        ? 'text-clinical-primary'
                        : 'text-slate-400'
                    }`}
                  >
                    Gate {gate.gateIndex}
                  </p>
                  <p className="text-xs sm:text-sm font-semibold text-slate-900 truncate">
                    {gate.title}
                  </p>
                  <span className="sr-only">Status: {gate.status}</span>
                  {gate.metric && (
                    <p className="text-[11px] text-slate-500 font-mono truncate hidden sm:block">
                      {gate.metric}
                    </p>
                  )}
                </div>
              </li>

              {/* Connecting Divider */}
              {!isLast && (
                <div
                  className={`flex-1 h-0.5 mx-2 sm:mx-4 transition-colors duration-300 ${
                    isPassed ? 'bg-status-pass' : isFailed ? 'bg-status-fail' : 'bg-slate-200'
                  }`}
                  aria-hidden="true"
                />
              )}
            </React.Fragment>
          );
        })}
      </ol>
    </nav>
  );
};
