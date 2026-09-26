import React from 'react';
import { Sparkles, AlertCircle, Cpu } from 'lucide-react';
import { ModelObservation, ICDR_GRADES } from '../types/clinical';

interface ScoreDistributionCardProps {
  observation: ModelObservation;
}

export const ScoreDistributionCard: React.FC<ScoreDistributionCardProps> = ({ observation }) => {
  const primaryGradeInfo = ICDR_GRADES[observation.primaryClassGrade];

  // Helper color map for the 5 grades
  const getGradeBarColor = (grade: number, isPrimary: boolean) => {
    switch (grade) {
      case 0:
        return isPrimary ? 'bg-emerald-500' : 'bg-emerald-400';
      case 1:
        return isPrimary ? 'bg-cyan-500' : 'bg-cyan-400';
      case 2:
        return isPrimary ? 'bg-amber-500' : 'bg-amber-400';
      case 3:
        return isPrimary ? 'bg-orange-500' : 'bg-orange-400';
      case 4:
        return isPrimary ? 'bg-rose-500' : 'bg-rose-400';
      default:
        return 'bg-slate-400';
    }
  };

  return (
    <section
      aria-labelledby="model-observation-heading"
      className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4"
    >
      {/* Header and Primary Metric */}
      <div className="flex items-start justify-between">
        <div>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">
            <Cpu className="w-3 h-3 mr-1 text-slate-500" />
            AI Assistive Inference Engine
          </span>
          <h3 id="model-observation-heading" className="text-base font-bold text-slate-900 mt-1">
            Preliminary Model-Generated Observation
          </h3>
        </div>
        <div className="text-right">
          <span className="text-2xl font-black text-slate-900 font-mono tracking-tight">
            {observation.primaryScore.toFixed(2)}
          </span>
          <p className="text-[10px] uppercase font-semibold text-slate-500">
            Model-Generated Class Score
          </p>
        </div>
      </div>

      {/* Primary Candidate Class Banner */}
      <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div
            className="w-3.5 h-3.5 rounded-full shadow-sm"
            style={{ backgroundColor: primaryGradeInfo.color }}
            aria-hidden="true"
          />
          <div>
            <div className="font-bold text-sm text-slate-900">
              {primaryGradeInfo.label}
            </div>
            <div className="text-[11px] text-slate-500">
              {primaryGradeInfo.technicalTerm}
            </div>
          </div>
        </div>
        <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700">
          Highest Relative Score
        </span>
      </div>

      {/* 5-Class Relative Score Distribution Breakdown */}
      <div
        className="space-y-2.5 pt-2 border-t border-slate-100"
        role="region"
        aria-label="Full 5-Class Score Distribution"
      >
        <div className="flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider">
          <span>Full 5-Class Score Breakdown</span>
          <span className="font-mono text-[11px] text-slate-400">[0.00 – 1.00]</span>
        </div>

        {observation.classScores.map((item) => {
          const isPrimary = item.grade === observation.primaryClassGrade;
          const percentage = Math.round(item.score * 100);

          return (
            <div key={item.grade} className="space-y-1">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span className={isPrimary ? 'font-bold text-slate-900' : 'text-slate-600'}>
                  {item.label}
                </span>
                <span className={`font-mono ${isPrimary ? 'font-bold text-slate-900' : 'text-slate-500'}`}>
                  {item.score.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden" aria-hidden="true">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getGradeBarColor(
                    item.grade,
                    isPrimary
                  )}`}
                  style={{ width: `${Math.max(2, percentage)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Explainability Insights */}
      <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs space-y-1.5">
        <div className="flex items-center text-teal-800 font-semibold gap-1">
          <Sparkles className="w-3.5 h-3.5 text-teal-600" />
          <span>Explainability Attribution Insight</span>
        </div>
        <div className="text-slate-600 text-[11px] leading-relaxed">
          <strong className="text-slate-800">Target Bottleneck:</strong>{' '}
          <span className="font-mono text-slate-700">{observation.targetLayer}</span>
        </div>
        <div className="text-slate-600 text-[11px] leading-relaxed">
          <strong className="text-slate-800">Peak Activation Region:</strong>{' '}
          <span>{observation.topActivationRegion}</span>
        </div>
      </div>

      {/* Regulatory Boundary Footer */}
      <div className="p-2.5 bg-amber-50/60 border border-amber-200 rounded text-[11px] text-amber-900 leading-relaxed flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
        <div>
          <strong>Boundary Notice:</strong> Scores reflect preliminary feature activations derived from the pre-trained EfficientNet-B0 model. Scores do not represent clinical probability, disease certainty, or confirmed diagnosis.
        </div>
      </div>
    </section>
  );
};
