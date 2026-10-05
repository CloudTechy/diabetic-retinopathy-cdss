import React, { useState, useEffect, useRef } from 'react';
import {
  PlusCircle,
  Search,
  Filter,
  Eye,
  FileCheck,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Activity
} from 'lucide-react';
import { AssessmentRecord, ReviewStatus, EyeLaterality } from '../types/clinical';
import { clinicalApi } from '../services/api';
import { describeApiError } from '../utils/apiError';

interface DashboardScreenProps {
  onSelectAssessment: (assessment: AssessmentRecord) => void;
  onNewAssessment: () => void;
  onNavigateHistory: () => void;
}

export const DashboardScreen: React.FC<DashboardScreenProps> = ({
  onSelectAssessment,
  onNewAssessment,
  onNavigateHistory,
}) => {
  const [records, setRecords] = useState<AssessmentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  // A failed request is shown as a failure. An earlier revision emptied the
  // list instead, so a server error read as "no records".
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<'all' | ReviewStatus>('all');
  const [lateralityFilter, setLateralityFilter] = useState<'all' | EyeLaterality>('all');

  const searchInputRef = useRef<HTMLInputElement>(null);

  const fetchRecords = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const data = await clinicalApi.getWorklist();
      setRecords(Array.isArray(data) ? data : []);
    } catch (err) {
      setRecords([]);
      setLoadError(describeApiError(err, 'The records could not be loaded'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  // Keyboard shortcut listener: Alt+N -> New Assessment, / -> Focus search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && (e.key === 'n' || e.key === 'N')) {
        e.preventDefault();
        onNewAssessment();
      } else if (e.key === '/' && (e.target as HTMLElement)?.tagName !== 'INPUT') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNewAssessment]);

  // Filter records
  const filteredRecords = records.filter((rec) => {
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchP = rec.patientId.toLowerCase().includes(q);
      const matchId = rec.id.toLowerCase().includes(q);
      if (!matchP && !matchId) return false;
    }
    if (statusFilter !== 'all' && rec.status !== statusFilter) return false;
    if (lateralityFilter !== 'all' && rec.laterality !== lateralityFilter) return false;
    return true;
  });

  // Calculate Metrics
  const totalRecords = records.length;   // every record the API returned, not today's
  const pendingReview = records.filter((r) => r.status === 'needs_review').length;
  const rejectedCount = records.filter((r) => r.status === 'rejected').length;
  const highPriority = records.filter((r) => {
    const grade = r.modelObservation?.primaryClassGrade;
    return (grade === 3 || grade === 4) && r.status === 'needs_review';
  }).length;

  // The gate that actually failed, read from the record. The table used to
  // print "Gate 3 Failed (Blur)" for every rejected record, whichever gate
  // had rejected it.
  const describeGates = (rec: AssessmentRecord): { failed: boolean; text: string } => {
    const gates = rec.validationGates || [];
    const failedGate = gates.find((g) => g.status === 'failed');
    if (failedGate) return { failed: true, text: `Gate ${failedGate.gateIndex} failed` };
    if (rec.status === 'rejected') return { failed: true, text: 'Rejected at validation' };
    if (gates.length > 0 && gates.every((g) => g.status === 'passed')) return { failed: false, text: `All ${gates.length} passed` };
    return { failed: false, text: 'Validation not complete' };
  };

  const renderStatusBadge = (status: ReviewStatus) => {
    switch (status) {
      case 'validating':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 animate-pulse border border-blue-200">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-600 mr-1.5"></span>
            Validating...
          </span>
        );
      case 'needs_review':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-900 border border-amber-300">
            <Clock className="w-3 h-3 mr-1 text-amber-700" />
            Needs Review
          </span>
        );
      case 'completed':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-teal-100 text-teal-900 border border-teal-300">
            <FileCheck className="w-3 h-3 mr-1 text-teal-700" />
            Completed
          </span>
        );
      case 'rejected':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-200 text-slate-700 border border-slate-300">
            <XCircle className="w-3 h-3 mr-1 text-slate-500" />
            Rejected
          </span>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Clinician Authority Persistent Banner */}
      <div className="bg-amber-50/80 border-l-4 border-amber-500 p-3.5 rounded-r-xl shadow-xs flex items-center justify-between text-xs text-amber-900">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
          <span>
            <strong>Clinical Safety Protocol:</strong> Model outputs are non-binding preliminary observations requiring independent clinical interpretation.
          </span>
        </div>
        <span className="font-mono text-[11px] text-amber-800 bg-amber-100 px-2 py-0.5 rounded hidden lg:inline">
          Fail-Closed Invariant Active
        </span>
      </div>

      {/* Top Metrics Ribbon */}
      <section aria-label="Assessment Worklist Summary Metrics" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-slate-500">Total records</p>
            <p className="text-2xl font-black text-slate-900 mt-1">{totalRecords}</p>
            <p className="text-[11px] text-slate-400">Retinal Encounters</p>
          </div>
          <div className="p-3 bg-slate-100 text-slate-700 rounded-xl">
            <Activity className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 2 */}
        <div className="bg-white p-4 rounded-xl border border-amber-200 shadow-xs flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-amber-700">Pending Review</p>
            <p className="text-2xl font-black text-amber-900 mt-1">{pendingReview}</p>
            <p className="text-[11px] text-amber-700">Awaiting Clinician Action</p>
          </div>
          <div className="p-3 bg-amber-50 text-amber-600 rounded-xl">
            <Clock className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 3 */}
        <div className="bg-white p-4 rounded-xl border border-rose-200 shadow-xs flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-rose-700">Marked for Attention</p>
            <p className="text-2xl font-black text-rose-700 mt-1">{highPriority}</p>
            <p className="text-[11px] text-rose-600">Severe NPDR or PDR (Grade 3/4)</p>
          </div>
          <div className="p-3 bg-rose-50 text-rose-600 rounded-xl">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 4 */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-slate-500">Technical Rejections</p>
            <p className="text-2xl font-black text-slate-700 mt-1">{rejectedCount}</p>
            <p className="text-[11px] text-slate-400">Failed Gates 1–3</p>
          </div>
          <div className="p-3 bg-slate-100 text-slate-600 rounded-xl">
            <XCircle className="w-5 h-5" />
          </div>
        </div>
      </section>

      {/* Quick Action Bar & Worklist Filter Controls */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Left: Search input */}
        <div className="relative flex-1 max-w-md">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
            <Search className="h-4 w-4" />
          </div>
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search patient ID or assessment ID (Press '/' to focus)..."
            className="block w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg bg-white shadow-xs focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary"
          />
        </div>

        {/* Right: Filters & Action Buttons */}
        <div className="flex items-center space-x-2 flex-wrap sm:flex-nowrap gap-y-2">
          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as any)}
            className="text-xs border border-slate-300 rounded-lg px-2.5 py-2 bg-white text-slate-700 focus:ring-2 focus:ring-clinical-primary"
            aria-label="Filter by review status"
          >
            <option value="all">All Statuses</option>
            <option value="needs_review">Needs Review</option>
            <option value="completed">Completed</option>
            <option value="rejected">Rejected</option>
          </select>

          {/* Eye Laterality Filter */}
          <select
            value={lateralityFilter}
            onChange={(e) => setLateralityFilter(e.target.value as any)}
            className="text-xs border border-slate-300 rounded-lg px-2.5 py-2 bg-white text-slate-700 focus:ring-2 focus:ring-clinical-primary"
            aria-label="Filter by eye laterality"
          >
            <option value="all">All Eyes (OD/OS)</option>
            <option value="OD">OD (Right Eye)</option>
            <option value="OS">OS (Left Eye)</option>
          </select>

          {/* Search History Navigation */}
          <button
            onClick={onNavigateHistory}
            className="px-3 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition shadow-xs flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-clinical-primary"
          >
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <span>Audit History</span>
          </button>

          {/* Primary Action Button: New Assessment (Alt+N) */}
          <button
            onClick={onNewAssessment}
            className="px-4 py-2 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg transition shadow-sm flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-clinical-primary"
            title="Upload New Retinal Fundus (Alt+N)"
          >
            <PlusCircle className="w-4 h-4" />
            <span>New Assessment (Alt+N)</span>
          </button>
        </div>
      </div>

      {/* Interactive Worklist Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        {/* Phone: one card per record. The seven-column table below needs a
            wide screen and only scrolled sideways on a phone. */}
        <ul className="md:hidden divide-y divide-slate-200" aria-label="Assessment worklist">
          {loading ? (
            <li className="px-4 py-10 text-center text-slate-500 text-xs font-medium">Loading clinical worklist queue...</li>
          ) : loadError ? (
            <li role="alert" className="px-4 py-10 text-center text-rose-700 text-xs font-semibold">{loadError}</li>
          ) : filteredRecords.length === 0 ? (
            <li className="px-4 py-10 text-center text-slate-500 text-xs">No assessment records found matching current query.</li>
          ) : (
            filteredRecords.map((rec) => {
              const gates = describeGates(rec);
              return (
                <li key={rec.id} className="p-4 space-y-2 text-xs">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="font-bold text-slate-900 break-all">{rec.patientId}</p>
                      <p className="text-[10px] text-slate-400 font-mono">{rec.id}</p>
                    </div>
                    <div className="flex-shrink-0">{renderStatusBadge(rec.status)}</div>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-600">
                    <span className="font-mono font-bold">{rec.laterality}</span>
                    <span className="font-mono">
                      {new Date(rec.acquisitionDate).toLocaleDateString()}{' '}
                      {new Date(rec.acquisitionDate).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className={gates.failed ? 'text-rose-700 font-semibold' : ''}>{gates.text}</span>
                  </div>
                  <p className="text-slate-800">
                    {rec.modelObservation ? (
                      <>
                        <span className="font-semibold">{rec.modelObservation.primaryClassLabel}</span>{' '}
                        <span className="font-mono text-slate-600">({rec.modelObservation.primaryScore.toFixed(2)})</span>{' '}
                        <span className="text-[10px] text-slate-400">model-generated class score</span>
                      </>
                    ) : (
                      <span className="text-slate-400 italic">Inference aborted</span>
                    )}
                  </p>
                  <button
                    onClick={() => onSelectAssessment(rec)}
                    className={`w-full inline-flex items-center justify-center px-3 py-2 rounded-lg text-xs font-bold ${
                      rec.status === 'needs_review'
                        ? 'bg-clinical-primary text-white'
                        : 'bg-slate-100 text-slate-700'
                    }`}
                  >
                    {rec.status === 'needs_review' ? 'Review & Grade' : 'View Record'}
                    <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
                  </button>
                </li>
              );
            })
          )}
        </ul>

        <div className="hidden md:block overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-xs text-left" role="table">
            <thead className="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th scope="col" className="px-4 py-3">Patient / Study ID</th>
                <th scope="col" className="px-4 py-3">Record created</th>
                <th scope="col" className="px-4 py-3">Eye</th>
                <th scope="col" className="px-4 py-3">Quality Gates</th>
                <th scope="col" className="px-4 py-3">Preliminary Model Score</th>
                <th scope="col" className="px-4 py-3">Status</th>
                <th scope="col" className="px-4 py-3 text-right">Clinical Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-500">
                    <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-clinical-primary"></div>
                    <p className="mt-2 text-xs font-medium">Loading clinical worklist queue...</p>
                  </td>
                </tr>
              ) : loadError ? (
                <tr>
                  <td colSpan={7} role="alert" className="px-4 py-12 text-center text-rose-700 text-xs font-semibold">
                    {loadError}
                  </td>
                </tr>
              ) : filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-500">
                    No assessment records found matching current query.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((rec) => {
                  const isSevere =
                    rec.modelObservation &&
                    (rec.modelObservation.primaryClassGrade === 3 ||
                      rec.modelObservation.primaryClassGrade === 4);

                  return (
                    <tr
                      key={rec.id}
                      className="hover:bg-slate-50/80 transition-colors focus-within:bg-slate-50"
                    >
                      {/* Patient & Assessment Reference */}
                      <td className="px-4 py-3 whitespace-nowrap font-medium text-slate-900">
                        <div className="font-bold flex items-center gap-1.5">
                          <span>{rec.patientId}</span>
                          {isSevere && (
                            <span className="w-2 h-2 rounded-full bg-rose-500" title="Attention: Higher Stage Observation (Grade 3/4)" />
                          )}
                        </div>
                        <span className="text-[10px] text-slate-400 font-mono">{rec.id}</span>
                      </td>

                      {/* record creation time */}
                      <td className="px-4 py-3 whitespace-nowrap text-slate-600 font-mono text-[11px]">
                        <div>{new Date(rec.acquisitionDate).toLocaleDateString()}</div>
                        <span className="text-slate-400">
                          {new Date(rec.acquisitionDate).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      </td>

                      {/* Eye Laterality Badge */}
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold ${
                            rec.laterality === 'OD'
                              ? 'bg-teal-50 text-teal-800 border border-teal-200'
                              : 'bg-blue-50 text-blue-800 border border-blue-200'
                          }`}
                        >
                          <Eye className="w-3 h-3 mr-1" />
                          {rec.laterality}
                        </span>
                      </td>

                      {/* Quality Gates */}
                      <td className="px-4 py-3 whitespace-nowrap">
                        {describeGates(rec).failed ? (
                          <span className="text-rose-700 font-semibold text-[11px] flex items-center gap-1">
                            <XCircle className="w-3.5 h-3.5" />
                            {describeGates(rec).text}
                          </span>
                        ) : (
                          <div className="flex items-center gap-1">
                            <span className={`w-2 h-2 rounded-full ${describeGates(rec).text.startsWith('All') ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                            <span className="text-slate-700 text-[11px]">{describeGates(rec).text}</span>
                          </div>
                        )}
                        <span className="text-[10px] text-slate-400 font-mono block">
                          Laplacian: {rec.qualityMetrics.laplacianVariance.toFixed(1)}
                        </span>
                      </td>

                      {/* Preliminary Model Class Score (strictly non-diagnostic terminology) */}
                      <td className="px-4 py-3 whitespace-nowrap">
                        {rec.modelObservation ? (
                          <div>
                            <div className="font-semibold text-slate-900 flex items-center gap-1.5">
                              <span>{rec.modelObservation.primaryClassLabel}</span>
                              <span className="font-mono text-xs text-slate-600">
                                ({rec.modelObservation.primaryScore.toFixed(2)})
                              </span>
                            </div>
                            <span className="text-[10px] text-slate-400">
                              model-generated class score
                            </span>
                          </div>
                        ) : (
                          <span className="text-slate-400 italic text-[11px]">
                            Inference aborted
                          </span>
                        )}
                      </td>

                      {/* Review Status Pill */}
                      <td className="px-4 py-3 whitespace-nowrap">
                        {renderStatusBadge(rec.status)}
                      </td>

                      {/* Clinical Action Button */}
                      <td className="px-4 py-3 whitespace-nowrap text-right">
                        <button
                          onClick={() => onSelectAssessment(rec)}
                          className={`inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-bold transition shadow-xs focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                            rec.status === 'needs_review'
                              ? 'bg-clinical-primary hover:bg-clinical-primary-hover text-white'
                              : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                          }`}
                        >
                          {rec.status === 'needs_review' ? 'Review & Grade' : 'View Record'}
                          <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
