import React, { useState, useEffect, useRef } from 'react';
import {
  Search,
  Download,
  Eye,
  Clock,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  RotateCcw
} from 'lucide-react';
import {
  AssessmentRecord,
  WorklistFilter,
  ICDR_GRADES
} from '../types/clinical';
import { clinicalApi } from '../services/api';
import { AuditDrawer } from '../components/AuditDrawer';

interface RecordHistoryScreenProps {
  onSelectAssessment: (assessment: AssessmentRecord) => void;
}

export const RecordHistoryScreen: React.FC<RecordHistoryScreenProps> = ({
  onSelectAssessment,
}) => {
  const [records, setRecords] = useState<AssessmentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedAuditRecord, setSelectedAuditRecord] = useState<AssessmentRecord | null>(null);

  // Filters State
  const [filters, setFilters] = useState<WorklistFilter>({
    searchQuery: '',
    status: 'all',
    laterality: 'all',
    grade: 'all',
    agreement: 'all',
  });

  // Pagination State
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 5;

  const searchInputRef = useRef<HTMLInputElement>(null);

  const fetchRecords = async () => {
    setLoading(true);
    try {
      const data = await clinicalApi.searchRecords(filters);
      setRecords(Array.isArray(data) ? data : []);
    } catch {
      setRecords([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [filters]);

  // Keyboard shortcut '/' focuses search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && (e.target as HTMLElement)?.tagName !== 'INPUT') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleResetFilters = () => {
    setFilters({
      searchQuery: '',
      status: 'all',
      laterality: 'all',
      grade: 'all',
      agreement: 'all',
    });
    setCurrentPage(1);
  };

  // Export CSV for clinical audit committees
  const handleExportCsv = () => {
    const headers = [
      'Assessment ID',
      'Patient ID',
      'Laterality',
      'Acquisition Date',
      'Validation Status',
      'Laplacian Variance',
      'Model Candidate Stage',
      'Model Score',
      'Certified Clinical Stage',
      'Agreement',
      'Reviewing Clinician',
    ];

    const rows = records.map((r) => [
      r.id,
      r.patientId,
      r.laterality,
      r.acquisitionDate,
      r.status,
      r.qualityMetrics.laplacianVariance.toFixed(1),
      r.modelObservation?.primaryClassLabel || 'N/A',
      r.modelObservation?.primaryScore.toFixed(2) || 'N/A',
      r.clinicianReview?.certifiedGradeLabel || 'N/A',
      r.clinicianReview?.agreement || 'N/A',
      r.clinicianReview?.clinicianName || 'N/A',
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.map((val) => `"${val}"`).join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `DR-CDSS-Clinical-Audit-${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(records.length / pageSize));
  const paginatedRecords = records.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-black text-slate-900 leading-tight">
            Record History, Search & Clinical Audit Ledger
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Search patient records, review certified clinical grades, and inspect immutable audit logs for research and clinical governance.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={handleExportCsv}
            className="px-3.5 py-2 text-xs font-bold text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 rounded-lg shadow-xs transition flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-clinical-primary"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export CSV Audit Log</span>
          </button>
        </div>
      </div>

      {/* Faceted Filter Toolbar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {/* Search Query */}
          <div className="lg:col-span-2 relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
              <Search className="h-4 w-4" />
            </div>
            <input
              ref={searchInputRef}
              type="text"
              value={filters.searchQuery}
              onChange={(e) => {
                setFilters({ ...filters, searchQuery: e.target.value });
                setCurrentPage(1);
              }}
              placeholder="Search patient / assessment ID ('/' to focus)..."
              className="block w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg bg-white shadow-xs focus:ring-2 focus:ring-clinical-primary"
            />
          </div>

          {/* Status Filter */}
          <div>
            <select
              value={filters.status}
              onChange={(e) => {
                setFilters({ ...filters, status: e.target.value as any });
                setCurrentPage(1);
              }}
              className="w-full text-xs p-2 border border-slate-300 rounded-lg bg-white text-slate-700"
              aria-label="Filter by encounter status"
            >
              <option value="all">Status: All</option>
              <option value="completed">Completed</option>
              <option value="needs_review">Needs Review</option>
              <option value="rejected">Rejected</option>
            </select>
          </div>

          {/* Eye Laterality */}
          <div>
            <select
              value={filters.laterality}
              onChange={(e) => {
                setFilters({ ...filters, laterality: e.target.value as any });
                setCurrentPage(1);
              }}
              className="w-full text-xs p-2 border border-slate-300 rounded-lg bg-white text-slate-700"
              aria-label="Filter by laterality"
            >
              <option value="all">Eye: All (OD / OS)</option>
              <option value="OD">OD (Right Eye)</option>
              <option value="OS">OS (Left Eye)</option>
            </select>
          </div>

          {/* ICDR Severity Grade */}
          <div>
            <select
              value={filters.grade}
              onChange={(e) => {
                const val = e.target.value === 'all' ? 'all' : Number(e.target.value);
                setFilters({ ...filters, grade: val as any });
                setCurrentPage(1);
              }}
              className="w-full text-xs p-2 border border-slate-300 rounded-lg bg-white text-slate-700"
              aria-label="Filter by ICDR grade"
            >
              <option value="all">Severity: All Grades</option>
              {Object.values(ICDR_GRADES).map((item) => (
                <option key={item.grade} value={item.grade}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          {/* Agreement Filter */}
          <div>
            <select
              value={filters.agreement}
              onChange={(e) => {
                setFilters({ ...filters, agreement: e.target.value as any });
                setCurrentPage(1);
              }}
              className="w-full text-xs p-2 border border-slate-300 rounded-lg bg-white text-slate-700"
              aria-label="Filter by clinician agreement"
            >
              <option value="all">Agreement: All</option>
              <option value="agree">Clinician Agreed</option>
              <option value="disagree">Clinician Overrode</option>
              <option value="inconclusive">Inconclusive</option>
            </select>
          </div>
        </div>

        <div className="flex justify-between items-center text-[11px] text-slate-500 pt-1">
          <span>Found {records.length} matching assessment records</span>
          <button
            type="button"
            onClick={handleResetFilters}
            className="text-clinical-primary hover:underline flex items-center gap-1 font-medium"
          >
            <RotateCcw className="w-3 h-3" />
            Reset Filters
          </button>
        </div>
      </div>

      {/* Historical Records Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th scope="col" className="px-4 py-3">Patient / ID</th>
                <th scope="col" className="px-4 py-3">Eye</th>
                <th scope="col" className="px-4 py-3">Acquired</th>
                <th scope="col" className="px-4 py-3">Model Candidate Score</th>
                <th scope="col" className="px-4 py-3">Professional Review Response</th>
                <th scope="col" className="px-4 py-3">Agreement</th>
                <th scope="col" className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-500">
                    <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-clinical-primary"></div>
                    <p className="mt-2 text-xs font-medium">Filtering records...</p>
                  </td>
                </tr>
              ) : paginatedRecords.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-500">
                    No historical assessments matching selected criteria.
                  </td>
                </tr>
              ) : (
                paginatedRecords.map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-50 transition-colors">
                    {/* Patient Reference */}
                    <td className="px-4 py-3 whitespace-nowrap font-medium text-slate-900">
                      <div className="font-bold">{rec.patientId}</div>
                      <span className="text-[10px] text-slate-400 font-mono">{rec.id}</span>
                    </td>

                    {/* Eye Laterality */}
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

                    {/* Date */}
                    <td className="px-4 py-3 whitespace-nowrap text-slate-600 font-mono text-[11px]">
                      <div>{new Date(rec.acquisitionDate).toLocaleDateString()}</div>
                      <span className="text-slate-400 text-[10px]">
                        {new Date(rec.acquisitionDate).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                    </td>

                    {/* Model Candidate */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      {rec.modelObservation ? (
                        <div>
                          <span className="font-semibold text-slate-800">
                            {rec.modelObservation.primaryClassLabel}
                          </span>
                          <span className="font-mono text-slate-500 text-[11px] block">
                            Score: {rec.modelObservation.primaryScore.toFixed(2)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400 italic">Aborted</span>
                      )}
                    </td>

                    {/* Professional Review Response */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      {rec.clinicianReview ? (
                        <div>
                          <span className="font-bold text-teal-900">
                            {rec.clinicianReview.certifiedGradeLabel}
                          </span>
                          <span className="text-[10px] text-slate-400 block">
                            {rec.clinicianReview.clinicianName.split(',')[0]}
                          </span>
                        </div>
                      ) : (
                        <span className="text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200 text-[11px] font-medium">
                          Pending Human Review
                        </span>
                      )}
                    </td>

                    {/* Agreement Pill */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      {rec.clinicianReview ? (
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold uppercase ${
                            rec.clinicianReview.agreement === 'agree'
                              ? 'bg-emerald-100 text-emerald-800'
                              : rec.clinicianReview.agreement === 'disagree'
                              ? 'bg-amber-100 text-amber-800'
                              : 'bg-blue-100 text-blue-800'
                          }`}
                        >
                          {rec.clinicianReview.agreement}
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[11px]">—</span>
                      )}
                    </td>

                    {/* Action Buttons */}
                    <td className="px-4 py-3 whitespace-nowrap text-right space-x-1">
                      <button
                        type="button"
                        onClick={() => setSelectedAuditRecord(rec)}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-md transition"
                        title="View Immutable Audit Ledger"
                      >
                        <Clock className="w-3 h-3 inline mr-1 text-slate-500" />
                        Audit
                      </button>

                      <button
                        type="button"
                        onClick={() => onSelectAssessment(rec)}
                        className="px-3 py-1 text-[11px] font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-md transition"
                      >
                        Open
                        <ArrowRight className="w-3 h-3 inline ml-1" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Toolbar */}
        <div className="px-4 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-600">
          <div>
            Showing{' '}
            <span className="font-bold text-slate-900">
              {records.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}
            </span>{' '}
            to{' '}
            <span className="font-bold text-slate-900">
              {Math.min(currentPage * pageSize, records.length)}
            </span>{' '}
            of <span className="font-bold text-slate-900">{records.length}</span> records
          </div>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              className="p-1 rounded border border-slate-300 hover:bg-white disabled:opacity-40 transition"
              aria-label="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="font-mono text-[11px]">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              className="p-1 rounded border border-slate-300 hover:bg-white disabled:opacity-40 transition"
              aria-label="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Audit Drawer Modal */}
      {selectedAuditRecord && (
        <AuditDrawer
          assessment={selectedAuditRecord}
          onClose={() => setSelectedAuditRecord(null)}
        />
      )}
    </div>
  );
};
