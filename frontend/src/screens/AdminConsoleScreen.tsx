import React, { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle, Check, CheckCircle2, ClipboardList, Clock, Copy, KeyRound, RefreshCw, Search, ShieldCheck, UserCheck, UserPlus, UserX, Users, X,
} from 'lucide-react';
import { clinicalApi } from '../services/api';
import { describeApiError } from '../utils/apiError';
import { AccountSummary, AdminAuditEvent, CreatedPhysician, UserAccount } from '../types/clinical';

type Tab = 'overview' | 'users' | 'create' | 'audit';

interface PendingAction {
  kind: 'approve' | 'reject' | 'suspend' | 'reactivate' | 'reset-password';
  user: UserAccount;
}

const STATUS_CLASS: Record<string, string> = {
  PENDING_APPROVAL: 'bg-amber-100 text-amber-800 border-amber-200',
  ACTIVE: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  SUSPENDED: 'bg-slate-200 text-slate-700 border-slate-300',
  REJECTED: 'bg-rose-100 text-rose-800 border-rose-200',
};

const ACTION_LABEL: Record<PendingAction['kind'], string> = {
  approve: 'Approve account',
  reject: 'Reject registration',
  suspend: 'Suspend account',
  reactivate: 'Reactivate account',
  'reset-password': 'Issue a temporary password',
};

const fmt = (iso?: string | null) => (iso ? new Date(iso).toLocaleString() : '—');

const StatusPill: React.FC<{ status: string }> = ({ status }) => (
  <span className={`inline-flex px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${STATUS_CLASS[status] || 'bg-slate-100 text-slate-700 border-slate-200'}`}>
    {status}
  </span>
);

/**
 * Super Administrator console: account states, pending registrations, the
 * account table, creation, and the administrative audit history. Every
 * action here is enforced by the backend as well; this screen only presents
 * what the /admin routes allow.
 */
export const AdminConsoleScreen: React.FC = () => {
  const [tab, setTab] = useState<Tab>('overview');
  const [summary, setSummary] = useState<AccountSummary | null>(null);
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [pending, setPending] = useState<UserAccount[]>([]);
  const [events, setEvents] = useState<AdminAuditEvent[]>([]);
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [submittingOne, setSubmittingOne] = useState(false);
  const [submittingBatch, setSubmittingBatch] = useState(false);
  const [actionRunning, setActionRunning] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);
  const [notice, setNotice] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [action, setAction] = useState<PendingAction | null>(null);
  const [actionReason, setActionReason] = useState('');
  const [issued, setIssued] = useState<CreatedPhysician[] | null>(null);
  const [auditUserId, setAuditUserId] = useState<string>('');

  const [form, setForm] = useState({ fullName: '', username: '', email: '', licenseNumber: '', facility: '', activateImmediately: true });
  const [batch, setBatch] = useState({ count: 3, usernamePrefix: 'demo.physician', facility: 'Supervised demonstration' });

  const fail = (err: unknown, what: string) => setNotice({ kind: 'error', text: describeApiError(err, what) });

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [s, u, p] = await Promise.all([
        clinicalApi.adminSummary(),
        clinicalApi.adminListUsers({ q: query || undefined, status: statusFilter || undefined, role: roleFilter || undefined }),
        clinicalApi.adminPendingUsers(),
      ]);
      setSummary(s);
      setUsers(u);
      setPending(p);
    } catch (err) {
      fail(err, 'The accounts could not be loaded');
    } finally {
      setLoading(false);
    }
  }, [query, statusFilter, roleFilter]);

  const loadAudit = useCallback(async (userId?: string) => {
    try {
      setEvents(await clinicalApi.adminAuditEvents({ userId: userId || undefined, limit: 300 }));
    } catch (err) {
      fail(err, 'The audit history could not be loaded');
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (tab === 'audit') loadAudit(auditUserId); }, [tab, auditUserId, loadAudit]);

  const copyCredential = (username: string, password?: string) => {
    if (!password) return;
    navigator.clipboard.writeText(`Username: ${username}\nTemporary Password: ${password}`);
    setCopiedId(username);
    setTimeout(() => setCopiedId(null), 2500);
  };

  const copyAllCredentials = () => {
    if (!issued || issued.length === 0) return;
    const text = issued
      .map((c) => `Account: ${c.user.fullName || c.user.username}\nUsername: ${c.user.username}\nTemporary Password: ${c.temporaryPassword}\nLogin URL: ${window.location.origin}/#auth\n`)
      .join('\n-------------------------\n');
    navigator.clipboard.writeText(text);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2500);
  };

  const runAction = async () => {
    if (!action || actionRunning) return;
    const { kind, user } = action;
    setActionRunning(true);
    try {
      if (kind === 'approve') await clinicalApi.adminApprove(user.id);
      else if (kind === 'reject') await clinicalApi.adminReject(user.id, actionReason || undefined);
      else if (kind === 'suspend') await clinicalApi.adminSuspend(user.id, actionReason || undefined);
      else if (kind === 'reactivate') await clinicalApi.adminReactivate(user.id);
      else if (kind === 'reset-password') {
        const res = await clinicalApi.adminResetPassword(user.id);
        setIssued([res]);
      }
      setNotice({ kind: 'success', text: `${ACTION_LABEL[kind]}: done for ${user.username}.` });
      setAction(null);
      setActionReason('');
      await refresh();
      if (tab === 'audit') await loadAudit(auditUserId);
    } catch (err) {
      fail(err, `${ACTION_LABEL[kind]} failed`);
      setAction(null);
    } finally {
      setActionRunning(false);
    }
  };

  const createOne = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submittingOne) return;
    setSubmittingOne(true);
    try {
      const res = await clinicalApi.adminCreatePhysician({
        fullName: form.fullName.trim(), username: form.username.trim(), email: form.email.trim(),
        licenseNumber: form.licenseNumber.trim() || undefined, facility: form.facility.trim() || undefined,
        activateImmediately: form.activateImmediately,
      });
      setIssued([res]);
      setNotice({ kind: 'success', text: `Physician account ${res.user.username} created successfully.` });
      setForm({ fullName: '', username: '', email: '', licenseNumber: '', facility: '', activateImmediately: true });
      await refresh();
    } catch (err) {
      fail(err, 'The physician account was not created');
    } finally {
      setSubmittingOne(false);
    }
  };

  const createBatch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submittingBatch) return;
    setSubmittingBatch(true);
    try {
      const res = await clinicalApi.adminCreatePhysiciansBatch({
        count: batch.count, usernamePrefix: batch.usernamePrefix.trim(), facility: batch.facility.trim() || undefined,
      });
      setIssued(res.created);
      setNotice({ kind: 'success', text: `${res.created.length} demonstration accounts created successfully.` });
      await refresh();
    } catch (err) {
      fail(err, 'The demonstration accounts were not created');
    } finally {
      setSubmittingBatch(false);
    }
  };

  const actionButtons = (u: UserAccount) => (
    <div className="flex flex-wrap gap-1 justify-end">
      {u.status === 'PENDING_APPROVAL' && (
        <>
          <button
            onClick={() => setAction({ kind: 'approve', user: u })}
            disabled={actionRunning || loading}
            className="px-2 py-1 rounded text-[11px] font-bold text-white bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Approve
          </button>
          <button
            onClick={() => setAction({ kind: 'reject', user: u })}
            disabled={actionRunning || loading}
            className="px-2 py-1 rounded text-[11px] font-bold text-white bg-rose-600 hover:bg-rose-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reject
          </button>
        </>
      )}
      {u.status === 'ACTIVE' && (
        <button
          onClick={() => setAction({ kind: 'suspend', user: u })}
          disabled={actionRunning || loading}
          className="px-2 py-1 rounded text-[11px] font-bold text-slate-700 bg-slate-200 hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Suspend
        </button>
      )}
      {u.status === 'SUSPENDED' && (
        <button
          onClick={() => setAction({ kind: 'reactivate', user: u })}
          disabled={actionRunning || loading}
          className="px-2 py-1 rounded text-[11px] font-bold text-white bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reactivate
        </button>
      )}
      {(u.status === 'ACTIVE' || u.status === 'SUSPENDED') && (
        <button
          onClick={() => setAction({ kind: 'reset-password', user: u })}
          disabled={actionRunning || loading}
          className="px-2 py-1 rounded text-[11px] font-bold text-amber-800 bg-amber-100 hover:bg-amber-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Temp password
        </button>
      )}
      <button
        onClick={() => { setAuditUserId(u.id); setTab('audit'); }}
        disabled={actionRunning || loading}
        className="px-2 py-1 rounded text-[11px] font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Audit
      </button>
    </div>
  );

  const card = (label: string, value: number | undefined, tone: string, Icon: React.ComponentType<{ className?: string }>) => (
    <div className={`rounded-xl border p-4 bg-white shadow-xs ${tone}`}>
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">{label}</span>
        <Icon className="w-4 h-4 text-slate-400" />
      </div>
      <p className="text-2xl font-black text-slate-900 mt-1">{value ?? '—'}</p>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-5">
      <div className="border-b border-slate-200 pb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-black text-slate-900 leading-tight flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-clinical-primary" /> Account Administration
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Super Administrator console. This role administers accounts and does not open clinical assessment content.
          </p>
        </div>
        <button onClick={refresh} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-300 bg-white hover:bg-slate-50">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {notice && (
        <div role={notice.kind === 'error' ? 'alert' : 'status'}
          className={`p-3 rounded-lg border text-xs font-semibold ${notice.kind === 'error' ? 'bg-rose-50 border-rose-200 text-rose-800' : 'bg-emerald-50 border-emerald-200 text-emerald-800'}`}>
          {notice.text}
        </div>
      )}

      <nav className="flex flex-wrap gap-1 border-b border-slate-200" aria-label="Console sections">
        {([['overview', 'Overview', Users], ['users', 'All accounts', ClipboardList], ['create', 'Create physician', UserPlus], ['audit', 'Audit history', Clock]] as const).map(([id, label, Icon]) => (
          <button key={id} onClick={() => setTab(id)} aria-current={tab === id ? 'page' : undefined}
            className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold border-b-2 -mb-px ${tab === id ? 'border-clinical-primary text-clinical-primary' : 'border-transparent text-slate-500 hover:text-slate-800'}`}>
            <Icon className="w-3.5 h-3.5" /> {label}
          </button>
        ))}
      </nav>

      {tab === 'overview' && (
        <div className="space-y-5">
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-3" aria-label="Account summary">
            {card('Pending approval', summary?.pending, 'border-amber-200', Clock)}
            {card('Active', summary?.active, 'border-emerald-200', UserCheck)}
            {card('Suspended', summary?.suspended, 'border-slate-300', UserX)}
            {card('Rejected', summary?.rejected, 'border-rose-200', AlertTriangle)}
          </section>
          <section className="bg-white rounded-xl border border-slate-200 shadow-xs">
            <header className="px-4 py-3 border-b border-slate-200 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">Pending registration requests</h3>
              <span className="text-[11px] text-slate-400">{pending.length} waiting</span>
            </header>
            {pending.length === 0 ? (
              <p className="px-4 py-8 text-center text-xs text-slate-500">No registration request is waiting.</p>
            ) : (
              <ul className="divide-y divide-slate-200">
                {pending.map((u) => (
                  <li key={u.id} className="px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs">
                    <div className="min-w-0">
                      <p className="font-bold text-slate-900">{u.fullName} <span className="font-mono font-normal text-slate-500">({u.username})</span></p>
                      <p className="text-[11px] text-slate-500">{u.email} · requested {fmt(u.createdAt)}{u.licenseNumber ? ` · reg. no. ${u.licenseNumber} (unverified)` : ''}</p>
                    </div>
                    {actionButtons(u)}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}

      {tab === 'users' && (
        <section className="bg-white rounded-xl border border-slate-200 shadow-xs">
          <div className="p-4 border-b border-slate-200 grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="relative sm:col-span-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search name, username or email"
                className="w-full pl-9 pr-3 py-2 border border-slate-300 rounded-lg text-xs" aria-label="Search accounts" />
            </div>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="px-3 py-2 border border-slate-300 rounded-lg text-xs" aria-label="Status filter">
              <option value="">Status: all</option>
              {['PENDING_APPROVAL', 'ACTIVE', 'SUSPENDED', 'REJECTED'].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)} className="px-3 py-2 border border-slate-300 rounded-lg text-xs" aria-label="Role filter">
              <option value="">Role: all</option>
              <option value="PHYSICIAN">PHYSICIAN</option>
              <option value="SUPERADMIN">SUPERADMIN</option>
            </select>
          </div>
          <ul className="md:hidden divide-y divide-slate-200">
            {users.map((u) => (
              <li key={u.id} className="p-4 text-xs space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="font-bold text-slate-900">{u.fullName}</p>
                    <p className="font-mono text-[11px] text-slate-500">{u.username} · {u.email}</p>
                  </div>
                  <StatusPill status={u.status} />
                </div>
                <p className="text-[11px] text-slate-500">{u.role} · {u.assessmentCount} assessments · last sign-in {fmt(u.lastLoginAt)}{u.mustChangePassword ? ' · temporary password' : ''}</p>
                {actionButtons(u)}
              </li>
            ))}
          </ul>
          <div className="hidden md:block overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-xs text-left">
              <thead className="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="px-4 py-3">Account</th><th className="px-4 py-3">Role</th><th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Assessments</th><th className="px-4 py-3">Last sign-in</th><th className="px-4 py-3">Created</th><th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {users.length === 0 ? (
                  <tr><td colSpan={7} className="px-4 py-10 text-center text-slate-500">No account matches the filters.</td></tr>
                ) : users.map((u) => (
                  <tr key={u.id} className="hover:bg-slate-50/80">
                    <td className="px-4 py-3">
                      <p className="font-bold text-slate-900">{u.fullName}</p>
                      <p className="font-mono text-[11px] text-slate-500">{u.username} · {u.email}</p>
                      {u.mustChangePassword && <p className="text-[10px] text-amber-700">temporary password not yet changed</p>}
                    </td>
                    <td className="px-4 py-3 font-mono">{u.role}</td>
                    <td className="px-4 py-3"><StatusPill status={u.status} /></td>
                    <td className="px-4 py-3 font-mono">{u.assessmentCount}</td>
                    <td className="px-4 py-3 font-mono text-[11px]">{fmt(u.lastLoginAt)}</td>
                    <td className="px-4 py-3 font-mono text-[11px]">{fmt(u.createdAt)}</td>
                    <td className="px-4 py-3">{actionButtons(u)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {tab === 'create' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <form onSubmit={createOne} className="bg-white rounded-xl border border-slate-200 shadow-xs p-5 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">Create a physician account</h3>
            {([['fullName', 'Full name', 'text'], ['username', 'Username', 'text'], ['email', 'Email address', 'email'],
              ['licenseNumber', 'Professional registration number (optional, recorded as typed, not verified)', 'text'],
              ['facility', 'Facility (optional)', 'text']] as const).map(([id, label, type]) => (
              <div key={id}>
                <label htmlFor={`create-${id}`} className="block text-xs font-semibold text-slate-700">{label}</label>
                <input id={`create-${id}`} type={type} value={form[id]} required={id === 'fullName' || id === 'username' || id === 'email'}
                  disabled={submittingOne}
                  onChange={(e) => setForm((f) => ({ ...f, [id]: e.target.value }))}
                  className="mt-1 block w-full px-3 py-2 border border-slate-300 rounded-lg text-xs disabled:bg-slate-100 disabled:cursor-not-allowed" />
              </div>
            ))}
            <label className="flex items-center gap-2 text-xs text-slate-700">
              <input type="checkbox" checked={form.activateImmediately} disabled={submittingOne} onChange={(e) => setForm((f) => ({ ...f, activateImmediately: e.target.checked }))} />
              Activate immediately (otherwise the account waits for approval)
            </label>
            <p className="text-[11px] text-slate-500">A temporary password is generated and shown once after creation. The physician must change it at the first sign-in.</p>
            <button
              type="submit"
              disabled={submittingOne}
              className={`w-full py-2.5 rounded-lg text-xs font-bold text-white transition-all flex items-center justify-center gap-2 ${
                submittingOne
                  ? 'bg-clinical-primary/60 cursor-not-allowed opacity-80'
                  : 'bg-clinical-primary hover:bg-clinical-primary/90'
              }`}
            >
              {submittingOne && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>{submittingOne ? 'Creating physician account...' : 'Create account'}</span>
            </button>
          </form>

          <form onSubmit={createBatch} className="bg-white rounded-xl border border-slate-200 shadow-xs p-5 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">Batch accounts for a supervised demonstration</h3>
            <p className="text-[11px] text-slate-500">Creates numbered accounts (prefix-01, prefix-02, ...) with example.org addresses, each with its own temporary password. Up to ten.</p>
            <div>
              <label htmlFor="batch-prefix" className="block text-xs font-semibold text-slate-700">Username prefix</label>
              <input id="batch-prefix" value={batch.usernamePrefix} disabled={submittingBatch} onChange={(e) => setBatch((b) => ({ ...b, usernamePrefix: e.target.value }))} className="mt-1 block w-full px-3 py-2 border border-slate-300 rounded-lg text-xs disabled:bg-slate-100 disabled:cursor-not-allowed" />
            </div>
            <div>
              <label htmlFor="batch-count" className="block text-xs font-semibold text-slate-700">Number of accounts (1–10)</label>
              <input id="batch-count" type="number" min={1} max={10} value={batch.count} disabled={submittingBatch} onChange={(e) => setBatch((b) => ({ ...b, count: Number(e.target.value) }))} className="mt-1 block w-full px-3 py-2 border border-slate-300 rounded-lg text-xs disabled:bg-slate-100 disabled:cursor-not-allowed" />
            </div>
            <div>
              <label htmlFor="batch-facility" className="block text-xs font-semibold text-slate-700">Facility (optional)</label>
              <input id="batch-facility" value={batch.facility} disabled={submittingBatch} onChange={(e) => setBatch((b) => ({ ...b, facility: e.target.value }))} className="mt-1 block w-full px-3 py-2 border border-slate-300 rounded-lg text-xs disabled:bg-slate-100 disabled:cursor-not-allowed" />
            </div>
            <button
              type="submit"
              disabled={submittingBatch}
              className={`w-full py-2.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 ${
                submittingBatch
                  ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                  : 'text-slate-800 bg-slate-200 hover:bg-slate-300'
              }`}
            >
              {submittingBatch && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>{submittingBatch ? 'Generating demonstration accounts...' : 'Create demonstration accounts'}</span>
            </button>
          </form>
        </div>
      )}

      {tab === 'audit' && (
        <section className="bg-white rounded-xl border border-slate-200 shadow-xs">
          <header className="px-4 py-3 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">Account-administration audit history</h3>
            <div className="flex items-center gap-2 text-[11px]">
              {auditUserId && (
                <span className="font-mono text-slate-500">filtered to account {auditUserId.slice(0, 8)}…
                  <button onClick={() => setAuditUserId('')} className="ml-2 underline font-semibold">show all</button>
                </span>
              )}
              <span className="text-slate-400">{events.length} events · read-only</span>
            </div>
          </header>
          {events.length === 0 ? (
            <p className="px-4 py-8 text-center text-xs text-slate-500">No administrative event recorded.</p>
          ) : (
            <ul className="divide-y divide-slate-200">
              {events.map((e) => (
                <li key={e.id} className="px-4 py-3 text-xs flex flex-wrap gap-x-4 gap-y-1">
                  <span className="font-mono text-[11px] text-slate-500 w-40 flex-shrink-0">{fmt(e.timestamp)}</span>
                  <span className="font-mono text-[11px] font-bold text-slate-700 w-64 flex-shrink-0">{e.eventType || e.action}</span>
                  <span className={`text-[10px] font-bold uppercase ${e.outcome === 'failure' ? 'text-rose-700' : 'text-emerald-700'}`}>{e.outcome}</span>
                  <span className="text-slate-700 basis-full sm:basis-auto sm:flex-1">{e.details} <span className="text-slate-400">— by {e.actor}</span></span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {action && (
        <div role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="bg-white rounded-xl shadow-2xl border border-slate-300 max-w-md w-full p-5 space-y-4">
            <h3 id="confirm-title" className="text-sm font-bold text-slate-900">{ACTION_LABEL[action.kind]}</h3>
            <p className="text-xs text-slate-600">
              Account <span className="font-mono font-bold">{action.user.username}</span> ({action.user.fullName}), currently <StatusPill status={action.user.status} />.
              {action.kind === 'suspend' && ' Every session of this account will be invalidated.'}
              {action.kind === 'reset-password' && ' The current password stops working; a new temporary password will be shown once.'}
              {action.kind === 'reject' && ' The applicant will not be able to sign in.'}
            </p>
            {(action.kind === 'reject' || action.kind === 'suspend') && (
              <div>
                <label htmlFor="action-reason" className="block text-xs font-semibold text-slate-700">Reason (optional, recorded in the audit log)</label>
                <input id="action-reason" value={actionReason} onChange={(e) => setActionReason(e.target.value)} className="mt-1 block w-full px-3 py-2 border border-slate-300 rounded-lg text-xs" />
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                disabled={actionRunning}
                onClick={() => { setAction(null); setActionReason(''); }}
                className="px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-300 rounded-lg hover:bg-slate-100 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                disabled={actionRunning}
                onClick={runAction}
                className="px-4 py-1.5 text-xs font-bold text-white bg-clinical-primary rounded-lg hover:bg-clinical-primary/90 inline-flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionRunning ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                <span>{actionRunning ? 'Executing...' : 'Confirm'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {issued && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="issued-title"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs"
        >
          <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-lg w-full overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            <div className="p-4 sm:p-5 border-b border-slate-100 flex items-start justify-between bg-amber-50/80">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-amber-100 border border-amber-200 flex items-center justify-center text-amber-800 shrink-0">
                  <KeyRound className="w-5 h-5" />
                </div>
                <div>
                  <h3 id="issued-title" className="text-sm font-black text-slate-900">
                    Temporary Credentials Generated
                  </h3>
                  <p className="text-[11px] text-slate-600 mt-0.5">
                    {issued.length > 1
                      ? `${issued.length} demonstration accounts generated successfully.`
                      : 'Physician account created successfully.'}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  setIssued(null);
                  setTab('users');
                }}
                className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-amber-100/50 transition-colors"
                aria-label="Close dialog"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 sm:p-5 space-y-3 max-h-[60vh] overflow-y-auto">
              <div className="p-3 bg-amber-50/90 border border-amber-200 rounded-lg text-xs text-amber-900 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong>Shown once:</strong> The server never stores or logs plaintext temporary passwords. Clinicians are required to set their own permanent password on first sign-in.
                </p>
              </div>

              <div className="space-y-2">
                {issued.map((c) => (
                  <div
                    key={c.user.id}
                    className="p-3 rounded-lg border border-slate-200 bg-slate-50/80 flex items-center justify-between gap-3 text-xs hover:border-slate-300 transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="font-bold text-slate-900 truncate">{c.user.fullName || c.user.username}</p>
                      <p className="font-mono text-[11px] text-slate-500 truncate">{c.user.username}</p>
                      <div className="mt-1 flex items-center gap-1.5">
                        <span className="text-[10px] uppercase font-bold text-slate-400">Temp Pass:</span>
                        <code className="font-mono font-bold text-amber-900 bg-amber-100/80 px-2 py-0.5 rounded text-[11px] select-all border border-amber-200/50">
                          {c.temporaryPassword}
                        </code>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => copyCredential(c.user.username, c.temporaryPassword)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-100 text-xs font-semibold text-slate-700 shrink-0 transition-colors shadow-2xs"
                    >
                      {copiedId === c.user.username ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-600" />
                          <span className="text-emerald-700">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-slate-500" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 border-t border-slate-100 bg-slate-50 flex flex-wrap items-center justify-between gap-2">
              <button
                type="button"
                onClick={copyAllCredentials}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-100 text-xs font-bold text-slate-800 transition-colors"
              >
                {copiedAll ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4 text-slate-600" />}
                <span>{copiedAll ? 'All Copied to Clipboard!' : 'Copy All Credentials'}</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setIssued(null);
                  setTab('users');
                }}
                className="px-4 py-2 rounded-lg bg-clinical-primary hover:bg-clinical-primary/90 text-xs font-bold text-white transition-colors"
              >
                Done &amp; View All Accounts
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
