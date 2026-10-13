import React, { useState } from 'react';
import { KeyRound, ShieldAlert } from 'lucide-react';
import { ClinicianUser } from '../types/clinical';
import { clinicalApi } from '../services/api';
import { describeApiError } from '../utils/apiError';
import { passwordProblem } from '../utils/session';

interface ChangePasswordScreenProps {
  currentUser: ClinicianUser;
  /** true when the account holds a temporary password and nothing else may happen first */
  mandatory: boolean;
  onChanged: () => void;
  onCancel?: () => void;
}

/**
 * Password change. When the password is temporary (set by the administrator
 * or the bootstrap), this screen is the only one the account can reach, and
 * a successful change signs the user out of every session, including this
 * one, so they sign in again with the password they chose.
 */
export const ChangePasswordScreen: React.FC<ChangePasswordScreenProps> = ({ currentUser, mandatory, onChanged, onCancel }) => {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const problem = passwordProblem(newPassword);
    if (problem) { setError(problem); return; }
    if (newPassword !== confirmation) { setError('The new password and its confirmation do not match.'); return; }
    if (newPassword === currentPassword) { setError('The new password must differ from the current one.'); return; }
    setSubmitting(true);
    try {
      await clinicalApi.changePassword({ currentPassword, newPassword, newPasswordConfirmation: confirmation });
      onChanged();
    } catch (err) {
      setError(describeApiError(err, 'The password was not changed'));
    } finally {
      setSubmitting(false);
    }
  };

  const input = (id: string, label: string, value: string, onChange: (v: string) => void, autoComplete: string) => (
    <div>
      <label htmlFor={id} className="block text-xs font-semibold text-slate-700">{label}</label>
      <input id={id} type="password" required value={value} onChange={(e) => onChange(e.target.value)} autoComplete={autoComplete}
        className="mt-1 appearance-none block w-full px-3 py-2 border border-slate-300 rounded-lg shadow-sm text-xs focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary" />
    </div>
  );

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-center py-10 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center space-y-2">
        <div className="inline-flex p-3 bg-clinical-primary rounded-2xl text-white shadow-md">
          <KeyRound className="h-7 w-7" />
        </div>
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">
          {mandatory ? 'Set Your Password' : 'Change Password'}
        </h2>
        <p className="text-xs text-slate-500">Signed in as <span className="font-mono">{currentUser.username}</span> ({currentUser.role})</p>
      </div>

      <div className="mt-6 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-7 px-6 shadow-xl rounded-2xl sm:px-9 border border-slate-200 space-y-5">
          {mandatory && (
            <div className="p-3.5 bg-amber-50 rounded-xl border border-amber-200 flex items-start gap-3 text-xs text-amber-900">
              <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                The password you signed in with is temporary. Choose your own password before continuing; nothing else
                is available until you do. Afterwards you will sign in again with the new password.
              </p>
            </div>
          )}
          {error && <div role="alert" className="p-3 bg-rose-50 text-rose-800 text-xs rounded-lg border border-rose-200">{error}</div>}
          <form className="space-y-4" onSubmit={handleSubmit}>
            {input('current-password', mandatory ? 'Temporary password' : 'Current password', currentPassword, setCurrentPassword, 'current-password')}
            {input('new-password', 'New password (at least 10 characters, letters and digits)', newPassword, setNewPassword, 'new-password')}
            {input('confirm-password', 'Confirm new password', confirmation, setConfirmation, 'new-password')}
            <button type="submit" disabled={submitting}
              className="w-full flex justify-center py-2.5 px-4 rounded-lg shadow-sm text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary/90 disabled:opacity-50">
              {submitting ? 'Saving...' : 'Change password'}
            </button>
            {!mandatory && onCancel && (
              <button type="button" onClick={onCancel} className="w-full text-[11px] font-semibold text-slate-500 hover:text-slate-800">Cancel</button>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};
