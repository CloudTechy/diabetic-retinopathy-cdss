import React, { useState } from 'react';
import { ArrowLeft, CheckCircle2, UserPlus } from 'lucide-react';
import { clinicalApi } from '../services/api';
import { describeApiError } from '../utils/apiError';
import { validateRegistration } from '../utils/session';

interface RegisterScreenProps {
  onBackToSignIn: () => void;
}

/**
 * Physician self-registration. The request creates a PENDING_APPROVAL
 * account; the applicant is told so and is NOT signed in. There is no role
 * or status field on this form or in the request it sends.
 */
export const RegisterScreen: React.FC<RegisterScreenProps> = ({ onBackToSignIn }) => {
  const [form, setForm] = useState({
    fullName: '', username: '', email: '', licenseNumber: '', facility: '', password: '', passwordConfirmation: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState<{ username: string; message: string } | null>(null);

  const set = (field: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const problems = validateRegistration(form);
    setErrors(problems);
    setServerError(null);
    if (Object.keys(problems).length) return;
    setSubmitting(true);
    try {
      const res = await clinicalApi.register({
        fullName: form.fullName.trim(),
        username: form.username.trim(),
        email: form.email.trim(),
        password: form.password,
        passwordConfirmation: form.passwordConfirmation,
        licenseNumber: form.licenseNumber.trim() || undefined,
        facility: form.facility.trim() || undefined,
      });
      setSubmitted({ username: res.username, message: res.message });
    } catch (err) {
      setServerError(describeApiError(err, 'The registration request was not submitted'));
    } finally {
      setSubmitting(false);
    }
  };

  const field = (
    id: keyof typeof form, label: string, type = 'text', required = true, hint?: string,
  ) => (
    <div>
      <label htmlFor={`reg-${id}`} className="block text-xs font-semibold text-slate-700">
        {label}{required ? '' : <span className="text-slate-400 font-normal"> (optional)</span>}
      </label>
      <input
        id={`reg-${id}`}
        type={type}
        value={form[id]}
        onChange={set(id)}
        required={required}
        autoComplete={type === 'password' ? 'new-password' : 'off'}
        className="mt-1 appearance-none block w-full px-3 py-2 border border-slate-300 rounded-lg shadow-sm text-xs focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary"
      />
      {hint && !errors[id] && <p className="mt-1 text-[11px] text-slate-400">{hint}</p>}
      {errors[id] && <p role="alert" className="mt-1 text-[11px] text-rose-700 font-semibold">{errors[id]}</p>}
    </div>
  );

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-center py-10 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center space-y-2">
        <div className="inline-flex p-3 bg-clinical-primary rounded-2xl text-white shadow-md">
          <UserPlus className="h-7 w-7" />
        </div>
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">Request a Physician Account</h2>
        <p className="text-xs text-slate-500">
          A Super Administrator reviews every request. Nothing can be done with the account until it is approved.
        </p>
      </div>

      <div className="mt-6 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-7 px-6 shadow-xl rounded-2xl sm:px-9 border border-slate-200 space-y-5">
          {submitted ? (
            <div className="space-y-4" role="status">
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-start gap-3 text-xs text-emerald-900">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <span className="font-bold block">Registration request submitted</span>
                  <p className="leading-relaxed">{submitted.message}</p>
                  <p className="font-mono text-[11px] text-emerald-800">Username: {submitted.username} · Status: PENDING_APPROVAL</p>
                </div>
              </div>
              <button type="button" onClick={onBackToSignIn}
                className="w-full py-2.5 rounded-lg text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary/90">
                Return to sign-in
              </button>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit} noValidate>
              {serverError && (
                <div role="alert" className="p-3 bg-rose-50 text-rose-800 text-xs rounded-lg border border-rose-200">{serverError}</div>
              )}
              {field('fullName', 'Full name')}
              {field('username', 'Username', 'text', true, 'Letters, digits, dot, underscore or hyphen.')}
              {field('email', 'Email address', 'email')}
              {field('licenseNumber', 'Professional registration number', 'text', false,
                'Recorded as typed. The system does not verify it against any professional register.')}
              {field('facility', 'Facility', 'text', false)}
              {field('password', 'Password', 'password', true, 'At least 10 characters with letters and digits.')}
              {field('passwordConfirmation', 'Confirm password', 'password')}
              <button type="submit" disabled={submitting}
                className="w-full flex justify-center py-2.5 px-4 rounded-lg shadow-sm text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary/90 disabled:opacity-50">
                {submitting ? 'Submitting...' : 'Submit registration request'}
              </button>
            </form>
          )}

          <button type="button" onClick={onBackToSignIn}
            className="w-full inline-flex items-center justify-center gap-1.5 text-[11px] font-semibold text-slate-500 hover:text-slate-800">
            <ArrowLeft className="w-3.5 h-3.5" /> Back to sign-in
          </button>
        </div>
      </div>
    </div>
  );
};
