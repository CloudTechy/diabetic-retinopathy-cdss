import React, { useState } from 'react';
import { Lock, Eye, AlertTriangle, UserPlus } from 'lucide-react';
import { ClinicianUser } from '../types/clinical';
import { clinicalApi } from '../services/api';
import { describeApiError } from '../utils/apiError';

interface SignInScreenProps {
  onSignInSuccess: (user: ClinicianUser) => void;
  onRegister: () => void;
}

export const SignInScreen: React.FC<SignInScreenProps> = ({ onSignInSuccess, onRegister }) => {
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const user = await clinicalApi.login({
        username: email,
        password: password,
      });
      onSignInSuccess(user);
    } catch (err: any) {
      // 401 is a credentials problem; 403 is the account's state (pending,
      // rejected, suspended), which the server states in its own words
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      setErrorMessage(
        status === 401
          ? 'Authentication failed. Please verify your username and password.'
          : status === 403 && typeof detail === 'string'
          ? detail
          : describeApiError(err, 'Sign-in did not complete')
      );
    } finally {
      setIsLoading(false);
    }
  };

  // No demonstration account exists any more. Earlier revisions offered two
  // personas, then one "Fill demonstration credentials" button; accounts now
  // come from the Super Administrator or from an approved registration.

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-center py-12 sm:px-6 lg:px-8 bg-slate-50">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Logo and Headings */}
        <div className="text-center space-y-2">
          <div className="inline-flex p-3 bg-clinical-primary rounded-2xl text-white shadow-md">
            <Eye className="h-8 w-8" />
          </div>
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">
            Clinical Practitioner Authentication
          </h2>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            Diabetic Retinopathy Clinical Decision Support System (DR-CDSS)
          </p>
        </div>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-6 shadow-xl rounded-2xl sm:px-10 border border-slate-200 space-y-6">
          {/* Scope banner. Deliberately not phrased as a device
              classification: this is a research prototype. */}
          <div className="p-3.5 bg-amber-50 rounded-xl border border-amber-200 flex items-start space-x-3 text-xs text-amber-900">
            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">Restricted Clinical Decision Support System</span>
              <p className="text-[11px] text-amber-800 leading-relaxed">
                Research prototype. Accounts are created or approved by a Super Administrator; every action is logged against the signed-in account.
              </p>
            </div>
          </div>

          {errorMessage && (
            <div className="p-3 bg-rose-50 text-rose-800 text-xs rounded-lg border border-rose-200">
              {errorMessage}
            </div>
          )}

          {/* Form */}
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="staff-id" className="block text-xs font-semibold text-slate-700">
                Username or email
              </label>
              <div className="mt-1">
                <input
                  id="staff-id"
                  name="email"
                  type="text"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-slate-300 rounded-lg shadow-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary text-xs"
                  placeholder="your username"
                  autoComplete="username"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-xs font-semibold text-slate-700">
                Password
              </label>
              <div className="mt-1">
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-slate-300 rounded-lg shadow-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary text-xs"
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-lg shadow-sm text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-clinical-primary transition-all disabled:opacity-50"
              >
                {isLoading ? (
                  <span className="inline-flex items-center gap-2">
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Authenticating...
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2">
                    <Lock className="w-3.5 h-3.5" />
                    Practitioner Sign In
                  </span>
                )}
              </button>
            </div>
          </form>

          {/* Registration: a prospective physician requests an account; a Super Administrator approves it */}
          <div className="pt-2 border-t border-slate-100">
            <span className="text-[11px] font-semibold text-slate-500 block mb-2">
              No account yet? A physician can request one; it becomes usable once a Super Administrator approves it.
            </span>
            <button
              type="button"
              onClick={onRegister}
              className="w-full p-2.5 border border-clinical-primary/40 rounded-lg text-left hover:bg-slate-50 transition-colors text-[11px]"
            >
              <div className="flex items-center gap-1.5 font-semibold text-clinical-primary">
                <UserPlus className="w-3.5 h-3.5" />
                Request an account
              </div>
            </button>
          </div>
        </div>

        {/* Footer Reference */}
        <p className="mt-4 text-center text-[11px] text-slate-400">
          Academic Research Prototype • PGD Computer Science, Faculty of Physical Sciences
        </p>
      </div>
    </div>
  );
};
