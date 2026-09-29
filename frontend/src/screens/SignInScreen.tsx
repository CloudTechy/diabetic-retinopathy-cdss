import React, { useState } from 'react';
import { Shield, Lock, Eye, AlertTriangle, Building2, CheckCircle2, UserCheck } from 'lucide-react';
import { ClinicianUser } from '../types/clinical';
import { clinicalApi } from '../services/api';

interface SignInScreenProps {
  onSignInSuccess: (user: ClinicianUser) => void;
}

export const SignInScreen: React.FC<SignInScreenProps> = ({ onSignInSuccess }) => {
  const [email, setEmail] = useState<string>('dr.adaeze@retina-research.org');
  const [password, setPassword] = useState<string>('••••••••••••');
  const [facility, setFacility] = useState<string>('St. Jude Retinal Diagnostic Unit — Ward 4B');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const user = await clinicalApi.login({
        email,
        facility,
      });
      onSignInSuccess(user);
    } catch {
      setErrorMessage('Authentication failed. Please verify clinical staff ID and credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickDemoFill = (role: 'consultant' | 'optometrist') => {
    if (role === 'consultant') {
      setEmail('dr.adaeze@retina-research.org');
      setFacility('St. Jude Retinal Diagnostic Unit — Ward 4B');
    } else {
      setEmail('t.evans@community-eye-study.org');
      setFacility('Community Optometry Screening Centre — Suite 2');
    }
  };

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
          {/* SaMD Disclaimer Banner */}
          <div className="p-3.5 bg-amber-50 rounded-xl border border-amber-200 flex items-start space-x-3 text-xs text-amber-900">
            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">Restricted Clinical Decision Support System</span>
              <p className="text-[11px] text-amber-800 leading-relaxed">
                Authorized for credentialed healthcare practitioners only. All activity is logged and cryptographically bound to staff identity.
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
                Clinical Practitioner ID / Username
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
                  placeholder="e.g. dr.adaeze@retina-research.org"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-xs font-semibold text-slate-700">
                Password / Secure PIN
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
              <label htmlFor="facility-select" className="block text-xs font-semibold text-slate-700">
                Clinical Facility / Screening Centre
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                  <Building2 className="h-4 w-4" />
                </div>
                <select
                  id="facility-select"
                  value={facility}
                  onChange={(e) => setFacility(e.target.value)}
                  className="block w-full pl-9 pr-3 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary bg-white"
                >
                  <option value="St. Jude Retinal Diagnostic Unit — Ward 4B">
                    St. Jude Retinal Diagnostic Unit — Ward 4B
                  </option>
                  <option value="Community Optometry Screening Centre — Suite 2">
                    Community Optometry Screening Centre — Suite 2
                  </option>
                  <option value="Regional Eye Hospital — Vitreoretinal Service">
                    Regional Eye Hospital — Vitreoretinal Service
                  </option>
                </select>
              </div>
            </div>

            {/* Session Notice: Auto-timeout */}
            <div className="pt-2 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-100">
              <span className="flex items-center text-teal-700 font-mono font-medium">
                <Lock className="w-3.5 h-3.5 mr-1 text-teal-600" />
                Authenticated Session
              </span>
              <span className="font-mono text-slate-500">
                Timeout: 15 min inactivity
              </span>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center items-center py-2.5 px-4 border border-transparent rounded-lg shadow-sm text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-clinical-primary transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <span>Authenticating with Clinical Directory...</span>
              ) : (
                <>
                  <Shield className="w-4 h-4 mr-2" />
                  Sign In to Clinical Session
                </>
              )}
            </button>
          </form>

          {/* Quick Credential Pre-fill for Demonstration */}
          <div className="pt-4 border-t border-slate-200">
            <p className="text-[11px] font-semibold text-slate-500 mb-2 uppercase tracking-wider text-center">
              Demonstration Research Roles
            </p>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => handleQuickDemoFill('consultant')}
                className="p-2 border border-slate-200 hover:border-teal-500 hover:bg-teal-50/50 rounded-lg text-left text-xs transition focus-visible:ring-2 focus-visible:ring-clinical-primary"
              >
                <div className="font-bold text-slate-900 flex items-center gap-1">
                  <UserCheck className="w-3.5 h-3.5 text-teal-600" />
                  Consultant
                </div>
                <div className="text-[10px] text-slate-500 truncate">
                  Dr. Adaeze Okonjo
                </div>
              </button>

              <button
                type="button"
                onClick={() => handleQuickDemoFill('optometrist')}
                className="p-2 border border-slate-200 hover:border-teal-500 hover:bg-teal-50/50 rounded-lg text-left text-xs transition focus-visible:ring-2 focus-visible:ring-clinical-primary"
              >
                <div className="font-bold text-slate-900 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
                  Optometrist
                </div>
                <div className="text-[10px] text-slate-500 truncate">
                  T. Evans, MCOptom
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Footer Reference */}
        <p className="mt-4 text-center text-[11px] text-slate-400">
          Academic Research Prototype • MSc Biomedical Software Engineering
        </p>
      </div>
    </div>
  );
};
