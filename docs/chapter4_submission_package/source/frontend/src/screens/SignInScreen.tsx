import React, { useState } from 'react';
import { Shield, Lock, Eye, AlertTriangle, Building2, CheckCircle2, UserCheck } from 'lucide-react';
import { ClinicianUser } from '../types/clinical';
import { clinicalApi } from '../services/api';

interface SignInScreenProps {
  onSignInSuccess: (user: ClinicianUser) => void;
}

export const SignInScreen: React.FC<SignInScreenProps> = ({ onSignInSuccess }) => {
  const [email, setEmail] = useState<string>('dr.demo@research-prototype.local');
  const [password, setPassword] = useState<string>('••••••••••••');
  const [password, setPassword] = useState<string>('Research Prototype Environment');
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
    } catch {
      setErrorMessage('Authentication failed. Please verify clinical staff ID and credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickDemoFill = (role: 'consultant' | 'optometrist') => {
    if (role === 'consultant') {
      setEmail('dr.demo@research-prototype.local');
      setPassword('password123');
    } else {
      setEmail('optometrist.demo@research-prototype.local');
      setPassword('password123');
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
                  placeholder="e.g. dr.demo@research-prototype.local"
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

            

        {/* Footer Reference */}
        <p className="mt-4 text-center text-[11px] text-slate-400">
          Academic Research Prototype • PGD Computer Science, Faculty of Physical Sciences
        </p>
      </div>
    </div>
  );
};

