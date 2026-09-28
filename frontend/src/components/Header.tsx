import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Lock, LogOut, Clock, Eye, Search, PlusCircle, AlertCircle } from 'lucide-react';
import { ClinicianUser } from '../types/clinical';

interface HeaderProps {
  currentUser: ClinicianUser | null;
  onNavigate: (screen: 'dashboard' | 'new_assessment' | 'history' | 'signin') => void;
  activeScreen: string;
  onLogout: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentUser,
  onNavigate,
  activeScreen,
  onLogout,
}) => {
  const [secondsRemaining, setSecondsRemaining] = useState<number>(15 * 60);
  const [showTimeoutWarning, setShowTimeoutWarning] = useState<boolean>(false);

  // Inactivity countdown per NHS / Clinical Workstation safety policy
  useEffect(() => {
    if (!currentUser) return;

    const timer = setInterval(() => {
      setSecondsRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          onLogout();
          return 0;
        }
        if (prev <= 120 && !showTimeoutWarning) {
          setShowTimeoutWarning(true);
        }
        return prev - 1;
      });
    }, 1000);

    // Reset countdown on user interaction
    const handleActivity = () => {
      setSecondsRemaining(15 * 60);
      if (showTimeoutWarning) setShowTimeoutWarning(false);
    };

    window.addEventListener('mousemove', handleActivity);
    window.addEventListener('keydown', handleActivity);

    return () => {
      clearInterval(timer);
      window.removeEventListener('mousemove', handleActivity);
      window.removeEventListener('keydown', handleActivity);
    };
  }, [currentUser, showTimeoutWarning, onLogout]);

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <>
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm" role="banner">
        {/* Top Regulatory Safety Ribbon */}
        <div className="bg-slate-900 text-slate-300 text-[11px] px-4 py-1 flex items-center justify-between font-mono">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-teal-400"></span>
            <span className="font-semibold text-slate-100 uppercase tracking-wider">
              AI Clinical Decision Support Aid (Research & Decision Support)
            </span>
            <span className="text-slate-400 hidden sm:inline">|</span>
            <span className="text-slate-400 hidden sm:inline">
              Preliminary computational observation only — final diagnosis rests exclusively with the reviewing clinician.
            </span>
          </div>

          <div className="flex items-center space-x-3">
            {/* TLS 1.3 Badge */}
            <span
              className="inline-flex items-center px-1.5 py-0.5 rounded bg-slate-800 text-teal-300 text-[10px] font-mono border border-slate-700"
              title="TLS 1.3 End-to-End Encrypted Session"
            >
              <Lock className="w-2.5 h-2.5 mr-1 text-teal-400" />
              TLS 1.3 Secured
            </span>

            {/* Inactivity Countdown */}
            {currentUser && (
              <span
                className={`inline-flex items-center text-[10px] font-mono ${
                  secondsRemaining < 180 ? 'text-amber-400 font-bold animate-pulse' : 'text-slate-400'
                }`}
                title="Clinical workstation auto-lock timer (15 min inactivity policy)"
              >
                <Clock className="w-2.5 h-2.5 mr-1" />
                Session: {formatTime(secondsRemaining)}
              </span>
            )}
          </div>
        </div>

        {/* Primary Navbar */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => onNavigate('dashboard')}
              className="flex items-center space-x-3 text-left focus-visible:ring-2 focus-visible:ring-clinical-primary focus-visible:outline-none rounded-lg p-1"
              aria-label="Diabetic Retinopathy CDSS Home"
            >
              <div className="p-2 bg-clinical-primary rounded-lg text-white shadow-sm flex items-center justify-center">
                <Eye className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-base font-bold text-slate-900 leading-tight">
                  DR-CDSS <span className="text-xs font-normal text-slate-500 uppercase tracking-wider">Clinical Suite</span>
                </h1>
                <p className="text-[11px] text-slate-500">
                  Diabetic Retinopathy Decision Support • EfficientNet-B0
                </p>
              </div>
            </button>

            {/* Navigation tabs */}
            {currentUser && (
              <nav className="hidden md:flex items-center space-x-1 ml-6" aria-label="Clinical Navigation">
                <button
                  onClick={() => onNavigate('dashboard')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                    activeScreen === 'dashboard'
                      ? 'bg-clinical-primary-light text-clinical-primary font-bold'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                  aria-current={activeScreen === 'dashboard' ? 'page' : undefined}
                >
                  <Activity className="w-3.5 h-3.5 inline mr-1.5" />
                  Worklist
                </button>

                <button
                  onClick={() => onNavigate('new_assessment')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                    activeScreen === 'new_assessment' || activeScreen === 'validation'
                      ? 'bg-clinical-primary-light text-clinical-primary font-bold'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                  aria-current={activeScreen === 'new_assessment' ? 'page' : undefined}
                >
                  <PlusCircle className="w-3.5 h-3.5 inline mr-1.5" />
                  New Assessment
                </button>

                <button
                  onClick={() => onNavigate('history')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                    activeScreen === 'history'
                      ? 'bg-clinical-primary-light text-clinical-primary font-bold'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                  aria-current={activeScreen === 'history' ? 'page' : undefined}
                >
                  <Search className="w-3.5 h-3.5 inline mr-1.5" />
                  Record History
                </button>
              </nav>
            )}
          </div>

          {/* Right Session / User Area */}
          <div className="flex items-center space-x-3">
            {currentUser ? (
              <div className="flex items-center space-x-3">
                <div className="text-right hidden sm:block">
                  <div className="text-xs font-bold text-slate-900 leading-tight">
                    {currentUser.name}
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    {currentUser.licenseNumber} • {currentUser.facility.split('—')[0]}
                  </div>
                </div>

                <div
                  className="w-8 h-8 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center font-bold text-xs border border-teal-200"
                  aria-hidden="true"
                >
                  {currentUser.name.split(' ')[1]?.[0] || 'DR'}
                </div>

                <button
                  onClick={onLogout}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition-colors focus-visible:ring-2 focus-visible:ring-rose-500"
                  title="Sign out of clinical session"
                  aria-label="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <span className="text-xs font-medium text-slate-500 flex items-center">
                <ShieldCheck className="w-3.5 h-3.5 mr-1 text-slate-400" />
                Unauthenticated Clinical Workstation
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Inactivity Warning Modal */}
      {showTimeoutWarning && (
        <div
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="timeout-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
        >
          <div className="bg-white rounded-xl shadow-2xl border border-amber-300 max-w-md w-full p-6 space-y-4">
            <div className="flex items-center space-x-3 text-amber-600">
              <AlertCircle className="w-6 h-6 flex-shrink-0" />
              <h3 id="timeout-title" className="text-base font-bold text-slate-900">
                Clinical Session Inactivity Warning
              </h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Per healthcare data protection standards (NHS Caldicott / HIPAA), your session will automatically terminate in{' '}
              <span className="font-mono font-bold text-amber-600">{formatTime(secondsRemaining)}</span> due to workstation inactivity.
            </p>
            <div className="flex justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={onLogout}
                className="px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 rounded-lg border border-slate-300"
              >
                Log Out Now
              </button>
              <button
                type="button"
                onClick={() => {
                  setSecondsRemaining(15 * 60);
                  setShowTimeoutWarning(false);
                }}
                className="px-4 py-1.5 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm"
              >
                Keep Session Active
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
