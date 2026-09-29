import React, { useState, useEffect } from 'react';
import { ClinicianUser, AssessmentRecord, EyeLaterality } from './types/clinical';
import { clinicalApi, DEFAULT_CLINICIAN } from './services/api';
import { Header } from './components/Header';
import { SignInScreen } from './screens/SignInScreen';
import { DashboardScreen } from './screens/DashboardScreen';
import { NewAssessmentScreen } from './screens/NewAssessmentScreen';
import { ValidationStepperScreen } from './screens/ValidationStepperScreen';
import { DecisionSupportScreen } from './screens/DecisionSupportScreen';
import { ProfessionalReviewModal } from './screens/ProfessionalReviewModal';
import { CompletedAssessmentScreen } from './screens/CompletedAssessmentScreen';
import { RecordHistoryScreen } from './screens/RecordHistoryScreen';

export type ScreenState =
  | 'signin'
  | 'dashboard'
  | 'new_assessment'
  | 'validation'
  | 'decision_support'
  | 'completed_assessment'
  | 'history';

export const App: React.FC = () => {
  // Authentication State
  const [currentUser, setCurrentUser] = useState<ClinicianUser | null>(DEFAULT_CLINICIAN);
  const [activeScreen, setActiveScreen] = useState<ScreenState>('dashboard');

  // Currently Active Assessment
  const [activeAssessment, setActiveAssessment] = useState<AssessmentRecord | null>(null);

  // Review Modal Visibility (Screen 6)
  const [showReviewModal, setShowReviewModal] = useState<boolean>(false);

  // Toast notification
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // Support URL Hash navigation (e.g. #signin, #dashboard, #decision_support, #review, #completed, #history, #validation, #rejected)
  useEffect(() => {
    const handleHash = async () => {
      const hash = window.location.hash.replace('#', '').toLowerCase();
      const records = await clinicalApi.getWorklist();
      const pendingRecord = records.find(r => r.status === 'needs_review') || records[0];
      const completedRecord = records.find(r => r.status === 'completed') || records[1];
      const rejectedRecord = records.find(r => r.status === 'rejected') || records[3] || records[0];

      if (hash === 'signin') {
        setCurrentUser(null);
        setActiveScreen('signin');
        setShowReviewModal(false);
      } else {
        setCurrentUser(DEFAULT_CLINICIAN);
        if (hash === 'dashboard' || !hash) {
          setActiveScreen('dashboard');
          setShowReviewModal(false);
        } else if (hash === 'new_assessment') {
          setActiveScreen('new_assessment');
          setShowReviewModal(false);
        } else if (hash === 'validation') {
          setActiveAssessment(pendingRecord);
          setActiveScreen('validation');
          setShowReviewModal(false);
        } else if (hash === 'rejected') {
          setActiveAssessment(rejectedRecord);
          setActiveScreen('validation');
          setShowReviewModal(false);
        } else if (hash === 'decision_support') {
          setActiveAssessment(pendingRecord);
          setActiveScreen('decision_support');
          setShowReviewModal(false);
        } else if (hash === 'review') {
          setActiveAssessment(pendingRecord);
          setActiveScreen('decision_support');
          setShowReviewModal(true);
        } else if (hash === 'completed') {
          setActiveAssessment(completedRecord);
          setActiveScreen('completed_assessment');
          setShowReviewModal(false);
        } else if (hash === 'history') {
          setActiveScreen('history');
          setShowReviewModal(false);
        }
      }
    };

    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  // Keyboard shortcut listener: Escape closes modals; Alt+D -> Dashboard
  useEffect(() => {
    const handleGlobalKeys = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (showReviewModal) setShowReviewModal(false);
      } else if (e.altKey && (e.key === 'd' || e.key === 'D')) {
        e.preventDefault();
        setActiveScreen('dashboard');
      }
    };
    window.addEventListener('keydown', handleGlobalKeys);
    return () => window.removeEventListener('keydown', handleGlobalKeys);
  }, [showReviewModal]);

  // Screen 1: Sign in handler
  const handleSignInSuccess = (user: ClinicianUser) => {
    setCurrentUser(user);
    setActiveScreen('dashboard');
    showToast(`Welcome back, ${user.name}`);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setActiveAssessment(null);
    setActiveScreen('signin');
    setShowReviewModal(false);
    showToast('Signed out of clinical session.');
  };

  // Navigation from Header
  const handleNavigate = (screen: 'dashboard' | 'new_assessment' | 'history' | 'signin') => {
    setActiveScreen(screen);
  };

  // Screen 2 Worklist select assessment
  const handleSelectAssessmentFromWorklist = (assessment: AssessmentRecord) => {
    setActiveAssessment(assessment);
    if (assessment.status === 'completed') {
      setActiveScreen('completed_assessment');
    } else if (assessment.status === 'rejected') {
      setActiveScreen('validation');
    } else {
      setActiveScreen('decision_support');
    }
  };

  // Screen 3 Ingestion submit -> Screen 4 Validation
  const handleStartValidation = async (payload: {
    patientId: string;
    laterality: EyeLaterality;
    cameraModel: string;
    isMydriatic: boolean;
    clinicalNotes: string;
    imageDataUrl: string;
    fileSizeBytes: number;
    filename: string;
    simulateGateFailure?: 1 | 2 | 3 | null;
    candidateGrade?: number;
  }) => {
    try {
      const created = await clinicalApi.createAssessment(payload);
      setActiveAssessment(created);
      setActiveScreen('validation');
    } catch {
      showToast('Error uploading assessment image.');
    }
  };

  // Screen 4 Validation Complete
  const handleValidationComplete = (passed: boolean) => {
    if (passed && activeAssessment) {
      setActiveScreen('decision_support');
      showToast('All 3 validation gates passed. Decision-support generated.');
    } else {
      setActiveScreen('dashboard');
    }
  };

  // Screen 6 Review submitted -> Screen 7 Completed
  const handleReviewSubmitted = (updated: AssessmentRecord) => {
    setActiveAssessment(updated);
    setShowReviewModal(false);
    setActiveScreen('completed_assessment');
    showToast('Clinical certification signed and locked. Record is immutable.');
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-800 antialiased font-sans">
      {/* Header with TLS badge, 15-min timeout, and navigation tabs */}
      <Header
        currentUser={currentUser}
        onNavigate={handleNavigate}
        activeScreen={activeScreen}
        onLogout={handleLogout}
      />

      {/* Toast Notification Banner */}
      {toastMessage && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-5 right-5 z-50 bg-slate-900 text-teal-300 text-xs px-4 py-2.5 rounded-xl shadow-2xl border border-slate-700 flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2 duration-200"
        >
          <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping"></span>
          <span className="font-medium text-slate-100">{toastMessage}</span>
        </div>
      )}

      {/* Main Screen Router */}
      <main className="flex-1 w-full" role="main">
        {/* Screen 1: Sign-In */}
        {activeScreen === 'signin' && (
          <SignInScreen onSignInSuccess={handleSignInSuccess} />
        )}

        {/* Screen 2: Dashboard / Clinical Worklist */}
        {activeScreen === 'dashboard' && (
          <DashboardScreen
            onSelectAssessment={handleSelectAssessmentFromWorklist}
            onNewAssessment={() => setActiveScreen('new_assessment')}
            onNavigateHistory={() => setActiveScreen('history')}
          />
        )}

        {/* Screen 3: New Assessment Upload */}
        {activeScreen === 'new_assessment' && (
          <NewAssessmentScreen
            onStartValidation={handleStartValidation}
            onCancel={() => setActiveScreen('dashboard')}
          />
        )}

        {/* Screen 4: Real-Time 3-Stage Validation Stepper */}
        {activeScreen === 'validation' && activeAssessment && (
          <ValidationStepperScreen
            assessment={activeAssessment}
            onValidationComplete={handleValidationComplete}
            onRetry={() => setActiveScreen('new_assessment')}
          />
        )}

        {/* Screen 5: Decision-Support Result Workspace */}
        {activeScreen === 'decision_support' && activeAssessment && (
          <DecisionSupportScreen
            assessment={activeAssessment}
            onInitiateReview={() => setShowReviewModal(true)}
            onViewCompleted={() => setActiveScreen('completed_assessment')}
          />
        )}

        {/* Screen 7: Completed Assessment View */}
        {activeScreen === 'completed_assessment' && activeAssessment && (
          <CompletedAssessmentScreen
            assessment={activeAssessment}
            onBackToDashboard={() => setActiveScreen('dashboard')}
          />
        )}

        {/* Screen 8: Record History & Search */}
        {activeScreen === 'history' && (
          <RecordHistoryScreen
            onSelectAssessment={handleSelectAssessmentFromWorklist}
          />
        )}
      </main>

      {/* Screen 6: Professional Review Modal (Overlay on Screen 5) */}
      {showReviewModal && activeAssessment && currentUser && (
        <ProfessionalReviewModal
          assessment={activeAssessment}
          currentUser={currentUser}
          onClose={() => setShowReviewModal(false)}
          onReviewSubmitted={handleReviewSubmitted}
        />
      )}

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            Diabetic Retinopathy Clinical Decision Support System (DR-CDSS) © 2026 • Researcher: Onyekelu Chukwuebuka Elochukwu (2024516020FN)
          </span>
          <span className="font-mono text-[11px] text-slate-400">
            MSc Academic Research Prototype • EfficientNet-B0 Decision Support • WCAG 2.1 AA
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
