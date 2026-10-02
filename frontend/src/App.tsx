import React, { useState, useEffect } from 'react';
import { ClinicianUser, AssessmentRecord, EyeLaterality } from './types/clinical';
import { clinicalApi } from './services/api';
import { browserStorage } from './services/storage';
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
  // Authentication State with persistent storage
  const [currentUser, setCurrentUser] = useState<ClinicianUser | null>(() => {
    return browserStorage.getStoredUser();
  });

  // Active Screen and Assessment State with persistent restoration
  const [activeScreen, setActiveScreen] = useState<ScreenState>(() => {
    const session = browserStorage.getSessionState();
    return (session.activeScreen as ScreenState) || 'dashboard';
  });

  const [activeAssessment, setActiveAssessment] = useState<AssessmentRecord | null>(null);

  // Review Modal Visibility (Screen 6)
  const [showReviewModal, setShowReviewModal] = useState<boolean>(false);

  // Toast notification
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // Helper for synchronized persistent navigation
  const navigateTo = (screen: ScreenState, assessment?: AssessmentRecord | null) => {
    setActiveScreen(screen);
    if (assessment !== undefined) {
      setActiveAssessment(assessment);
    }
    const currentAss = assessment !== undefined ? assessment : activeAssessment;
    const assId = currentAss?.id || null;
    browserStorage.saveSessionState(screen, assId);

    const hashSuffix = (screen === 'decision_support' || screen === 'completed_assessment' || screen === 'validation') && assId
      ? `/${assId}`
      : '';
    window.location.hash = `#${screen}${hashSuffix}`;
  };

  // Synchronize URL Hash and persist across page reloads
  useEffect(() => {
    const handleHash = async () => {
      const rawHash = window.location.hash.replace('#', '').trim();
      const records = await clinicalApi.getWorklist();

      if (rawHash === 'signin') {
        setCurrentUser(null);
        browserStorage.saveStoredUser(null);
        setActiveScreen('signin');
        setShowReviewModal(false);
        return;
      }

      // Check stored clinician session
      const storedUser = browserStorage.getStoredUser();
      if (storedUser) {
        setCurrentUser(storedUser);
      } else if (!currentUser) {
        setCurrentUser(null);
        setActiveScreen('signin');
        return;
      }

      const session = browserStorage.getSessionState();
      const [hashScreen, hashRecordId] = rawHash.split('/');

      let targetScreen: string = hashScreen || session.activeScreen || 'dashboard';
      if (targetScreen === 'completed') targetScreen = 'completed_assessment';

      // Look up target assessment
      const targetRecordId = hashRecordId || session.activeAssessmentId;
      let targetRecord: AssessmentRecord | null = null;
      if (targetRecordId) {
        targetRecord = await clinicalApi.getAssessmentById(targetRecordId);
      }

      const pendingRecord = records.find(r => r.status === 'needs_review') || records[0];
      const completedRecord = records.find(r => r.status === 'completed') || records[1];
      const rejectedRecord = records.find(r => r.status === 'rejected') || records[3] || records[0];

      if (targetScreen === 'dashboard') {
        setActiveScreen('dashboard');
        setShowReviewModal(false);
      } else if (targetScreen === 'new_assessment') {
        setActiveScreen('new_assessment');
        setShowReviewModal(false);
      } else if (targetScreen === 'validation') {
        const ass = targetRecord || pendingRecord;
        setActiveAssessment(ass);
        setActiveScreen('validation');
        setShowReviewModal(false);
      } else if (targetScreen === 'rejected') {
        const ass = targetRecord || rejectedRecord;
        setActiveAssessment(ass);
        setActiveScreen('validation');
        setShowReviewModal(false);
      } else if (targetScreen === 'decision_support') {
        const ass = targetRecord || pendingRecord;
        setActiveAssessment(ass);
        setActiveScreen('decision_support');
        setShowReviewModal(false);
      } else if (targetScreen === 'review') {
        const ass = targetRecord || pendingRecord;
        setActiveAssessment(ass);
        setActiveScreen('decision_support');
        setShowReviewModal(true);
      } else if (targetScreen === 'completed_assessment') {
        const ass = targetRecord || completedRecord;
        setActiveAssessment(ass);
        setActiveScreen('completed_assessment');
        setShowReviewModal(false);
      } else if (targetScreen === 'history') {
        setActiveScreen('history');
        setShowReviewModal(false);
      } else {
        setActiveScreen('dashboard');
        setShowReviewModal(false);
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
        navigateTo('dashboard');
      }
    };
    window.addEventListener('keydown', handleGlobalKeys);
    return () => window.removeEventListener('keydown', handleGlobalKeys);
  }, [showReviewModal]);

  // Screen 1: Sign in handler
  const handleSignInSuccess = (user: ClinicianUser) => {
    setCurrentUser(user);
    browserStorage.saveStoredUser(user);
    navigateTo('dashboard', null);
    showToast(`Welcome back, ${user.name}`);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setActiveAssessment(null);
    browserStorage.saveStoredUser(null);
    browserStorage.saveSessionState('signin', null);
    window.location.hash = '#signin';
    setActiveScreen('signin');
    setShowReviewModal(false);
    showToast('Signed out of clinical session.');
  };

  // Navigation from Header
  const handleNavigate = (screen: 'dashboard' | 'new_assessment' | 'history' | 'signin') => {
    navigateTo(screen);
  };

  // Screen 2 Worklist select assessment
  const handleSelectAssessmentFromWorklist = (assessment: AssessmentRecord) => {
    setActiveAssessment(assessment);
    if (assessment.status === 'completed') {
      navigateTo('completed_assessment', assessment);
    } else if (assessment.status === 'rejected') {
      navigateTo('validation', assessment);
    } else {
      navigateTo('decision_support', assessment);
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
    clientValidation?: any;
  }) => {
    try {
      const created = await clinicalApi.createAssessment(payload);
      navigateTo('validation', created);
    } catch {
      showToast('Error uploading assessment image.');
    }
  };

  // Screen 4 Validation Complete
  const handleValidationComplete = (passed: boolean) => {
    if (passed && activeAssessment) {
      navigateTo('decision_support', activeAssessment);
      showToast('All 3 validation gates passed. Decision-support generated.');
    } else {
      navigateTo('dashboard', null);
    }
  };

  // Screen 6 Review submitted -> Screen 7 Completed
  const handleReviewSubmitted = (updated: AssessmentRecord) => {
    setActiveAssessment(updated);
    setShowReviewModal(false);
    navigateTo('completed_assessment', updated);
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
            PGD Academic Research Prototype • Department of Computer Science • Faculty of Physical Sciences
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
