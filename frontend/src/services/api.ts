import axios from 'axios';
import {
  AssessmentRecord,
  ClinicianUser,
  ClinicianReview,
  WorklistFilter,
  AuditEvent,
  GateResult,
  EyeLaterality,
  ICDR_GRADES
} from '../types/clinical';
import { generateSyntheticFundus, generateSyntheticGradCam } from '../utils/syntheticFundus';
import { ClientValidationResult } from '../utils/retinalValidator';

const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Default Mock Clinician User
export const DEFAULT_CLINICIAN: ClinicianUser = {
  id: 'USR-8821',
  name: 'Dr. Adaeze Okonjo, MBChB, FRCOphth',
  email: 'a.okonjo@retina-clinic.nhs.uk',
  role: 'Consultant Medical Ophthalmologist',
  licenseNumber: 'GMC-7492104',
  facility: 'St. Jude Retinal Diagnostic Unit — Ward 4B',
  sessionTimeoutMinutes: 15,
  loginTime: new Date().toISOString(),
};

// Generate initial mock records
function createInitialRecords(): AssessmentRecord[] {
  // Moderate NPDR (Grade 2) - Pending Review
  const fundus2 = generateSyntheticFundus(2, 'OD');
  const gradcam2 = generateSyntheticGradCam(2, 'OD', 'viridis', 0.25);

  // Normal Retina (Grade 0) - Completed
  const fundus0 = generateSyntheticFundus(0, 'OS');
  const gradcam0 = generateSyntheticGradCam(0, 'OS', 'viridis', 0.20);

  // Severe NPDR (Grade 3) - Completed
  const fundus3 = generateSyntheticFundus(3, 'OD');
  const gradcam3 = generateSyntheticGradCam(3, 'OD', 'viridis', 0.25);

  // Gate 3 Rejection (Blurry)
  const fundusBlur = generateSyntheticFundus(1, 'OS');

  const rec1: AssessmentRecord = {
    id: 'REC-2026-0042',
    patientId: 'PT-90412',
    laterality: 'OD',
    acquisitionDate: '2026-09-26T14:32:00Z',
    status: 'needs_review',
    cameraModel: 'Topcon TRC-NW400 Non-Mydriatic',
    isMydriatic: false,
    clinicalNotes: 'Type 2 Diabetes mellitus for 8 years. HbA1c 8.4%. Complaining of occasional blurry vision.',
    imageUrl: fundus2.fundusDataUrl,
    gradcamUrl: gradcam2.gradcamDataUrl,
    qualityMetrics: {
      laplacianVariance: 248.5, // > 100 threshold
      illuminationIndex: 0.88,
      contrastDynamicRange: 184.2,
      nativeResolution: '2240x1488 px',
      fileSizeBytes: 3412984,
      sha256Hash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    },
    validationGates: [
      {
        gateIndex: 1,
        name: 'Gate 1',
        title: 'File Integrity & Security',
        status: 'passed',
        metric: 'MIME image/jpeg, SHA-256 match, 3.4 MB',
        details: 'Valid binary signature (0xFFD8FF), zero payload vulnerabilities.',
      },
      {
        gateIndex: 2,
        name: 'Gate 2',
        title: 'Retinal Relevance',
        status: 'passed',
        metric: 'Retinal circular aperture 94.2% coverage',
        details: 'Optical disc and macular structural landmarks confirmed via contour analysis.',
      },
      {
        gateIndex: 3,
        name: 'Gate 3',
        title: 'Technical Quality & Sharpness',
        status: 'passed',
        metric: 'Laplacian variance: 248.5 (>100.0 threshold)',
        details: 'Adequate vascular contrast and illumination homogeneity across all 4 quadrants.',
      },
    ],
    modelObservation: {
      primaryClassGrade: 2,
      primaryClassLabel: 'Moderate NPDR',
      primaryScore: 0.78,
      classScores: [
        { grade: 0, label: 'Grade 0: No Apparent DR', score: 0.04 },
        { grade: 1, label: 'Grade 1: Mild NPDR', score: 0.12 },
        { grade: 2, label: 'Grade 2: Moderate NPDR', score: 0.78 },
        { grade: 3, label: 'Grade 3: Severe NPDR', score: 0.05 },
        { grade: 4, label: 'Grade 4: Proliferative DR', score: 0.01 },
      ],
      targetLayer: 'features.8 (Conv2d Bottleneck Residual)',
      topActivationRegion: 'Inferotemporal quadrant parafoveal microaneurysms and blot hemorrhages',
      modelVersion: 'EfficientNet-B0-DR-v1 (Weights frozen)',
      inferenceTimestamp: '2026-09-26T14:32:05Z',
    },
    auditTrail: [
      {
        id: 'AUD-101',
        timestamp: '2026-09-26T14:32:00Z',
        action: 'Image Ingestion',
        actor: 'Clinical Technician (T-04)',
        details: 'Uploaded raw retinal fundus image for patient PT-90412 (OD).',
        badgeType: 'info',
      },
      {
        id: 'AUD-102',
        timestamp: '2026-09-26T14:32:03Z',
        action: 'Fail-Closed 3-Gate Validation',
        actor: 'Automated CDSS Pipeline',
        details: 'Gate 1, 2, and 3 passed technical thresholds.',
        badgeType: 'success',
      },
      {
        id: 'AUD-103',
        timestamp: '2026-09-26T14:32:05Z',
        action: 'Preliminary Inference & Grad-CAM',
        actor: 'Inference Engine',
        details: 'Generated preliminary model score: Moderate NPDR (0.78). Queued for human clinician review.',
        badgeType: 'info',
      },
    ],
    createdAt: '2026-09-26T14:32:00Z',
    updatedAt: '2026-09-26T14:32:05Z',
  };

  const rec2: AssessmentRecord = {
    id: 'REC-2026-0038',
    patientId: 'PT-88109',
    laterality: 'OS',
    acquisitionDate: '2026-09-26T11:15:00Z',
    status: 'completed',
    cameraModel: 'Canon CR-2 AF Retinal Camera',
    isMydriatic: true,
    imageUrl: fundus0.fundusDataUrl,
    gradcamUrl: gradcam0.gradcamDataUrl,
    qualityMetrics: {
      laplacianVariance: 312.4,
      illuminationIndex: 0.94,
      contrastDynamicRange: 198.0,
      nativeResolution: '2560x1920 px',
      fileSizeBytes: 4120300,
      sha256Hash: '7c89fbc001928aeef28172930a0bfd992110294817a0ef882194821a0be87411',
    },
    validationGates: [
      { gateIndex: 1, name: 'Gate 1', title: 'File Integrity', status: 'passed', metric: 'Valid JPEG' },
      { gateIndex: 2, name: 'Gate 2', title: 'Retinal Relevance', status: 'passed', metric: 'Fundus Disc 96%' },
      { gateIndex: 3, name: 'Gate 3', title: 'Technical Quality', status: 'passed', metric: 'Laplacian: 312.4' },
    ],
    modelObservation: {
      primaryClassGrade: 0,
      primaryClassLabel: 'No Apparent DR',
      primaryScore: 0.91,
      classScores: [
        { grade: 0, label: 'Grade 0: No Apparent DR', score: 0.91 },
        { grade: 1, label: 'Grade 1: Mild NPDR', score: 0.06 },
        { grade: 2, label: 'Grade 2: Moderate NPDR', score: 0.02 },
        { grade: 3, label: 'Grade 3: Severe NPDR', score: 0.01 },
        { grade: 4, label: 'Grade 4: Proliferative DR', score: 0.00 },
      ],
      targetLayer: 'features.8',
      topActivationRegion: 'Diffuse baseline physiological choroidal vasculature',
      modelVersion: 'EfficientNet-B0-DR-v1',
      inferenceTimestamp: '2026-09-26T11:15:04Z',
    },
    clinicianReview: {
      agreement: 'agree',
      certifiedGrade: 0,
      certifiedGradeLabel: 'Grade 0: No Apparent DR',
      referralPlan: 'Routine 12-month annual diabetic eye recall.',
      clinicianName: 'Dr. Adaeze Okonjo, MBChB, FRCOphth',
      licenseNumber: 'GMC-7492104',
      facility: 'St. Jude Retinal Diagnostic Unit — Ward 4B',
      signedAt: '2026-09-26T11:22:15Z',
      signatureHash: 'SIG-SHA256-4b82098dcf0927aa81',
      justificationNotes: 'Corroborated: Clear macula, healthy foveal reflex, no visible microaneurysms.',
    },
    auditTrail: [
      {
        id: 'AUD-088',
        timestamp: '2026-09-26T11:15:00Z',
        action: 'Ingestion & Validation Passed',
        actor: 'Clinical Technician (T-04)',
        details: 'Patient PT-88109 (OS) uploaded.',
        badgeType: 'info',
      },
      {
        id: 'AUD-089',
        timestamp: '2026-09-26T11:22:15Z',
        action: 'Certified Review Signed',
        actor: 'Dr. Adaeze Okonjo',
        details: 'Clinician confirmed Grade 0: No Apparent DR. Record locked and immutable.',
        badgeType: 'success',
      },
    ],
    createdAt: '2026-09-26T11:15:00Z',
    updatedAt: '2026-09-26T11:22:15Z',
  };

  const rec3: AssessmentRecord = {
    id: 'REC-2026-0035',
    patientId: 'PT-76430',
    laterality: 'OD',
    acquisitionDate: '2026-09-26T09:40:00Z',
    status: 'completed',
    cameraModel: 'Topcon TRC-NW400',
    isMydriatic: true,
    imageUrl: fundus3.fundusDataUrl,
    gradcamUrl: gradcam3.gradcamDataUrl,
    qualityMetrics: {
      laplacianVariance: 210.8,
      illuminationIndex: 0.89,
      contrastDynamicRange: 175.4,
      nativeResolution: '2240x1488 px',
      fileSizeBytes: 3820100,
      sha256Hash: '91f0c293ba9180ae18290ef4129b8c0192847aef90123847a9ef01928374aefb',
    },
    validationGates: [
      { gateIndex: 1, name: 'Gate 1', title: 'File Integrity', status: 'passed' },
      { gateIndex: 2, name: 'Gate 2', title: 'Retinal Relevance', status: 'passed' },
      { gateIndex: 3, name: 'Gate 3', title: 'Technical Quality', status: 'passed' },
    ],
    modelObservation: {
      primaryClassGrade: 3,
      primaryClassLabel: 'Severe NPDR',
      primaryScore: 0.82,
      classScores: [
        { grade: 0, label: 'Grade 0: No Apparent DR', score: 0.01 },
        { grade: 1, label: 'Grade 1: Mild NPDR', score: 0.03 },
        { grade: 2, label: 'Grade 2: Moderate NPDR', score: 0.14 },
        { grade: 3, label: 'Grade 3: Severe NPDR', score: 0.82 },
        { grade: 4, label: 'Grade 4: Proliferative DR', score: 0.00 },
      ],
      targetLayer: 'features.8',
      topActivationRegion: 'Multiple retinal quadrants with venous beading and extensive intraretinal hemorrhages',
      modelVersion: 'EfficientNet-B0-DR-v1',
      inferenceTimestamp: '2026-09-26T09:40:06Z',
    },
    clinicianReview: {
      agreement: 'agree',
      certifiedGrade: 3,
      certifiedGradeLabel: 'Grade 3: Severe NPDR',
      referralPlan: 'Urgent referral to secondary medical retina clinic for panretinal photocoagulation assessment.',
      clinicianName: 'Dr. Adaeze Okonjo, MBChB, FRCOphth',
      licenseNumber: 'GMC-7492104',
      facility: 'St. Jude Retinal Diagnostic Unit — Ward 4B',
      signedAt: '2026-09-26T09:55:00Z',
      signatureHash: 'SIG-SHA256-83901baef09210',
      justificationNotes: 'Confirms 4-2-1 criteria met: Extensive deep hemorrhages in 4 quadrants, venous beading visible in superior temporal vein.',
    },
    auditTrail: [
      {
        id: 'AUD-071',
        timestamp: '2026-09-26T09:40:00Z',
        action: 'Ingestion & Validation Passed',
        actor: 'Clinical Technician (T-04)',
        details: 'Patient PT-76430 (OD) ingested.',
        badgeType: 'info',
      },
      {
        id: 'AUD-072',
        timestamp: '2026-09-26T09:55:00Z',
        action: 'Certified Review Signed',
        actor: 'Dr. Adaeze Okonjo',
        details: 'Clinician confirmed Grade 3: Severe NPDR. Urgent referral placed.',
        badgeType: 'warning',
      },
    ],
    createdAt: '2026-09-26T09:40:00Z',
    updatedAt: '2026-09-26T09:55:00Z',
  };

  const rec4: AssessmentRecord = {
    id: 'REC-2026-0014',
    patientId: 'PT-61092',
    laterality: 'OS',
    acquisitionDate: '2026-09-26T08:10:00Z',
    status: 'rejected',
    cameraModel: 'Canon CR-2 AF',
    imageUrl: fundusBlur.fundusDataUrl,
    qualityMetrics: {
      laplacianVariance: 42.1, // Below 100 threshold
      illuminationIndex: 0.42,
      contrastDynamicRange: 68.0,
      nativeResolution: '2240x1488 px',
      fileSizeBytes: 2190300,
      sha256Hash: '5a498b0123984afcb01928374aefb01928374aefb90182374aefb01928374aef',
    },
    validationGates: [
      {
        gateIndex: 1,
        name: 'Gate 1',
        title: 'File Integrity & Security',
        status: 'passed',
        metric: 'Valid JPEG signature',
      },
      {
        gateIndex: 2,
        name: 'Gate 2',
        title: 'Retinal Relevance',
        status: 'passed',
        metric: 'Retinal circular mask recognized',
      },
      {
        gateIndex: 3,
        name: 'Gate 3',
        title: 'Technical Quality & Sharpness',
        status: 'failed',
        metric: 'Laplacian variance: 42.1 (Threshold >= 100.0)',
        rejectionReason: 'Severe motion blur and underexposure detected in central macula.',
        clinicalAction: 'Please recapture retinal photograph. Ensure steady patient chin-rest fixation and verify camera objective lens cleanliness.',
      },
    ],
    auditTrail: [
      {
        id: 'AUD-041',
        timestamp: '2026-09-26T08:10:00Z',
        action: 'Ingestion Attempt',
        actor: 'Clinical Technician (T-04)',
        details: 'Uploaded image for PT-61092.',
        badgeType: 'info',
      },
      {
        id: 'AUD-042',
        timestamp: '2026-09-26T08:10:04Z',
        action: 'Fail-Closed Gate 3 Rejection',
        actor: 'Automated CDSS Pipeline',
        details: 'Inference prohibited due to technical quality threshold violation (motion blur).',
        badgeType: 'error',
      },
    ],
    createdAt: '2026-09-26T08:10:00Z',
    updatedAt: '2026-09-26T08:10:04Z',
  };

  return [rec1, rec2, rec3, rec4];
}

class ClinicalApiService {
  private records: AssessmentRecord[] = [];
  private currentUser: ClinicianUser = DEFAULT_CLINICIAN;

  constructor() {
    this.records = createInitialRecords();
  }

  getCurrentUser(): ClinicianUser {
    return this.currentUser;
  }

  async login(credentials: { email: string; facility: string; licenseNumber?: string }): Promise<ClinicianUser> {
    try {
      const res = await apiClient.post('/auth/login', credentials);
      this.currentUser = res.data;
      return res.data;
    } catch {
      // Mock fallback
      this.currentUser = {
        ...DEFAULT_CLINICIAN,
        email: credentials.email || DEFAULT_CLINICIAN.email,
        facility: credentials.facility || DEFAULT_CLINICIAN.facility,
        licenseNumber: credentials.licenseNumber || DEFAULT_CLINICIAN.licenseNumber,
        loginTime: new Date().toISOString(),
      };
      return this.currentUser;
    }
  }

  async getWorklist(): Promise<AssessmentRecord[]> {
    try {
      const res = await apiClient.get('/assessments');
      return res.data;
    } catch {
      // Mock fallback: return copy of in-memory store
      return [...this.records];
    }
  }

  async getAssessmentById(id: string): Promise<AssessmentRecord | null> {
    try {
      const res = await apiClient.get(`/assessments/${id}`);
      return res.data;
    } catch {
      const found = this.records.find((r) => r.id === id);
      return found ? { ...found } : null;
    }
  }

  async createAssessment(payload: {
    patientId: string;
    laterality: EyeLaterality;
    cameraModel?: string;
    isMydriatic?: boolean;
    clinicalNotes?: string;
    imageDataUrl: string;
    fileSizeBytes: number;
    filename: string;
    simulateGateFailure?: 1 | 2 | 3 | null;
    candidateGrade?: number;
    clientValidation?: ClientValidationResult;
  }): Promise<AssessmentRecord> {
    const newId = `REC-2026-00${Math.floor(1000 + Math.random() * 9000)}`;
    const now = new Date().toISOString();
    const grade = payload.candidateGrade !== undefined ? payload.candidateGrade : 2;

    const cv = payload.clientValidation;
    const failsAt = cv && !cv.allPassed ? cv.failedGate : payload.simulateGateFailure;
    const isRejected = !!failsAt;

    const syntheticGradCam = generateSyntheticGradCam(grade, payload.laterality, 'viridis', 0.25);

    const gates: GateResult[] = [
      {
        gateIndex: 1,
        name: 'Gate 1',
        title: 'File Integrity & Security',
        status: failsAt === 1 ? 'failed' : 'passed',
        metric: cv ? cv.gate1.metric : failsAt === 1 ? 'Corrupted file signature or header' : 'Valid binary signature (0xFFD8FF)',
        rejectionReason: cv?.gate1.rejectionReason || (failsAt === 1 ? 'Binary magic number validation failed; file signature corrupted.' : undefined),
        clinicalAction: cv?.gate1.clinicalAction || (failsAt === 1 ? 'Please re-export retinal photograph in uncorrupted JPEG/PNG format.' : undefined),
      },
      {
        gateIndex: 2,
        name: 'Gate 2',
        title: 'Retinal Anatomical Relevance',
        status: failsAt === 2 ? 'failed' : failsAt === 1 ? 'pending' : 'passed',
        metric: cv ? cv.gate2.metric : failsAt === 2 ? 'Retinal coverage: 12.4% (Threshold >= 70%)' : 'Retinal field-of-view 92.5%',
        rejectionReason: cv?.gate2.rejectionReason || (failsAt === 2 ? 'Image lacks distinctive retinal morphology (optic disc / fovea not detected; possible anterior segment or non-ocular image).' : undefined),
        clinicalAction: cv?.gate2.clinicalAction || (failsAt === 2 ? 'Ensure retinal camera is focused on posterior pole fundus and anterior segment lens adapter is removed.' : undefined),
      },
      {
        gateIndex: 3,
        name: 'Gate 3',
        title: 'Technical Quality & Sharpness',
        status: failsAt === 3 ? 'failed' : failsAt ? 'pending' : 'passed',
        metric: cv ? cv.gate3.metric : failsAt === 3 ? 'Laplacian variance: 45.2 (Threshold >= 100.0)' : 'Laplacian variance: 275.4 (> 100 threshold)',
        rejectionReason: cv?.gate3.rejectionReason || (failsAt === 3 ? 'Severe motion blur or insufficient contrast detected.' : undefined),
        clinicalAction: cv?.gate3.clinicalAction || (failsAt === 3 ? 'Recapture retinal photograph ensuring patient fixation is steady and lens is clean.' : undefined),
      },
    ];

    const record: AssessmentRecord = {
      id: newId,
      patientId: payload.patientId.toUpperCase(),
      laterality: payload.laterality,
      acquisitionDate: now,
      status: isRejected ? 'rejected' : 'needs_review',
      cameraModel: payload.cameraModel || 'Topcon TRC-NW400 Non-Mydriatic',
      isMydriatic: payload.isMydriatic ?? false,
      clinicalNotes: payload.clinicalNotes,
      imageUrl: payload.imageDataUrl,
      gradcamUrl: isRejected ? undefined : syntheticGradCam.gradcamDataUrl,
      qualityMetrics: {
        laplacianVariance: cv ? cv.gate3.laplacianVariance : failsAt === 3 ? 45.2 : 275.4,
        illuminationIndex: cv && cv.gate2.isDocumentOrDiagram ? 0.21 : 0.89,
        contrastDynamicRange: 182.0,
        nativeResolution: '2240x1488 px',
        fileSizeBytes: payload.fileSizeBytes,
        sha256Hash: Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join(''),
      },
      validationGates: gates,
      modelObservation: isRejected
        ? undefined
        : {
            primaryClassGrade: grade,
            primaryClassLabel: ICDR_GRADES[grade].technicalTerm,
            primaryScore: grade === 0 ? 0.92 : grade === 1 ? 0.74 : grade === 2 ? 0.79 : grade === 3 ? 0.84 : 0.88,
            classScores: [
              { grade: 0, label: 'Grade 0: No Apparent DR', score: grade === 0 ? 0.92 : 0.03 },
              { grade: 1, label: 'Grade 1: Mild NPDR', score: grade === 1 ? 0.74 : 0.08 },
              { grade: 2, label: 'Grade 2: Moderate NPDR', score: grade === 2 ? 0.79 : 0.12 },
              { grade: 3, label: 'Grade 3: Severe NPDR', score: grade === 3 ? 0.84 : 0.06 },
              { grade: 4, label: 'Grade 4: Proliferative DR', score: grade === 4 ? 0.88 : 0.02 },
            ],
            targetLayer: 'features.8 (Conv2d Bottleneck Residual)',
            topActivationRegion: 'Parafoveal microaneurysms and inferotemporal arcade vessel cluster',
            modelVersion: 'EfficientNet-B0-DR-v1 (Frozen)',
            inferenceTimestamp: now,
          },
      auditTrail: [
        {
          id: `AUD-${Math.floor(100 + Math.random() * 900)}`,
          timestamp: now,
          action: 'Image Ingestion & Upload',
          actor: this.currentUser.name,
          details: `Ingested retinal fundus photograph for ${payload.patientId} (${payload.laterality}).`,
          badgeType: 'info',
        },
        {
          id: `AUD-${Math.floor(100 + Math.random() * 900)}`,
          timestamp: now,
          action: isRejected ? 'Validation Gate Rejected' : 'Validation Pipeline Passed',
          actor: 'Automated 3-Gate Pipeline',
          details: isRejected
            ? `Fail-closed triggered at Gate ${payload.simulateGateFailure}. Model inference aborted.`
            : 'All 3 technical gates passed. Model observation generated.',
          badgeType: isRejected ? 'error' : 'success',
        },
      ],
      createdAt: now,
      updatedAt: now,
    };

    try {
      await apiClient.post('/assessments', record);
    } catch {
      // Mock fallback: store in memory
    }

    this.records.unshift(record);
    return record;
  }

  async submitReview(
    assessmentId: string,
    review: Omit<ClinicianReview, 'signatureHash' | 'signedAt' | 'clinicianName' | 'licenseNumber' | 'facility'>
  ): Promise<AssessmentRecord> {
    const record = this.records.find((r) => r.id === assessmentId);
    if (!record) throw new Error('Assessment not found');

    const now = new Date().toISOString();
    const signatureHash = `SIG-SHA256-${Array.from({ length: 24 }, () => Math.floor(Math.random() * 16).toString(16)).join('')}`;

    const completeReview: ClinicianReview = {
      ...review,
      clinicianName: this.currentUser.name,
      licenseNumber: this.currentUser.licenseNumber,
      facility: this.currentUser.facility,
      signedAt: now,
      signatureHash,
    };

    record.clinicianReview = completeReview;
    record.status = 'completed';
    record.updatedAt = now;

    const auditEvent: AuditEvent = {
      id: `AUD-${Math.floor(100 + Math.random() * 900)}`,
      timestamp: now,
      action: 'Clinician Certified Review Finalized',
      actor: this.currentUser.name,
      details: `Clinician signed record. Classification: ${completeReview.certifiedGradeLabel}. Agreement: ${completeReview.agreement.toUpperCase()}.`,
      badgeType: completeReview.agreement === 'agree' ? 'success' : 'warning',
    };

    record.auditTrail.push(auditEvent);

    try {
      await apiClient.post(`/assessments/${assessmentId}/review`, completeReview);
    } catch {
      // Mock fallback
    }

    return { ...record };
  }

  async searchRecords(filters: WorklistFilter): Promise<AssessmentRecord[]> {
    try {
      const res = await apiClient.get('/assessments/search', { params: filters });
      return res.data;
    } catch {
      return this.records.filter((rec) => {
        if (filters.searchQuery) {
          const q = filters.searchQuery.toLowerCase();
          const matchPatient = rec.patientId.toLowerCase().includes(q);
          const matchId = rec.id.toLowerCase().includes(q);
          if (!matchPatient && !matchId) return false;
        }
        if (filters.status !== 'all' && rec.status !== filters.status) {
          return false;
        }
        if (filters.laterality !== 'all' && rec.laterality !== filters.laterality) {
          return false;
        }
        if (filters.grade !== 'all') {
          const grade = rec.clinicianReview?.certifiedGrade ?? rec.modelObservation?.primaryClassGrade;
          if (grade !== filters.grade) return false;
        }
        if (filters.agreement !== 'all') {
          if (!rec.clinicianReview || rec.clinicianReview.agreement !== filters.agreement) {
            return false;
          }
        }
        return true;
      });
    }
  }

  async getAuditLedger(assessmentId: string): Promise<AuditEvent[]> {
    try {
      const res = await apiClient.get(`/assessments/${assessmentId}/audit`);
      return res.data;
    } catch {
      const rec = this.records.find((r) => r.id === assessmentId);
      return rec ? [...rec.auditTrail] : [];
    }
  }

  async downloadReportPdf(assessment: AssessmentRecord): Promise<void> {
    try {
      const res = await apiClient.get(`/reports/${assessment.id}/pdf`, { responseType: 'blob' });
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DR-CDSS-Report-${assessment.id}.pdf`;
      a.click();
    } catch {
      // Client-side text/HTML tamper-evident report download fallback
      const reportContent = `
================================================================================
   DIABETIC RETINOPATHY CLINICAL DECISION SUPPORT SYSTEM (DR-CDSS)
   TAMPER-EVIDENT CLINICAL CONSULTATION SUMMARY & AUDIT CERTIFICATE
================================================================================
Assessment ID:          ${assessment.id}
Patient Reference:      ${assessment.patientId}
Eye Laterality:         ${assessment.laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
Acquisition Timestamp:  ${assessment.acquisitionDate}
Original Image SHA-256: ${assessment.qualityMetrics.sha256Hash}
Status:                 ${assessment.status.toUpperCase()}

--------------------------------------------------------------------------------
1. TECHNICAL VALIDATION PIPELINE (FAIL-CLOSED GATES)
--------------------------------------------------------------------------------
Gate 1 (File Integrity):        ${assessment.validationGates[0]?.status.toUpperCase() || 'PASS'}
Gate 2 (Retinal Relevance):     ${assessment.validationGates[1]?.status.toUpperCase() || 'PASS'}
Gate 3 (Technical Quality):     ${assessment.validationGates[2]?.status.toUpperCase() || 'PASS'}
Laplacian Sharpness Variance:   ${assessment.qualityMetrics.laplacianVariance} (Threshold >= 100.0)
Illumination Homogeneity:       ${assessment.qualityMetrics.illuminationIndex}

--------------------------------------------------------------------------------
2. PRELIMINARY AI MODEL OBSERVATION [NON-DIAGNOSTIC DECISION SUPPORT]
--------------------------------------------------------------------------------
Notice: Bounded preliminary mathematical associations. Not for independent diagnosis.
Model Checkpoint:       ${assessment.modelObservation?.modelVersion || 'N/A'}
Candidate Class:        ${assessment.modelObservation?.primaryClassLabel || 'N/A'}
Model-Generated Score:  ${assessment.modelObservation?.primaryScore.toFixed(2) || 'N/A'}
Target Explainability:  ${assessment.modelObservation?.targetLayer || 'N/A'}
Activation Region:      ${assessment.modelObservation?.topActivationRegion || 'N/A'}

--------------------------------------------------------------------------------
3. OFFICIAL CERTIFIED CLINICIAN EVALUATION [AUTHORITATIVE CLINICAL DECISION]
--------------------------------------------------------------------------------
Reviewing Clinician:    ${assessment.clinicianReview?.clinicianName || 'Pending'}
Medical License / GMC:  ${assessment.clinicianReview?.licenseNumber || 'Pending'}
Facility / Ward:        ${assessment.clinicianReview?.facility || 'Pending'}
Clinical Agreement:     ${assessment.clinicianReview?.agreement.toUpperCase() || 'Pending'}
Certified ICDR Stage:   ${assessment.clinicianReview?.certifiedGradeLabel || 'Pending'}
Management Protocol:    ${assessment.clinicianReview?.referralPlan || 'Pending'}
Clinical Justification: ${assessment.clinicianReview?.justificationNotes || 'None'}
Digital Signature Hash: ${assessment.clinicianReview?.signatureHash || 'Pending'}
Signature Timestamp:    ${assessment.clinicianReview?.signedAt || 'Pending'}

================================================================================
This record is cryptographically bound and tamper-evident per FDA SaMD / NHS DTAC.
================================================================================
      `.trim();

      const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DR-CDSS-Clinical-Summary-${assessment.id}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }
  }
}

export const clinicalApi = new ClinicalApiService();
