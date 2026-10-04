export type EyeLaterality = 'OD' | 'OS';

export type ReviewStatus = 'validating' | 'needs_review' | 'completed' | 'rejected';

export type GateStatus = 'pending' | 'in_progress' | 'passed' | 'failed';

export interface GateResult {
  name: string;
  gateIndex: 1 | 2 | 3;
  status: GateStatus;
  title: string;
  metric?: string;
  details?: string;
  rejectionReason?: string;
  clinicalAction?: string;
}

export type GateResult2 = GateResult;

export type AgreementType = 'agree' | 'disagree' | 'inconclusive';

export interface ICDRGradeInfo {
  grade: number;
  label: string;
  technicalTerm: string;
  description: string;
  color: string;
  badgeClass: string;
}

export const ICDR_GRADES: Record<number, ICDRGradeInfo> = {
  0: {
    grade: 0,
    label: 'Grade 0: No Apparent DR',
    technicalTerm: 'No Apparent Retinopathy',
    description: 'No microaneurysms, hemorrhages, or retinal lesions detected.',
    color: '#10B981',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300'
  },
  1: {
    grade: 1,
    label: 'Grade 1: Mild NPDR',
    technicalTerm: 'Mild Non-Proliferative Retinopathy',
    description: 'Microaneurysms only. Subtle vascular focal changes.',
    color: '#06B6D4',
    badgeClass: 'bg-cyan-100 text-cyan-800 border-cyan-300'
  },
  2: {
    grade: 2,
    label: 'Grade 2: Moderate NPDR',
    technicalTerm: 'Moderate Non-Proliferative Retinopathy',
    description: 'More than microaneurysms but less than Severe NPDR (dot-and-blot hemorrhages, hard exudates).',
    color: '#F59E0B',
    badgeClass: 'bg-amber-100 text-amber-800 border-amber-300'
  },
  3: {
    grade: 3,
    label: 'Grade 3: Severe NPDR',
    technicalTerm: 'Severe Non-Proliferative Retinopathy',
    description: 'Meets 4-2-1 rule: >=20 hemorrhages in 4 quadrants, venous beading in >=2 quadrants, or IRMA in >=1 quadrant.',
    color: '#F97316',
    badgeClass: 'bg-orange-100 text-orange-800 border-orange-300'
  },
  4: {
    grade: 4,
    label: 'Grade 4: Proliferative DR',
    technicalTerm: 'Proliferative Diabetic Retinopathy',
    description: 'Neovascularization of the disc/retina, preretinal or vitreous hemorrhage.',
    color: '#EF4444',
    badgeClass: 'bg-rose-100 text-rose-800 border-rose-300'
  }
};

export interface ModelObservation {
  primaryClassGrade: number;
  primaryClassLabel: string;
  primaryScore: number;
  classScores: {
    grade: number;
    label: string;
    score: number;
  }[];
  targetLayer: string;
  topActivationRegion: string;
  modelVersion: string;
  inferenceTimestamp: string;
}

export interface TechnicalQualityMetrics {
  laplacianVariance: number;
  illuminationIndex: number;
  contrastDynamicRange: number;
  nativeResolution: string;
  fileSizeBytes: number;
  sha256Hash: string;
}

export interface ClinicianReview {
  agreement: AgreementType;
  reviewerAssessedGrade: number;
  reviewerAssessedGradeLabel: string;
  justificationNotes?: string;
  inconclusiveReason?: string;
  clinicianName: string;
  licenseNumber: string;
  facility: string;
  signedAt: string;
  signatureHash: string;
}

export interface AuditEvent {
  id: string;
  timestamp: string;
  action: string;
  actor: string;
  details: string;
  badgeType: 'info' | 'success' | 'warning' | 'error';
}

export interface AssessmentRecord {
  id: string;
  patientId: string;
  laterality: EyeLaterality;
  acquisitionDate: string;   // the record's creation time, not an image-acquisition time (name is historical)
  status: ReviewStatus;
  cameraModel?: string;
  isMydriatic?: boolean | null;   // null = not recorded
  clinicalNotes?: string;
  imageUrl: string;
  gradcamUrl?: string;
  qualityMetrics: TechnicalQualityMetrics;
  validationGates: GateResult2[];
  modelObservation?: ModelObservation;
  clinicianReview?: ClinicianReview;
  auditTrail: AuditEvent[];
  createdAt: string;
  updatedAt: string;
}

export interface ClinicianUser {
  id: string;
  name: string;
  email: string;
  role: string;
  licenseNumber: string;
  facility: string;
  sessionTimeoutMinutes: number;
  loginTime: string;
}

export interface WorklistFilter {
  searchQuery: string;
  status: 'all' | ReviewStatus;
  laterality: 'all' | EyeLaterality;
  grade: 'all' | number;
  agreement: 'all' | AgreementType;
  dateFrom?: string;
  dateTo?: string;
}

