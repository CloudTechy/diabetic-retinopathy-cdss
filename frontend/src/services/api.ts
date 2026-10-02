import axios from 'axios';
import {
  AssessmentRecord,
  ClinicianUser,
  ClinicianReview,
  WorklistFilter,
  AuditEvent,
  EyeLaterality,
} from '../types/clinical';
import { ClientValidationResult } from '../utils/retinalValidator';
import { browserStorage } from './storage';

const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 5000,
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

class ClinicalApiService {
  private currentUser: ClinicianUser | null = null;

  getCurrentUser(): ClinicianUser | null {
    if (this.currentUser) return this.currentUser;
    const stored = browserStorage.getStoredUser();
    if (stored) {
      this.currentUser = stored;
      return stored;
    }
    return null;
  }

  async login(credentials: { username: string; password?: string }): Promise<ClinicianUser> {
    const params = new URLSearchParams();
    params.append('username', credentials.username);
    if (credentials.password) {
      params.append('password', credentials.password);
    }

    const res = await apiClient.post('/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    const token = res.data.access_token;
    if (token) {
      localStorage.setItem('access_token', token);
    }

    this.currentUser = {
      id: res.data.id || 'Unknown',
      name: res.data.name || credentials.username,
      email: res.data.email || credentials.username,
      role: res.data.role || 'Clinician',
      licenseNumber: res.data.licenseNumber || '',
      facility: res.data.facility || '',
      sessionTimeoutMinutes: 15,
      loginTime: new Date().toISOString(),
      ...res.data.user
    };

    browserStorage.saveStoredUser(this.currentUser);
    return this.currentUser!;
  }

  async logout(): Promise<void> {
    this.currentUser = null;
    browserStorage.saveStoredUser(null);
    localStorage.removeItem('access_token');
    sessionStorage.removeItem('access_token');
  }

  async getWorklist(): Promise<AssessmentRecord[]> {
    const res = await apiClient.get('/assessments');
    return res.data;
  }

  async getAssessmentById(id: string): Promise<AssessmentRecord | null> {
    const res = await apiClient.get(`/assessments/${id}`);
    return res.data;
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
    clientValidation?: ClientValidationResult;
  }): Promise<AssessmentRecord> {
    const res = await apiClient.post('/assessments', payload);
    return res.data;
  }

  async submitReview(
    assessmentId: string,
    review: Omit<ClinicianReview, 'signatureHash' | 'signedAt' | 'clinicianName' | 'licenseNumber' | 'facility'>
  ): Promise<AssessmentRecord> {
    const res = await apiClient.post(`/assessments/${assessmentId}/review`, review);
    return res.data;
  }

  async searchRecords(filters: WorklistFilter): Promise<AssessmentRecord[]> {
    const res = await apiClient.get('/assessments/search', { params: filters });
    return res.data;
  }

  async getAuditLedger(assessmentId: string): Promise<AuditEvent[]> {
    const res = await apiClient.get(`/assessments/${assessmentId}/audit`);
    return res.data;
  }

  async downloadReportPdf(assessment: AssessmentRecord): Promise<void> {
    const res = await apiClient.get(`/reports/${assessment.id}/pdf`, { responseType: 'blob' });
    const blob = new Blob([res.data], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DR-CDSS-Report-${assessment.id}.pdf`;
    a.click();
    window.URL.revokeObjectURL(url);
  }
}

export const clinicalApi = new ClinicalApiService();
