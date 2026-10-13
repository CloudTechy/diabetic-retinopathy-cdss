import axios from 'axios';
import {
  AccountSummary,
  AdminAuditEvent,
  CreatedPhysician,
  UserAccount,
  AssessmentRecord,
  ClinicianUser,
  ClinicianReview,
  WorklistFilter,
  AuditEvent,
  EyeLaterality,
} from '../types/clinical';
import { ClientValidationResult } from '../utils/retinalValidator';
import { browserStorage } from './storage';

// Web builds call the relative '/api/v1', which the dev server and the
// deployment reverse proxy forward to the backend. A native (Capacitor) build
// must be given its backend explicitly through VITE_API_BASE_URL at build
// time; no backend host is hard-coded here, because this repository makes no
// claim about any hosted instance.
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      sessionStorage.removeItem('access_token');
      browserStorage.saveStoredUser(null);
      if (window.location.hash !== '#signin') {
        window.location.hash = '#signin';
      }
    }
    return Promise.reject(error);
  }
);

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
    const res = await apiClient.post('/auth/login', {
      username: credentials.username,
      password: credentials.password || '',
    });

    const token = res.data.access_token;
    if (token) {
      localStorage.setItem('access_token', token);
    }

    // the account exactly as the server describes it; nothing is invented
    const u = res.data.user || {};
    this.currentUser = {
      id: u.id,
      username: u.username || credentials.username,
      name: u.name || credentials.username,
      email: u.email || '',
      role: u.role ?? '',
      status: u.status ?? '',
      mustChangePassword: Boolean(u.mustChangePassword),
      licenseNumber: u.licenseNumber || '',
      facility: u.facility || '',
      sessionTimeoutMinutes: u.sessionTimeoutMinutes || 15,
      loginTime: u.loginTime || new Date().toISOString(),
    };

    browserStorage.saveStoredUser(this.currentUser);
    return this.currentUser!;
  }

  async logout(): Promise<void> {
    // invalidate the session on the server first; local state is cleared either way
    try {
      await apiClient.post('/auth/logout');
    } catch {
      /* the token may already be invalid; nothing to do */
    }
    this.currentUser = null;
    browserStorage.saveStoredUser(null);
    localStorage.removeItem('access_token');
    sessionStorage.removeItem('access_token');
  }

  async me(): Promise<ClinicianUser> {
    const res = await apiClient.get('/auth/me');
    const u = res.data;
    const user: ClinicianUser = {
      id: u.id, username: u.username, name: u.name, email: u.email, role: u.role, status: u.status,
      mustChangePassword: Boolean(u.mustChangePassword), licenseNumber: u.licenseNumber || '', facility: u.facility || '',
      sessionTimeoutMinutes: u.sessionTimeoutMinutes || 15, loginTime: u.loginTime,
    };
    this.currentUser = user;
    browserStorage.saveStoredUser(user);
    return user;
  }

  async register(payload: {
    fullName: string; username: string; email: string; password: string; passwordConfirmation: string;
    licenseNumber?: string; facility?: string;
  }): Promise<{ id: string; username: string; status: string; message: string }> {
    const res = await apiClient.post('/auth/register', payload);
    return res.data;
  }

  async changePassword(payload: { currentPassword: string; newPassword: string; newPasswordConfirmation: string }): Promise<void> {
    await apiClient.post('/auth/change-password', payload);
  }

  // ---- Super Administrator console
  async adminSummary(): Promise<AccountSummary> {
    return (await apiClient.get('/admin/summary')).data;
  }

  async adminListUsers(params: { q?: string; role?: string; status?: string } = {}): Promise<UserAccount[]> {
    return (await apiClient.get('/admin/users', { params })).data;
  }

  async adminPendingUsers(): Promise<UserAccount[]> {
    return (await apiClient.get('/admin/users/pending')).data;
  }

  async adminCreatePhysician(payload: {
    fullName: string; username: string; email: string; licenseNumber?: string; facility?: string;
    temporaryPassword?: string; activateImmediately: boolean;
  }): Promise<CreatedPhysician> {
    return (await apiClient.post('/admin/users', payload)).data;
  }

  async adminCreatePhysiciansBatch(payload: { count: number; usernamePrefix: string; facility?: string }): Promise<{ created: CreatedPhysician[] }> {
    return (await apiClient.post('/admin/users/batch', payload)).data;
  }

  async adminApprove(userId: string): Promise<UserAccount> {
    return (await apiClient.patch(`/admin/users/${userId}/approve`)).data;
  }

  async adminReject(userId: string, reason?: string): Promise<UserAccount> {
    return (await apiClient.patch(`/admin/users/${userId}/reject`, { reason })).data;
  }

  async adminSuspend(userId: string, reason?: string): Promise<UserAccount> {
    return (await apiClient.patch(`/admin/users/${userId}/suspend`, { reason })).data;
  }

  async adminReactivate(userId: string): Promise<UserAccount> {
    return (await apiClient.patch(`/admin/users/${userId}/reactivate`)).data;
  }

  async adminResetPassword(userId: string): Promise<CreatedPhysician> {
    return (await apiClient.post(`/admin/users/${userId}/reset-password`)).data;
  }

  async adminAuditEvents(params: { userId?: string; eventType?: string; limit?: number } = {}): Promise<AdminAuditEvent[]> {
    return (await apiClient.get('/admin/audit-events', { params })).data;
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
    isMydriatic?: boolean | null;
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
