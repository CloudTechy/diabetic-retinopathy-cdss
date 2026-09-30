import { AssessmentRecord, ClinicianUser } from '../types/clinical';

const DB_NAME = 'dr_cdss_clinical_db';
const DB_VERSION = 1;
const STORE_ASSESSMENTS = 'assessments';
const STORE_METADATA = 'metadata';

const LOCAL_STORAGE_KEY_ASSESSMENTS = 'dr_cdss_assessments_cache';
const LOCAL_STORAGE_KEY_USER = 'dr_cdss_current_user';
const LOCAL_STORAGE_KEY_SCREEN = 'dr_cdss_active_screen';
const LOCAL_STORAGE_KEY_ASSESSMENT_ID = 'dr_cdss_active_assessment_id';

class BrowserStorageService {
  private dbPromise: Promise<IDBDatabase> | null = null;
  private isIndexedDBAvailable: boolean = typeof window !== 'undefined' && 'indexedDB' in window;

  private openDB(): Promise<IDBDatabase> {
    if (!this.isIndexedDBAvailable) {
      return Promise.reject(new Error('IndexedDB not supported'));
    }
    if (this.dbPromise) {
      return this.dbPromise;
    }

    this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      try {
        const request = window.indexedDB.open(DB_NAME, DB_VERSION);

        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains(STORE_ASSESSMENTS)) {
            db.createObjectStore(STORE_ASSESSMENTS, { keyPath: 'id' });
          }
          if (!db.objectStoreNames.contains(STORE_METADATA)) {
            db.createObjectStore(STORE_METADATA, { keyPath: 'key' });
          }
        };

        request.onsuccess = () => {
          resolve(request.result);
        };

        request.onerror = () => {
          reject(request.error || new Error('Failed to open IndexedDB'));
        };
      } catch (err) {
        reject(err);
      }
    });

    return this.dbPromise;
  }

  // Load all assessments from persistent storage (IndexedDB -> LocalStorage fallback)
  async loadAssessments(defaultRecords: AssessmentRecord[]): Promise<AssessmentRecord[]> {
    try {
      const db = await this.openDB();
      return new Promise<AssessmentRecord[]>((resolve) => {
        const tx = db.transaction(STORE_ASSESSMENTS, 'readonly');
        const store = tx.objectStore(STORE_ASSESSMENTS);
        const req = store.getAll();

        req.onsuccess = () => {
          const list = req.result as AssessmentRecord[];
          if (list && list.length > 0) {
            // Sort by acquisitionDate or createdAt descending
            list.sort((a, b) => new Date(b.createdAt || b.acquisitionDate).getTime() - new Date(a.createdAt || a.acquisitionDate).getTime());
            resolve(list);
          } else {
            // Initialize with default records
            this.saveAllAssessments(defaultRecords).catch(console.error);
            resolve([...defaultRecords]);
          }
        };

        req.onerror = () => {
          resolve(this.loadFromLocalStorage(defaultRecords));
        };
      });
    } catch {
      return this.loadFromLocalStorage(defaultRecords);
    }
  }

  // Save a single assessment to IndexedDB & fallback
  async saveAssessment(record: AssessmentRecord): Promise<void> {
    try {
      const db = await this.openDB();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_ASSESSMENTS, 'readwrite');
        const store = tx.objectStore(STORE_ASSESSMENTS);
        const req = store.put(record);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    } catch {
      // Fallback
    }

    // Also attempt lightweight summary or full fallback in localStorage if quota allows
    try {
      const existing = this.loadFromLocalStorage([]);
      const index = existing.findIndex((r) => r.id === record.id);
      if (index >= 0) {
        existing[index] = record;
      } else {
        existing.unshift(record);
      }
      localStorage.setItem(LOCAL_STORAGE_KEY_ASSESSMENTS, JSON.stringify(existing));
    } catch {
      // Ignore quota exceeded errors gracefully for large base64 fundus images
    }
  }

  // Save multiple assessments
  async saveAllAssessments(records: AssessmentRecord[]): Promise<void> {
    try {
      const db = await this.openDB();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_ASSESSMENTS, 'readwrite');
        const store = tx.objectStore(STORE_ASSESSMENTS);
        for (const rec of records) {
          store.put(rec);
        }
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Fallback
    }

    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_ASSESSMENTS, JSON.stringify(records));
    } catch {
      // Ignore quota error
    }
  }

  // Reset database back to default initial records
  async resetToDefaults(defaultRecords: AssessmentRecord[]): Promise<void> {
    try {
      const db = await this.openDB();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_ASSESSMENTS, 'readwrite');
        const store = tx.objectStore(STORE_ASSESSMENTS);
        store.clear();
        for (const rec of defaultRecords) {
          store.put(rec);
        }
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Fallback
    }

    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_ASSESSMENTS, JSON.stringify(defaultRecords));
    } catch {
      // Ignore quota error
    }
  }

  // Fallback to localStorage
  private loadFromLocalStorage(defaultRecords: AssessmentRecord[]): AssessmentRecord[] {
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY_ASSESSMENTS);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch {
      // ignore JSON or access error
    }
    return [...defaultRecords];
  }

  // Clinician User State
  getStoredUser(): ClinicianUser | null {
    try {
      const raw = localStorage.getItem(LOCAL_STORAGE_KEY_USER);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  saveStoredUser(user: ClinicianUser | null): void {
    try {
      if (user) {
        localStorage.setItem(LOCAL_STORAGE_KEY_USER, JSON.stringify(user));
      } else {
        localStorage.removeItem(LOCAL_STORAGE_KEY_USER);
      }
    } catch {
      // ignore
    }
  }

  // Active Session Navigation State
  getSessionState(): { activeScreen: string | null; activeAssessmentId: string | null } {
    try {
      return {
        activeScreen: localStorage.getItem(LOCAL_STORAGE_KEY_SCREEN),
        activeAssessmentId: localStorage.getItem(LOCAL_STORAGE_KEY_ASSESSMENT_ID),
      };
    } catch {
      return { activeScreen: null, activeAssessmentId: null };
    }
  }

  saveSessionState(screen: string, assessmentId?: string | null): void {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_SCREEN, screen);
      if (assessmentId) {
        localStorage.setItem(LOCAL_STORAGE_KEY_ASSESSMENT_ID, assessmentId);
      } else if (assessmentId === null) {
        localStorage.removeItem(LOCAL_STORAGE_KEY_ASSESSMENT_ID);
      }
    } catch {
      // ignore
    }
  }
}

export const browserStorage = new BrowserStorageService();
