import { describe, expect, it } from 'vitest';
import {
  accountStatusMessage,
  passwordProblem,
  routeAfterSignIn,
  screenAllowed,
  validateRegistration,
} from './session';

describe('routeAfterSignIn', () => {
  it('sends an account with a temporary password to the password change before anything else', () => {
    expect(routeAfterSignIn({ role: 'SUPERADMIN', mustChangePassword: true })).toBe('change_password');
    expect(routeAfterSignIn({ role: 'PHYSICIAN', mustChangePassword: true })).toBe('change_password');
  });
  it('sends a Super Administrator to the console and a physician to the dashboard', () => {
    expect(routeAfterSignIn({ role: 'SUPERADMIN', mustChangePassword: false })).toBe('admin');
    expect(routeAfterSignIn({ role: 'PHYSICIAN', mustChangePassword: false })).toBe('dashboard');
  });
});

describe('screenAllowed', () => {
  it('lets a Super Administrator open the console only', () => {
    const admin = { role: 'SUPERADMIN', mustChangePassword: false };
    expect(screenAllowed(admin, 'admin')).toBe(true);
    expect(screenAllowed(admin, 'dashboard')).toBe(false);
    expect(screenAllowed(admin, 'new_assessment')).toBe(false);
  });
  it('keeps a physician out of the console', () => {
    const physician = { role: 'PHYSICIAN', mustChangePassword: false };
    expect(screenAllowed(physician, 'admin')).toBe(false);
    expect(screenAllowed(physician, 'dashboard')).toBe(true);
  });
  it('allows nothing but the password change while the password is temporary', () => {
    const fresh = { role: 'PHYSICIAN', mustChangePassword: true };
    expect(screenAllowed(fresh, 'dashboard')).toBe(false);
    expect(screenAllowed(fresh, 'change_password')).toBe(true);
  });
});

describe('accountStatusMessage', () => {
  it('names each non-active state', () => {
    expect(accountStatusMessage('PENDING_APPROVAL')).toMatch(/awaiting administrative approval/);
    expect(accountStatusMessage('REJECTED')).toMatch(/not approved/);
    expect(accountStatusMessage('SUSPENDED')).toMatch(/suspended/);
  });
});

describe('passwordProblem', () => {
  it('applies the server policy: ten characters, letters and digits', () => {
    expect(passwordProblem('short1')).toMatch(/at least 10/);
    expect(passwordProblem('onlylettershere')).toMatch(/letters and digits/);
    expect(passwordProblem('1234567890123')).toMatch(/letters and digits/);
    expect(passwordProblem('Good-Pass-2026')).toBeNull();
  });
});

describe('validateRegistration', () => {
  const valid = {
    fullName: 'Ada Lovelace', username: 'ada.lovelace', email: 'ada@example.org',
    password: 'Strong-Pass-2026', passwordConfirmation: 'Strong-Pass-2026',
  };
  it('accepts a complete form', () => {
    expect(validateRegistration(valid)).toEqual({});
  });
  it('reports each defective field by name', () => {
    const errors = validateRegistration({
      fullName: 'A', username: 'a b', email: 'not-an-email', password: 'weak', passwordConfirmation: 'other',
    });
    expect(Object.keys(errors).sort()).toEqual(['email', 'fullName', 'password', 'passwordConfirmation', 'username']);
  });
  it('has no field through which a role or status could be chosen', () => {
    expect(Object.keys(valid)).not.toContain('role');
    expect(Object.keys(valid)).not.toContain('status');
  });
});
