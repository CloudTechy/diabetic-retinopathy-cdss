/**
 * Pure helpers for sign-in routing, account-state messages and registration
 * form validation. They hold no state and touch no browser API, so they are
 * unit-tested directly (src/utils/session.test.ts).
 */
export type AccountRole = 'SUPERADMIN' | 'PHYSICIAN';
export type AccountStatus = 'PENDING_APPROVAL' | 'ACTIVE' | 'REJECTED' | 'SUSPENDED';

export interface SessionUserLike {
  role: string;
  mustChangePassword?: boolean;
}

export type LandingScreen = 'change_password' | 'admin' | 'dashboard';

/** Where a freshly signed-in account goes. A temporary password is dealt with before anything else. */
export function routeAfterSignIn(user: SessionUserLike): LandingScreen {
  if (user.mustChangePassword) return 'change_password';
  if (user.role === 'SUPERADMIN') return 'admin';
  return 'dashboard';
}

/** Screens a signed-in account may open. A Super Administrator administers accounts and sees no clinical screen. */
export function screenAllowed(user: SessionUserLike, screen: string): boolean {
  if (user.mustChangePassword) return screen === 'change_password';
  if (user.role === 'SUPERADMIN') return screen === 'admin' || screen === 'change_password';
  return screen !== 'admin';
}

export function accountStatusMessage(status: AccountStatus | string): string {
  switch (status) {
    case 'PENDING_APPROVAL':
      return 'This account is awaiting administrative approval. You will be able to sign in once a Super Administrator approves it.';
    case 'REJECTED':
      return 'This registration was not approved. Contact the administrator if you believe this is an error.';
    case 'SUSPENDED':
      return 'This account is suspended. Contact the administrator to have it reactivated.';
    case 'ACTIVE':
      return 'This account is active.';
    default:
      return 'This account cannot sign in.';
  }
}

export const PASSWORD_MIN_LENGTH = 10;

/** The same rule the server applies (app/services/account_service.check_password_policy). */
export function passwordProblem(password: string): string | null {
  if (password.length < PASSWORD_MIN_LENGTH) return `The password must be at least ${PASSWORD_MIN_LENGTH} characters long.`;
  if (!/[A-Za-z]/.test(password) || !/[0-9]/.test(password)) return 'The password must contain both letters and digits.';
  return null;
}

export interface RegistrationForm {
  fullName: string;
  username: string;
  email: string;
  password: string;
  passwordConfirmation: string;
}

export function validateRegistration(form: RegistrationForm): Record<string, string> {
  const errors: Record<string, string> = {};
  if (form.fullName.trim().length < 2) errors.fullName = 'Enter your full name.';
  if (!/^[A-Za-z0-9._-]{3,100}$/.test(form.username.trim())) {
    errors.username = 'A username is 3 to 100 characters: letters, digits, dot, underscore or hyphen.';
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) errors.email = 'Enter a valid email address.';
  const pw = passwordProblem(form.password);
  if (pw) errors.password = pw;
  if (form.password !== form.passwordConfirmation) errors.passwordConfirmation = 'The passwords do not match.';
  return errors;
}
