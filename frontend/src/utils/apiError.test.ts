import { describe, expect, it } from 'vitest';
import { describeApiError } from './apiError';

describe('describeApiError', () => {
  it('renders a validation list field by field instead of "[object Object]"', () => {
    const err = { response: { status: 422, data: { detail: [
      { loc: ['body', 'patientId'], msg: 'Field required' },
      { loc: ['body', 'laterality'], msg: "Input should be 'OD' or 'OS'" },
    ] } } };
    expect(describeApiError(err, 'The assessment was not created'))
      .toBe("The assessment was not created (HTTP 422): patientId: Field required; laterality: Input should be 'OD' or 'OS'");
  });
  it('passes a string detail through with the status', () => {
    const err = { response: { status: 403, data: { detail: 'This account is suspended.' } } };
    expect(describeApiError(err, 'Sign-in did not complete')).toBe('Sign-in did not complete (HTTP 403): This account is suspended.');
  });
  it('names an unreachable server', () => {
    expect(describeApiError({ request: {} }, 'The records could not be loaded')).toMatch(/could not be reached/);
  });
  it('explains a bare status code', () => {
    expect(describeApiError({ response: { status: 503, data: {} } }, 'The report could not be downloaded')).toMatch(/not ready/);
  });
});
