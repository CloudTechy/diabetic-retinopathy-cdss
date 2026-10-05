/**
 * Turn a failed API call into the sentence a user needs: what the server
 * actually said, not a generic "error occurred".
 *
 * FastAPI answers a malformed request with 422 and `detail` as a LIST of
 * `{ loc, msg }` objects. Interpolating that list into a string printed
 * "[object Object]", so a missing field in the assessment payload reached the
 * user as an error with no cause. Each entry is rendered as "field: message".
 */
type ValidationItem = { loc?: Array<string | number>; msg?: string };

const fieldName = (loc: Array<string | number> | undefined): string =>
  (loc || []).filter((part) => part !== 'body' && part !== 'query' && part !== 'path').join('.');

const describeDetail = (detail: unknown): string | null => {
  if (!detail) return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((item: ValidationItem | string) => {
      if (typeof item === 'string') return item;
      const field = fieldName(item?.loc);
      const msg = item?.msg || 'invalid value';
      return field ? `${field}: ${msg}` : msg;
    });
    return parts.length ? parts.join('; ') : null;
  }
  if (typeof detail === 'object') {
    const d = detail as Record<string, unknown>;
    const text = d.message || d.reason || d.rejectionReason || d.error;
    const code = d.code || d.errorCode;
    if (typeof text === 'string') return typeof code === 'string' ? `${text} (${code})` : text;
    try {
      return JSON.stringify(detail);
    } catch {
      return null;
    }
  }
  return String(detail);
};

export function describeApiError(err: any, action: string): string {
  const response = err?.response;
  if (!response) {
    if (err?.code === 'ECONNABORTED') return `${action}: the server did not answer in time.`;
    if (err?.request) return `${action}: the server could not be reached. Check that the backend is running and the network is available.`;
    return `${action}: ${err?.message || 'unexpected error'}.`;
  }
  const status: number = response.status;
  const detail = describeDetail(response.data?.detail ?? response.data?.message);
  if (detail) return `${action} (HTTP ${status}): ${detail}`;
  if (status === 401) return `${action}: the session is not authenticated. Please sign in again.`;
  if (status === 403) return `${action}: this account is not permitted to do that.`;
  if (status === 404) return `${action}: the record was not found on the server.`;
  if (status === 413) return `${action}: the file is larger than the server accepts.`;
  if (status === 503) return `${action}: the server is not ready (for example, the model is not loaded).`;
  return `${action}: the server returned HTTP ${status} with no explanation.`;
}
