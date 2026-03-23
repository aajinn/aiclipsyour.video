/**
 * Centralised frontend config.
 * All env vars are read here — components import from this file, never from process.env directly.
 */

export const BACKEND_URL: string =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const api = {
  process:  `${BACKEND_URL}/api/process`,
  stream:   (jobId: string, token: string) =>
    `${BACKEND_URL}/api/stream/${jobId}?token=${encodeURIComponent(token)}`,
  download: (jobId: string, token: string) =>
    `${BACKEND_URL}/api/download/${jobId}?token=${encodeURIComponent(token)}`,
  status:   (jobId: string, token: string) =>
    `${BACKEND_URL}/api/status/${jobId}?token=${encodeURIComponent(token)}`,
  health:   `${BACKEND_URL}/api/health`,
};
