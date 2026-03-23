/**
 * Centralised frontend config.
 * All env vars are read here — components import from this file, never from process.env directly.
 */

// NEXT_PUBLIC_* vars are baked in at build time by Next.js.
// In development this comes from .env.development, in production from .env.production
// or from Vercel's Environment Variables dashboard.
export const BACKEND_URL: string = (() => {
  const url = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (!url) {
    // Warn loudly in dev; in prod this means the env var wasn't set at build time
    console.error(
      "[config] NEXT_PUBLIC_BACKEND_URL is not set. " +
      "Set it in .env.production or in the Vercel Environment Variables dashboard."
    );
  }
  return (url ?? "").replace(/\/$/, ""); // strip trailing slash
})();

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
