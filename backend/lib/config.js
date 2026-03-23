import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..", "..");

export const PORT               = parseInt(process.env.PORT               ?? "8000");
export const ALLOWED_ORIGIN     = process.env.ALLOWED_ORIGIN              ?? "*";
export const MAX_CONCURRENT     = parseInt(process.env.MAX_JOBS           ?? "10");
export const MAX_UPLOAD_MB      = parseInt(process.env.MAX_UPLOAD_MB      ?? "500");
export const JOB_TTL_MS         = parseInt(process.env.JOB_TTL            ?? "3600") * 1000;
export const RATE_LIMIT_UPLOADS = parseInt(process.env.RATE_LIMIT_UPLOADS ?? "5");
export const RATE_LIMIT_STREAM  = parseInt(process.env.RATE_LIMIT_STREAM  ?? "30");
export const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT    ?? "300") * 1000;
export const MIN_VIDEO_SECONDS  = 180;
export const MAX_QUEUE_DEPTH    = MAX_CONCURRENT * 4;

export const UPLOAD_DIR = path.join(root, "uploads");
export const OUTPUT_DIR = path.join(root, "outputs");

export const ALLOWED_EXTENSIONS = new Set([
  ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a",
]);

export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
