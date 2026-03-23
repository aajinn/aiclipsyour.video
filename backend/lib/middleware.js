import { timingSafeEqual } from "crypto";
import path from "path";
import { ALLOWED_EXTENSIONS, ALLOWED_ORIGIN, REQUEST_TIMEOUT_MS, UUID_RE } from "./config.js";
import { logger } from "./logger.js";

// ── Client IP ─────────────────────────────────────────────────────────────────

export function clientIp(req) {
  return (req.headers["x-forwarded-for"] ?? req.socket?.remoteAddress ?? "unknown")
    .split(",")[0].trim();
}

// ── Rate limiter ──────────────────────────────────────────────────────────────

export function makeRateLimiter(limit, windowMs = 3_600_000) {
  const bucket = new Map();
  return function rateLimit(label = "requests") {
    return (req, res, next) => {
      const ip   = clientIp(req);
      const now  = Date.now();
      const hits = (bucket.get(ip) ?? []).filter(t => now - t < windowMs);
      if (hits.length >= limit) {
        return res.status(429).json({ error: `Too many ${label} — please wait` });
      }
      hits.push(now);
      bucket.set(ip, hits);
      next();
    };
  };
}

// ── Validation helpers ────────────────────────────────────────────────────────

export function validateJobId(req, res, next) {
  if (!UUID_RE.test(req.params.jobId)) {
    return res.status(400).json({ error: "Invalid job ID" });
  }
  next();
}

export function validateToken(job, token) {
  if (!token || !job) return false;
  const a = Buffer.from(job.token);
  const b = Buffer.from(token);
  return a.length === b.length && timingSafeEqual(a, b);
}

const MAGIC_CHECKS = [
  buf => buf.slice(4, 8).toString("ascii") === "ftyp",
  buf => buf.slice(0, 4).equals(Buffer.from([0x1a, 0x45, 0xdf, 0xa3])),
  buf => buf.slice(0, 4).toString("ascii") === "RIFF",
  buf => buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0,
  buf => buf.slice(0, 3).toString("ascii") === "ID3",
];

export function validateUpload(filename, headerBuf) {
  const ext = path.extname(filename).toLowerCase();
  if (!ext || !ALLOWED_EXTENSIONS.has(ext)) {
    return { ok: false, status: 415, error: "Unsupported file type" };
  }
  if (!MAGIC_CHECKS.some(fn => fn(headerBuf))) {
    return { ok: false, status: 415, error: "File content does not match a supported format" };
  }
  return { ok: true };
}

export function safeFilename(name) {
  return (path.basename(name).replace(/\x00/g, "").replace(/[^\w\s\-.]/g, "_").trim().slice(0, 128)) || "upload";
}

// ── Global middleware ─────────────────────────────────────────────────────────

export function requestTimeout(req, res, next) {
  req.socket.setTimeout(REQUEST_TIMEOUT_MS);
  next();
}

export function securityHeaders(req, res, next) {
  res.set({
    "X-Content-Type-Options":    "nosniff",
    "X-Frame-Options":           "DENY",
    "X-XSS-Protection":          "1; mode=block",
    "Referrer-Policy":           "strict-origin-when-cross-origin",
    "Permissions-Policy":        "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Cache-Control":             "no-store",
    "Pragma":                    "no-cache",
  });
  next();
}

export function cors(req, res, next) {
  res.set({
    "Access-Control-Allow-Origin":  ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  });
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
}

export function requestLogger(req, res, next) {
  const start = Date.now();
  res.on("finish", () => {
    logger.info("request", {
      method: req.method,
      path:   req.path,
      status: res.statusCode,
      ms:     Date.now() - start,
      ip:     clientIp(req),
    });
  });
  next();
}

export function errorHandler(err, req, res, _next) {
  logger.error("unhandled", { msg: err.message, stack: err.stack });
  const status = err.status ?? 500;
  res.status(status).json({ error: status < 500 ? err.message : "Internal server error" });
}
