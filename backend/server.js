/**
 * server.js — Express HTTP server for aiclipsyour.video
 *
 * Security:
 *  - File extension whitelist + magic-byte validation
 *  - UUID v4 job ID validation (path traversal guard)
 *  - Per-IP sliding-window rate limiting (uploads + streams)
 *  - Job ownership token (random secret, required for stream/download)
 *  - Global queue depth cap
 *  - Upload size cap enforced during streaming
 *  - Request timeout middleware
 *  - Security headers on every response
 *  - CORS locked to configured origin
 *  - Generic error responses (no internal paths/stack traces)
 *  - Output files deleted after download (single-use links)
 */

import "dotenv/config";
import express from "express";
import multer from "multer";
import { randomUUID, timingSafeEqual } from "crypto";
import { createReadStream, mkdirSync, existsSync, unlinkSync } from "fs";
import { unlink } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { Worker } from "worker_threads";
import { clipVideo, probeDuration, BEST_PARAMS } from "./clipper.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────────────────
const PORT               = parseInt(process.env.PORT ?? "8000");
const ALLOWED_ORIGIN     = process.env.ALLOWED_ORIGIN ?? "*";
const MAX_CONCURRENT     = parseInt(process.env.MAX_JOBS ?? "10");
const MAX_UPLOAD_MB      = parseInt(process.env.MAX_UPLOAD_MB ?? "500");
const JOB_TTL_MS         = parseInt(process.env.JOB_TTL ?? "3600") * 1000;
const RATE_LIMIT_UPLOADS = parseInt(process.env.RATE_LIMIT_UPLOADS ?? "5");
const RATE_LIMIT_STREAM  = parseInt(process.env.RATE_LIMIT_STREAM ?? "30");
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT ?? "300") * 1000;
const MIN_VIDEO_SECONDS  = 180;
const MAX_QUEUE_DEPTH    = MAX_CONCURRENT * 4; // 40 queued jobs max

const UPLOAD_DIR = path.join(__dirname, "..", "uploads");
const OUTPUT_DIR = path.join(__dirname, "..", "outputs");
mkdirSync(UPLOAD_DIR, { recursive: true });
mkdirSync(OUTPUT_DIR, { recursive: true });

const ALLOWED_EXTENSIONS = new Set([".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a"]);
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

// Magic bytes for video/audio validation
const MAGIC_CHECKS = [
  buf => buf.slice(4, 8).toString("ascii") === "ftyp",          // mp4/mov
  buf => buf.slice(0, 4).equals(Buffer.from([0x1a, 0x45, 0xdf, 0xa3])), // mkv/webm
  buf => buf.slice(0, 4).toString("ascii") === "RIFF",          // avi/wav
  buf => buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0,           // mp3
  buf => buf.slice(0, 3).toString("ascii") === "ID3",           // mp3 id3
];

// ── State ─────────────────────────────────────────────────────────────────────
/** @type {Map<string, {status, token, filename, createdAt, duration, stage, error, outputPath, sseClients}>} */
const jobs = new Map();
let activeJobs = 0; // kept for health endpoint compat

// Rate limit buckets: ip -> [timestamp, ...]
const uploadBucket = new Map();
const streamBucket = new Map();

// ── Helpers ───────────────────────────────────────────────────────────────────

function clientIp(req) {
  return (req.headers["x-forwarded-for"] ?? req.socket.remoteAddress ?? "unknown").split(",")[0].trim();
}

function validateJobId(jobId) {
  if (!UUID_RE.test(jobId)) throw Object.assign(new Error("Invalid job ID"), { status: 400 });
}

function checkRateLimit(bucket, ip, limit, label = "requests") {
  const now  = Date.now();
  const hits = (bucket.get(ip) ?? []).filter(t => now - t < 3_600_000);
  if (hits.length >= limit) throw Object.assign(new Error(`Too many ${label} — please wait`), { status: 429 });
  hits.push(now);
  bucket.set(ip, hits);
}

function validateUpload(filename, headerBuf) {
  const ext = path.extname(filename).toLowerCase();
  if (!ext || !ALLOWED_EXTENSIONS.has(ext)) throw Object.assign(new Error("Unsupported file type"), { status: 415 });
  if (!MAGIC_CHECKS.some(fn => fn(headerBuf))) throw Object.assign(new Error("File content does not match a supported format"), { status: 415 });
}

function safeFilename(name) {
  return (path.basename(name).replace(/\x00/g, "").replace(/[^\w\s\-.]/g, "_").trim().slice(0, 128)) || "upload";
}

function verifyToken(job, token) {
  if (!token || !job) throw Object.assign(new Error("Invalid or missing access token"), { status: 403 });
  const a = Buffer.from(job.token);
  const b = Buffer.from(token);
  if (a.length !== b.length || !timingSafeEqual(a, b)) throw Object.assign(new Error("Invalid or missing access token"), { status: 403 });
}

function pushSSE(job, data) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  for (const res of job.sseClients) {
    try { res.write(payload); } catch { /* client disconnected */ }
  }
}

// ── Concurrency semaphore ─────────────────────────────────────────────────────

class Semaphore {
  constructor(max) {
    this.max     = max;
    this.current = 0;
    this.queue   = [];
  }
  acquire() {
    if (this.current < this.max) {
      this.current++;
      return Promise.resolve();
    }
    return new Promise(resolve => this.queue.push(resolve));
  }
  release() {
    this.current--;
    if (this.queue.length) {
      this.current++;
      this.queue.shift()();
    }
  }
}

const semaphore = new Semaphore(MAX_CONCURRENT);

// ── Job runner ────────────────────────────────────────────────────────────────

const STAGE_MAP = {
  "preparing":   "Preparing your video",
  "transcrib":   "Transcribing speech",
  "ai smart":    "AI smart cutting",
  "detecting":   "Detecting silences",
  "exporting":   "Exporting final video",
  "encoding":    "Exporting final video",
  "cutting":     "Exporting final video",
};

function mapStage(msg) {
  const lower = msg.toLowerCase();
  for (const [key, label] of Object.entries(STAGE_MAP)) {
    if (lower.includes(key)) return label;
  }
  return msg.slice(0, 40);
}

async function runJob(jobId, inputPath, outputPath, params) {
  await semaphore.acquire();

  const job = jobs.get(jobId);
  if (!job) { semaphore.release(); return; }
  job.status = "running";

  try {
    const result = await clipVideo({
      inputPath,
      outputPath,
      silenceThresh:    params.silence_thresh,
      minSilenceLen:    params.min_silence_len,
      paddingMs:        params.padding_ms,
      videoCodec:       params.video_codec,
      audioCodec:       params.audio_codec,
      crf:              params.crf,
      audioOnly:        params.audio_only,
      useGroq:          params.use_grok,
      aaiApiKey:        params.aai_api_key,
      groqApiKey:       params.grok_api_key,
      groqModel:        params.grok_model,
      languageCode:     params.language_code,
      speakerLabels:    params.speaker_labels,
      saveTranscriptFile: params.save_transcript,
      onStage: (stage) => {
        job.stage = stage;
        pushSSE(job, { type: "stage", stage });
      },
      onProgress: (msg) => {
        // map progress messages to stage labels where possible
        const stage = mapStage(msg);
        if (stage !== msg) {
          job.stage = stage;
          pushSSE(job, { type: "stage", stage });
        }
      },
    });

    job.status     = "done";
    job.outputPath = result.outputPath;
    pushSSE(job, { type: "done" });
  } catch (err) {
    console.error(`Job ${jobId} failed:`, err);
    job.status = "error";
    job.error  = err.message;
    pushSSE(job, { type: "error", msg: "Processing failed — check server logs." });
  } finally {
    semaphore.release();
    // Flush and close all SSE clients
    for (const res of job.sseClients) {
      try { res.end(); } catch { /* ignore */ }
    }
    job.sseClients = [];
    // Clean up input file
    try { unlinkSync(inputPath); } catch { /* ignore */ }
  }
}

// ── Cleanup loop ──────────────────────────────────────────────────────────────

setInterval(() => {
  const now = Date.now();
  for (const [jobId, job] of jobs) {
    if (job.createdAt < now - JOB_TTL_MS) {
      if (job.outputPath) try { unlinkSync(job.outputPath); } catch { /* ignore */ }
      jobs.delete(jobId);
    }
  }
}, 600_000);

// ── Express app ───────────────────────────────────────────────────────────────

const app = express();

// Request timeout
app.use((req, res, next) => {
  req.socket.setTimeout(REQUEST_TIMEOUT_MS);
  next();
});

// Security headers
app.use((req, res, next) => {
  res.set({
    "X-Content-Type-Options":  "nosniff",
    "X-Frame-Options":         "DENY",
    "X-XSS-Protection":        "1; mode=block",
    "Referrer-Policy":         "strict-origin-when-cross-origin",
    "Permissions-Policy":      "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Content-Security-Policy": [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src https://fonts.gstatic.com",
      "connect-src 'self'",
      "img-src 'self' data:",
      "frame-ancestors 'none'",
    ].join("; "),
  });
  if (req.path.startsWith("/api/")) {
    res.set({ "Cache-Control": "no-store", "Pragma": "no-cache" });
  }
  next();
});

// CORS
app.use((req, res, next) => {
  res.set("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.set("Access-Control-Allow-Methods", "GET, POST");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// ── Multer (streaming upload with size cap) ───────────────────────────────────

const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => cb(null, `${randomUUID()}_input${path.extname(safeFilename(file.originalname)) || ".mp4"}`),
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) return cb(Object.assign(new Error("Unsupported file type"), { status: 415 }));
    cb(null, true);
  },
});

// ── Routes ────────────────────────────────────────────────────────────────────

app.post("/api/process", (req, res, next) => {
  const ip = clientIp(req);
  try {
    checkRateLimit(uploadBucket, ip, RATE_LIMIT_UPLOADS, "uploads");
  } catch (e) { return res.status(e.status ?? 429).json({ detail: e.message }); }

  const waiting = [...jobs.values()].filter(j => j.status === "queued" || j.status === "running").length;
  if (waiting >= MAX_QUEUE_DEPTH) return res.status(503).json({ detail: "Server is busy — please try again shortly" });

  upload.single("file")(req, res, async (err) => {
    if (err) {
      if (err.code === "LIMIT_FILE_SIZE") return res.status(413).json({ detail: `File exceeds ${MAX_UPLOAD_MB} MB limit` });
      return res.status(err.status ?? 400).json({ detail: err.message });
    }
    if (!req.file) return res.status(400).json({ detail: "No file uploaded" });

    const inputPath = req.file.path;

    // Magic-byte check — read first 16 bytes
    try {
      const fd = await import("fs").then(m => m.openSync(inputPath, "r"));
      const buf = Buffer.alloc(16);
      const { readSync, closeSync } = await import("fs");
      readSync(fd, buf, 0, 16, 0);
      closeSync(fd);
      validateUpload(req.file.originalname, buf);
    } catch (e) {
      try { unlinkSync(inputPath); } catch { /* ignore */ }
      return res.status(e.status ?? 415).json({ detail: e.message });
    }

    // Duration check
    let duration = null;
    try {
      duration = probeDuration(inputPath);
      if (duration < MIN_VIDEO_SECONDS) {
        unlinkSync(inputPath);
        return res.status(422).json({ detail: `Video is ${Math.round(duration)}s — minimum ${MIN_VIDEO_SECONDS / 60} min required.` });
      }
    } catch (e) {
      console.warn("ffprobe failed:", e.message);
    }

    const aaiKey  = process.env.ASSEMBLYAI_API_KEY;
    const groqKey = process.env.GROQ_API_KEY;
    const params  = {
      silence_thresh:  BEST_PARAMS.silenceThresh,
      min_silence_len: BEST_PARAMS.minSilenceLen,
      padding_ms:      BEST_PARAMS.paddingMs,
      video_codec:     BEST_PARAMS.videoCodec,
      audio_codec:     BEST_PARAMS.audioCodec,
      crf:             BEST_PARAMS.crf,
      audio_only:      BEST_PARAMS.audioOnly,
      use_grok:        !!(aaiKey && groqKey),
      aai_api_key:     aaiKey,
      grok_api_key:    groqKey,
      grok_model:      BEST_PARAMS.groqModel,
      language_code:   BEST_PARAMS.languageCode,
      speaker_labels:  BEST_PARAMS.speakerLabels,
      save_transcript: BEST_PARAMS.saveTranscript,
    };

    const jobId    = randomUUID();
    const token    = randomUUID() + randomUUID(); // 72 chars of entropy
    const outExt   = path.extname(req.file.filename);
    const outPath  = path.join(OUTPUT_DIR, `${randomUUID()}_output${outExt}`);
    const safeName = safeFilename(req.file.originalname);

    jobs.set(jobId, {
      status:     "queued",
      token,
      filename:   safeName,
      createdAt:  Date.now(),
      duration,
      stage:      null,
      error:      null,
      outputPath: null,
      sseClients: [],
      ip,
    });

    // Fire and forget
    runJob(jobId, inputPath, outPath, params).catch(e => console.error("runJob error:", e));

    console.info(`Queued ${jobId} — ${safeName} from ${ip}`);
    return res.json({ job_id: jobId, token, duration });
  });
});

app.get("/api/stream/:jobId", (req, res) => {
  const { jobId } = req.params;
  const { token = "" } = req.query;
  const ip = clientIp(req);

  try {
    validateJobId(jobId);
    checkRateLimit(streamBucket, ip, RATE_LIMIT_STREAM, "stream requests");
  } catch (e) { return res.status(e.status ?? 400).json({ detail: e.message }); }

  const job = jobs.get(jobId);
  if (!job) return res.status(404).json({ detail: "Job not found" });
  try { verifyToken(job, token); } catch (e) { return res.status(403).json({ detail: e.message }); }

  res.set({
    "Content-Type":  "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":    "keep-alive",
  });
  res.flushHeaders();

  // Send current state immediately
  if (job.duration) res.write(`data: ${JSON.stringify({ type: "duration", duration: job.duration })}\n\n`);
  if (job.stage)    res.write(`data: ${JSON.stringify({ type: "stage",    stage:    job.stage    })}\n\n`);

  if (job.status === "done") {
    res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
    return res.end();
  }
  if (job.status === "error") {
    res.write(`data: ${JSON.stringify({ type: "error", msg: "Processing failed. Please try again." })}\n\n`);
    return res.end();
  }

  job.sseClients.push(res);

  // Keepalive
  const keepalive = setInterval(() => { try { res.write(": keepalive\n\n"); } catch { /* ignore */ } }, 25_000);

  req.on("close", () => {
    clearInterval(keepalive);
    job.sseClients = job.sseClients.filter(r => r !== res);
  });
});

app.get("/api/download/:jobId", async (req, res) => {
  const { jobId } = req.params;
  const { token = "" } = req.query;
  const ip = clientIp(req);

  try {
    validateJobId(jobId);
    checkRateLimit(streamBucket, ip, RATE_LIMIT_STREAM, "download requests");
  } catch (e) { return res.status(e.status ?? 400).json({ detail: e.message }); }

  const job = jobs.get(jobId);
  if (!job || job.status !== "done") return res.status(404).json({ detail: "Not ready" });
  try { verifyToken(job, token); } catch (e) { return res.status(403).json({ detail: e.message }); }

  const outputPath = job.outputPath;
  if (!outputPath || !existsSync(outputPath)) return res.status(410).json({ detail: "File already downloaded or expired" });

  const safeName = safeFilename(`clipped_${job.filename}`);
  res.set({
    "Content-Disposition": `attachment; filename="${safeName}"`,
    "Content-Type":        "video/mp4",
  });

  const stream = createReadStream(outputPath);
  stream.pipe(res);
  stream.on("close", () => {
    unlink(outputPath).catch(() => {});
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    status:   "ok",
    queued:   [...jobs.values()].filter(j => j.status === "queued").length,
    running:  [...jobs.values()].filter(j => j.status === "running").length,
    capacity: MAX_CONCURRENT,
    available: MAX_CONCURRENT - semaphore.current,
  });
});

// Serve static frontend
const staticDir = path.join(__dirname, "..", "static");
if (existsSync(staticDir)) {
  app.use(express.static(staticDir));
}

// Generic error handler
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(err.status ?? 500).json({ detail: err.status < 500 ? err.message : "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`aiclipsyour.video backend listening on http://0.0.0.0:${PORT}`);
});
