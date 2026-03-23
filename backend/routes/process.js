/**
 * POST /api/process
 * Accepts a video/audio upload, queues a clip job, returns { job_id, token, duration }
 */

import { Router } from "express";
import multer from "multer";
import { randomUUID } from "crypto";
import { openSync, readSync, closeSync, unlinkSync } from "fs";
import path from "path";
import { probeDuration } from "../clipper.js";
import { UPLOAD_DIR, OUTPUT_DIR, MAX_UPLOAD_MB, MIN_VIDEO_SECONDS, MAX_QUEUE_DEPTH, ALLOWED_EXTENSIONS } from "../lib/config.js";
import { makeRateLimiter, validateUpload, safeFilename, clientIp } from "../lib/middleware.js";
import { jobs, createJob } from "../lib/jobs.js";
import { runJob, buildParams } from "../lib/runner.js";
import { logger } from "../lib/logger.js";

const router = Router();

const uploadRateLimit = makeRateLimiter(
  parseInt(process.env.RATE_LIMIT_UPLOADS ?? "5")
)("uploads");

const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (_req, file, cb) =>
    cb(null, `${randomUUID()}_input${path.extname(safeFilename(file.originalname)) || ".mp4"}`),
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) return cb(Object.assign(new Error("Unsupported file type"), { status: 415 }));
    cb(null, true);
  },
});

router.post("/", uploadRateLimit, (req, res, next) => {
  // Queue depth guard
  const active = [...jobs.values()].filter(j => j.status === "queued" || j.status === "running").length;
  if (active >= MAX_QUEUE_DEPTH) {
    return res.status(503).json({ error: "Server is busy — please try again shortly" });
  }

  upload.single("file")(req, res, async (err) => {
    if (err) {
      if (err.code === "LIMIT_FILE_SIZE")
        return res.status(413).json({ error: `File exceeds ${MAX_UPLOAD_MB} MB limit` });
      return res.status(err.status ?? 400).json({ error: err.message });
    }
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const inputPath = req.file.path;

    // Magic-byte validation
    try {
      const fd  = openSync(inputPath, "r");
      const buf = Buffer.alloc(16);
      readSync(fd, buf, 0, 16, 0);
      closeSync(fd);
      const check = validateUpload(req.file.originalname, buf);
      if (!check.ok) {
        unlinkSync(inputPath);
        return res.status(check.status).json({ error: check.error });
      }
    } catch (e) {
      try { unlinkSync(inputPath); } catch { /* ignore */ }
      return res.status(415).json({ error: "Could not validate file" });
    }

    // Duration check
    let duration = null;
    try {
      duration = probeDuration(inputPath);
      if (duration < MIN_VIDEO_SECONDS) {
        unlinkSync(inputPath);
        return res.status(422).json({
          error: `Video is ${Math.round(duration)}s — minimum ${MIN_VIDEO_SECONDS / 60} min required.`,
        });
      }
    } catch (e) {
      logger.warn("ffprobe_failed", { msg: e.message });
    }

    const jobId    = randomUUID();
    const token    = randomUUID() + randomUUID();
    const outExt   = path.extname(req.file.filename);
    const outPath  = path.join(OUTPUT_DIR, `${randomUUID()}_output${outExt}`);
    const safeName = safeFilename(req.file.originalname);
    const ip       = clientIp(req);

    createJob(jobId, { token, filename: safeName, duration, ip });

    const params = buildParams();
    runJob(jobId, inputPath, outPath, params).catch(e =>
      logger.error("run_job_error", { jobId, msg: e.message })
    );

    logger.info("job_queued", { jobId, file: safeName, ip });
    return res.status(202).json({ job_id: jobId, token, duration });
  });
});

export default router;
