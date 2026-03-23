/**
 * GET /api/download/:jobId?token=
 * Single-use download — deletes the output file after streaming.
 */

import { Router } from "express";
import { createReadStream, existsSync } from "fs";
import { unlink } from "fs/promises";
import { makeRateLimiter, validateJobId, validateToken, safeFilename } from "../lib/middleware.js";
import { getJob } from "../lib/jobs.js";

const router = Router();

const downloadRateLimit = makeRateLimiter(
  parseInt(process.env.RATE_LIMIT_STREAM ?? "30")
)("download requests");

router.get("/:jobId", validateJobId, downloadRateLimit, (req, res) => {
  const { jobId } = req.params;
  const token = String(req.query.token ?? "");

  const job = getJob(jobId);
  if (!job)                  return res.status(404).json({ error: "Job not found" });
  if (job.status !== "done") return res.status(404).json({ error: "Job not ready" });
  if (!validateToken(job, token)) return res.status(403).json({ error: "Invalid or missing access token" });

  const outputPath = job.outputPath;
  if (!outputPath || !existsSync(outputPath)) {
    return res.status(410).json({ error: "File already downloaded or expired" });
  }

  const safeName = safeFilename(`clipped_${job.filename}`);
  res.set({
    "Content-Disposition": `attachment; filename="${safeName}"`,
    "Content-Type":        "video/mp4",
  });

  const stream = createReadStream(outputPath);
  stream.pipe(res);
  stream.on("close", () => unlink(outputPath).catch(() => {}));
});

export default router;
