/**
 * GET /api/stream/:jobId?token=
 * Server-Sent Events — streams job progress to the client.
 */

import { Router } from "express";
import { makeRateLimiter, validateJobId, validateToken, clientIp } from "../lib/middleware.js";
import { getJob } from "../lib/jobs.js";

const router = Router();

const streamRateLimit = makeRateLimiter(
  parseInt(process.env.RATE_LIMIT_STREAM ?? "30")
)("stream requests");

router.get("/:jobId", validateJobId, streamRateLimit, (req, res) => {
  const { jobId } = req.params;
  const token = String(req.query.token ?? "");

  const job = getJob(jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (!validateToken(job, token)) return res.status(403).json({ error: "Invalid or missing access token" });

  res.set({
    "Content-Type":      "text/event-stream",
    "Cache-Control":     "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
  });
  res.flushHeaders();

  // Replay current state for reconnecting clients
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

  const keepalive = setInterval(() => {
    try { res.write(": keepalive\n\n"); } catch { /* ignore */ }
  }, 25_000);

  req.on("close", () => {
    clearInterval(keepalive);
    job.sseClients = job.sseClients.filter(r => r !== res);
  });
});

export default router;
