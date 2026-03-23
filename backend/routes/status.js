/**
 * GET /api/status/:jobId?token=   — poll job status without SSE
 * GET /api/health                 — service health + capacity
 */

import { Router } from "express";
import { validateJobId, validateToken } from "../lib/middleware.js";
import { getJob, jobs, semaphore } from "../lib/jobs.js";
import { MAX_CONCURRENT } from "../lib/config.js";

const router = Router();

// Per-job status poll (useful for clients that can't use SSE)
router.get("/status/:jobId", validateJobId, (req, res) => {
  const token = String(req.query.token ?? "");
  const job   = getJob(req.params.jobId);

  if (!job) return res.status(404).json({ error: "Job not found" });
  if (!validateToken(job, token)) return res.status(403).json({ error: "Invalid or missing access token" });

  return res.json({
    job_id:   req.params.jobId,
    status:   job.status,
    stage:    job.stage,
    duration: job.duration,
    error:    job.error ?? undefined,
  });
});

// Service health
router.get("/health", (_req, res) => {
  const all     = [...jobs.values()];
  const queued  = all.filter(j => j.status === "queued").length;
  const running = all.filter(j => j.status === "running").length;

  res.json({
    status:    "ok",
    capacity:  MAX_CONCURRENT,
    available: semaphore.available,
    queued,
    running,
    total:     all.length,
  });
});

export default router;
