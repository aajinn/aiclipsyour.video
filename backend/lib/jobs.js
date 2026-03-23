/**
 * In-memory job store + semaphore.
 * Isolated here so routes never touch global state directly.
 */

import { unlinkSync } from "fs";
import { JOB_TTL_MS, MAX_CONCURRENT } from "./config.js";
import { logger } from "./logger.js";

// ── Job store ─────────────────────────────────────────────────────────────────

/** @type {Map<string, Job>} */
export const jobs = new Map();

/**
 * @typedef {object} Job
 * @property {"queued"|"running"|"done"|"error"} status
 * @property {string}   token
 * @property {string}   filename
 * @property {number}   createdAt
 * @property {number|null} duration
 * @property {string|null} stage
 * @property {string|null} error
 * @property {string|null} outputPath
 * @property {import("express").Response[]} sseClients
 * @property {string}   ip
 */

export function createJob(jobId, { token, filename, duration, ip }) {
  jobs.set(jobId, {
    status:     "queued",
    token,
    filename,
    createdAt:  Date.now(),
    duration,
    stage:      null,
    error:      null,
    outputPath: null,
    sseClients: [],
    ip,
  });
  return jobs.get(jobId);
}

export function getJob(jobId) {
  return jobs.get(jobId) ?? null;
}

export function pushSSE(job, data) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  for (const res of job.sseClients) {
    try { res.write(payload); } catch { /* disconnected */ }
  }
}

export function closeSSEClients(job) {
  for (const res of job.sseClients) {
    try { res.end(); } catch { /* ignore */ }
  }
  job.sseClients = [];
}

// ── Semaphore ─────────────────────────────────────────────────────────────────

class Semaphore {
  constructor(max) {
    this.max     = max;
    this.current = 0;
    this._queue  = [];
  }
  acquire() {
    if (this.current < this.max) { this.current++; return Promise.resolve(); }
    return new Promise(resolve => this._queue.push(resolve));
  }
  release() {
    this.current--;
    if (this._queue.length) { this.current++; this._queue.shift()(); }
  }
  get available() { return this.max - this.current; }
}

export const semaphore = new Semaphore(MAX_CONCURRENT);

// ── TTL cleanup ───────────────────────────────────────────────────────────────

setInterval(() => {
  const now   = Date.now();
  let cleaned = 0;
  for (const [jobId, job] of jobs) {
    if (job.createdAt < now - JOB_TTL_MS) {
      if (job.outputPath) try { unlinkSync(job.outputPath); } catch { /* ignore */ }
      jobs.delete(jobId);
      cleaned++;
    }
  }
  if (cleaned) logger.info("TTL cleanup", { removed: cleaned });
}, 600_000);
