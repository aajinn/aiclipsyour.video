/**
 * server.js — aiclipsyour.video API microservice
 *
 * Routes:
 *   POST /api/process              Upload video, queue job
 *   GET  /api/stream/:jobId        SSE progress stream
 *   GET  /api/download/:jobId      Single-use output download
 *   GET  /api/status/:jobId        Poll job status (JSON)
 *   GET  /api/health               Service health + capacity
 */

import "dotenv/config";
import express from "express";
import { mkdirSync } from "fs";
import { UPLOAD_DIR, OUTPUT_DIR, PORT } from "./lib/config.js";
import { requestTimeout, securityHeaders, cors, requestLogger, errorHandler } from "./lib/middleware.js";
import processRoute  from "./routes/process.js";
import streamRoute   from "./routes/stream.js";
import downloadRoute from "./routes/download.js";
import statusRoute   from "./routes/status.js";
import { logger } from "./lib/logger.js";

// Ensure storage dirs exist
mkdirSync(UPLOAD_DIR, { recursive: true });
mkdirSync(OUTPUT_DIR, { recursive: true });

const app = express();

// ── Global middleware ─────────────────────────────────────────────────────────
app.use(requestTimeout);
app.use(securityHeaders);
app.use(cors);
app.use(requestLogger);
app.use(express.json());

// ── Routes ────────────────────────────────────────────────────────────────────
app.use("/api/process",  processRoute);
app.use("/api/stream",   streamRoute);
app.use("/api/download", downloadRoute);
app.use("/api",          statusRoute);

// 404 for anything else
app.use((_req, res) => res.status(404).json({ error: "Not found" }));

// ── Error handler ─────────────────────────────────────────────────────────────
app.use(errorHandler);

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  logger.info("server_start", { port: PORT });
});
