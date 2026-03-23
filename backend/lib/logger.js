/**
 * Minimal structured JSON logger.
 * Outputs one JSON line per log entry — easy to pipe into any log aggregator.
 */

function log(level, msg, meta = {}) {
  process.stdout.write(JSON.stringify({ ts: new Date().toISOString(), level, msg, ...meta }) + "\n");
}

export const logger = {
  info:  (msg, meta) => log("info",  msg, meta),
  warn:  (msg, meta) => log("warn",  msg, meta),
  error: (msg, meta) => log("error", msg, meta),
};
