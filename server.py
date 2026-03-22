"""
FastAPI server for Aiclipsyour.video

Security hardening:
  1.  File extension whitelist + magic-byte validation
  2.  Path traversal guard — job_id must be a valid v4 UUID
  3.  Sanitised filenames — null bytes, path separators, unsafe chars stripped
  4.  Per-IP upload rate limit (sliding window, configurable)
  5.  Rate limit on SSE stream + download endpoints (separate, looser window)
  6.  Job ownership token — random secret issued at upload time, required for
      stream/download — prevents job enumeration / cross-user data access
  7.  Global queue depth cap — rejects new jobs when server is saturated
  8.  Upload size cap enforced during streaming (not just after full read)
  9.  Request timeout middleware — kills slow/stalled connections (Slowloris)
  10. Trusted-proxy guard — X-Forwarded-For only trusted from configured CIDRs
  11. Security headers on every response (CSP, HSTS, X-Frame-Options, etc.)
  12. CORS locked to configured origin
  13. Generic error responses — no internal paths or stack traces to clients
  14. API keys never written to event files or logs
  15. Output files deleted after download (single-use links)
"""
import os
import re
import uuid
import asyncio
import logging
import time
import json as _json
import subprocess
import traceback
import ipaddress
import hashlib
import secrets
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aiclipsyour")

# ── Config ────────────────────────────────────────────────────────────────────
ALLOWED_ORIGIN       = os.environ.get("ALLOWED_ORIGIN", "*")
MAX_CONCURRENT_JOBS  = int(os.environ.get("MAX_JOBS", 2))
MAX_UPLOAD_MB        = int(os.environ.get("MAX_UPLOAD_MB", 500))
JOB_TTL_SECONDS      = int(os.environ.get("JOB_TTL", 3600))
MIN_VIDEO_SECONDS    = 180
MAX_QUEUE_DEPTH      = MAX_CONCURRENT_JOBS * 4
RATE_LIMIT_UPLOADS   = int(os.environ.get("RATE_LIMIT_UPLOADS", 5))    # per IP per hour
RATE_LIMIT_STREAM    = int(os.environ.get("RATE_LIMIT_STREAM", 30))    # per IP per hour
REQUEST_TIMEOUT_SECS = int(os.environ.get("REQUEST_TIMEOUT", 300))     # 5 min max per request

# Comma-separated CIDRs that are allowed to set X-Forwarded-For.
# Empty = trust nobody (use direct client IP only).
# Example: "10.0.0.0/8,172.16.0.0/12" for typical reverse-proxy ranges.
_TRUSTED_PROXY_CIDRS_RAW = os.environ.get("TRUSTED_PROXIES", "")
_TRUSTED_PROXY_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
for _cidr in filter(None, _TRUSTED_PROXY_CIDRS_RAW.split(",")):
    try:
        _TRUSTED_PROXY_NETS.append(ipaddress.ip_network(_cidr.strip(), strict=False))
    except ValueError:
        log.warning("Invalid TRUSTED_PROXIES entry ignored: %s", _cidr)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a"}
MAGIC_BYTES: dict[bytes, str] = {
    b"\x00\x00\x00\x18ftyp": "mp4",
    b"\x00\x00\x00\x1cftyp": "mp4",
    b"\x00\x00\x00\x20ftyp": "mp4",
    b"ftyp":                  "mp4",
    b"\x1aE\xdf\xa3":         "mkv/webm",
    b"RIFF":                  "avi/wav",
    b"\xff\xfb":              "mp3",
    b"\xff\xf3":              "mp3",
    b"\xff\xf2":              "mp3",
    b"ID3":                   "mp3",
}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore, _executor
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    _executor  = ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
    asyncio.create_task(_cleanup_loop())
    log.info("Ready — max concurrent jobs: %d", MAX_CONCURRENT_JOBS)
    yield
    _executor.shutdown(wait=False, cancel_futures=True)

app = FastAPI(title="Aiclipsyour.video", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_semaphore: asyncio.Semaphore | None = None
_executor:  ProcessPoolExecutor | None = None
jobs: dict[str, dict] = {}

# Separate sliding-window buckets for uploads vs stream/download
_ip_uploads:  dict[str, list[float]] = defaultdict(list)
_ip_streams:  dict[str, list[float]] = defaultdict(list)

BEST_PARAMS = dict(
    silence_thresh=-35,
    min_silence_len=800,
    padding_ms=300,
    video_codec="libx264",
    audio_codec="aac",
    crf=23,
    audio_only=False,
    grok_model="llama-3.3-70b-versatile",
    language_code="en",
    speaker_labels=True,
    save_transcript=True,
)

STAGE_MAP = {
    "loading":       "Preparing your video",
    "extracting":    "Preparing your video",
    "transcrib":     "Transcribing speech",
    "smart cutting": "AI smart cutting",
    "detecting":     "Detecting silences",
    "cutting":       "Exporting final video",
    "concatenating": "Exporting final video",
    "writing":       "Exporting final video",
}


# ── Security middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def request_timeout(request: Request, call_next):
    """Kill connections that take longer than REQUEST_TIMEOUT_SECS (Slowloris defence)."""
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content={"detail": "Request timeout"})


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    h = response.headers
    h["X-Content-Type-Options"]  = "nosniff"
    h["X-Frame-Options"]         = "DENY"
    h["X-XSS-Protection"]        = "1; mode=block"
    h["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    h["Permissions-Policy"]      = "camera=(), microphone=(), geolocation=()"
    h["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    h["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none';"
    )
    # Never cache API responses
    if request.url.path.startswith("/api/"):
        h["Cache-Control"] = "no-store"
        h["Pragma"]        = "no-cache"
    return response


# ── Generic error handler — no internal details to clients ───────────────────

@app.exception_handler(StarletteHTTPException)
async def _http_err(request: Request, exc: StarletteHTTPException):
    # Pass through our own HTTPExceptions with their messages intact,
    # but suppress any unexpected 5xx detail that might leak paths/tracebacks.
    if exc.status_code >= 500:
        log.error("Unhandled %s on %s", exc.status_code, request.url.path)
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": "Internal server error"})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def _unhandled_err(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    """
    Return the real client IP.
    Only trust X-Forwarded-For when the direct peer is in TRUSTED_PROXY_NETS.
    Otherwise use the direct connection IP to prevent spoofing.
    """
    peer = request.client.host if request.client else "unknown"
    if not _TRUSTED_PROXY_NETS:
        return peer
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    trusted = any(peer_addr in net for net in _TRUSTED_PROXY_NETS)
    if trusted:
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
    return peer


def _validate_job_id(job_id: str):
    """Raise 400 if job_id is not a valid v4 UUID — prevents path traversal."""
    if not _UUID_RE.match(job_id):
        raise HTTPException(400, "Invalid job ID")


def _check_rate_limit(bucket: dict[str, list[float]], ip: str,
                      limit: int, window: int = 3600, label: str = "requests"):
    """Generic sliding-window rate limiter."""
    now  = time.time()
    hits = [t for t in bucket[ip] if now - t < window]
    bucket[ip] = hits
    if len(hits) >= limit:
        raise HTTPException(429, f"Too many {label} — please wait before trying again")
    bucket[ip].append(now)


def _validate_upload(filename: str, header_bytes: bytes):
    """Extension whitelist + magic-byte check."""
    # Strip null bytes — can truncate paths on some systems
    filename = filename.replace("\x00", "")
    suffix   = Path(filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file type")

    valid = False
    for magic in MAGIC_BYTES:
        if header_bytes.startswith(magic):
            valid = True
            break
    if not valid and header_bytes[4:8] == b"ftyp":
        valid = True
    if not valid and header_bytes[:4] == b"\x1aE\xdf\xa3":
        valid = True
    if not valid:
        raise HTTPException(415, "File content does not match a supported video/audio format")


def _safe_filename(name: str) -> str:
    """
    Strip directory components, null bytes, and unsafe characters.
    Caps at 128 chars to prevent filesystem issues.
    """
    name = name.replace("\x00", "")          # null-byte injection
    name = Path(name).name                   # strip any directory traversal
    name = re.sub(r"[^\w\s\-.]", "_", name) # only safe chars
    name = name.strip(". ")                  # no leading/trailing dots or spaces
    return name[:128] or "upload"


def _verify_job_token(job: dict, token: str | None):
    """
    Constant-time comparison of the job access token.
    Raises 403 if token is missing or wrong.
    """
    expected = job.get("token", "")
    if not token or not secrets.compare_digest(expected, token):
        raise HTTPException(403, "Invalid or missing access token")


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(job_id: str, input_path: str, output_path: str,
            params: dict, event_file: str) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dotenv import load_dotenv as _lde
    _lde(Path(__file__).parent / ".env")
    import clipper as _c

    def _push(event: dict):
        try:
            with open(event_file, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(event) + "\n")
                fh.flush()
        except Exception:
            pass

    def _noop(*a, **kw): pass

    orig = {k: getattr(_c, k) for k in ("ok", "err", "info", "warn", "step")}

    def _step(n, t, m):
        orig["step"](n, t, m)
        label = None
        for key, stage in STAGE_MAP.items():
            if key in m.lower():
                label = stage
                break
        _push({"type": "stage", "stage": label or m[:40]})

    _c.ok   = _noop
    _c.err  = lambda m: _push({"type": "error_detail", "msg": str(m)})
    _c.info = _noop
    _c.warn = _noop
    _c.step = _step

    try:
        _c.clip_video(
            input_path=input_path, output_path=output_path,
            silence_thresh=params["silence_thresh"],
            min_silence_len=params["min_silence_len"],
            padding_ms=params["padding_ms"],
            video_codec=params["video_codec"],
            audio_codec=params["audio_codec"],
            crf=params["crf"],
            audio_only=params["audio_only"],
            use_grok=params["use_grok"],
            aai_api_key=params.get("aai_api_key"),
            grok_api_key=params.get("grok_api_key"),
            grok_model=params["grok_model"],
            language_code=params["language_code"],
            speaker_labels=params["speaker_labels"],
            save_transcript_file=params["save_transcript"],
        )
        _push({"type": "done"})
    except Exception as e:
        tb = traceback.format_exc()
        _push({"type": "error", "msg": f"{type(e).__name__}: {e}", "tb": tb})
    finally:
        for k, v in orig.items():
            setattr(_c, k, v)
        _push({"type": "__sentinel__"})
        for _ in range(5):
            try:
                Path(input_path).unlink(missing_ok=True)
                break
            except PermissionError:
                import time as _t
                _t.sleep(0.5)


async def _drain_event_file(job_id: str, event_file: str, async_queue: asyncio.Queue):
    read_pos = 0
    while True:
        await asyncio.sleep(0.3)
        try:
            with open(event_file, "r", encoding="utf-8") as fh:
                fh.seek(read_pos)
                new_data = fh.read()
                read_pos = fh.tell()
        except FileNotFoundError:
            continue

        for line in new_data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = _json.loads(line)
            except _json.JSONDecodeError:
                continue

            if event.get("type") == "__sentinel__":
                await async_queue.put(None)
                return

            job = jobs.get(job_id)

            if event.get("type") == "stage" and job:
                job["stage"] = event["stage"]

            if event.get("type") == "error":
                tb = event.pop("tb", "")
                log.error("Job %s worker error:\n%s", job_id, tb)
                if job:
                    job["status"] = "error"
                    job["error"]  = event["msg"]
                await async_queue.put({"type": "error", "msg": "Processing failed — check server logs."})
                await async_queue.put(None)
                return

            if event.get("type") == "error_detail":
                log.warning("Job %s clipper: %s", job_id, event.get("msg"))
                continue

            if event.get("type") == "done":
                if job:
                    job["status"] = "done"
                    job["output"] = jobs[job_id].get("output_path", "")
                await async_queue.put({"type": "done"})
                await async_queue.put(None)
                return

            await async_queue.put(event)


async def _run_job(job_id: str, input_path: str, output_path: str, params: dict):
    async with _semaphore:
        job  = jobs[job_id]
        job["status"]      = "running"
        job["output_path"] = output_path

        async_queue: asyncio.Queue = job["queue"]
        loop = asyncio.get_running_loop()

        event_file = str(Path(output_path).parent / f"{job_id}_events.jsonl")
        open(event_file, "w").close()

        drain_task = asyncio.create_task(
            _drain_event_file(job_id, event_file, async_queue)
        )

        try:
            await loop.run_in_executor(
                _executor, _worker, job_id, input_path, output_path, params, event_file
            )
        except Exception as e:
            log.error("Job %s executor error:\n%s", job_id, traceback.format_exc())
            job["status"] = "error"
            job["error"]  = str(e)
            await async_queue.put({"type": "error", "msg": "Processing failed — check server logs."})
            await async_queue.put(None)
            drain_task.cancel()
        else:
            try:
                await asyncio.wait_for(drain_task, timeout=15)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                log.warning("Job %s drain timed out", job_id)
                await async_queue.put(None)
        finally:
            try:
                Path(event_file).unlink(missing_ok=True)
            except Exception:
                pass

        if job["status"] == "running":
            job["status"] = "done"
            job["output"] = output_path

        log.info("Job %s → %s", job_id, job["status"])


# ── SSE ───────────────────────────────────────────────────────────────────────

async def _sse_generator(job_id: str) -> AsyncGenerator[str, None]:
    job = jobs.get(job_id)
    if not job:
        yield "data: {\"type\":\"error\",\"msg\":\"Job not found\"}\n\n"
        return

    if job.get("duration"):
        yield f"data: {_json.dumps({'type': 'duration', 'duration': job['duration']})}\n\n"

    if job.get("stage"):
        yield f"data: {_json.dumps({'type': 'stage', 'stage': job['stage']})}\n\n"

    if job["status"] == "done":
        yield f"data: {_json.dumps({'type': 'done'})}\n\n"
        return
    if job["status"] == "error":
        yield "data: {\"type\":\"error\",\"msg\":\"Processing failed. Please try again.\"}\n\n"
        return

    queue: asyncio.Queue = job["queue"]
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue

        if event is None:
            break

        yield f"data: {_json.dumps(event)}\n\n"

        if event.get("type") in ("done", "error"):
            break


# ── Startup / shutdown (moved to lifespan above) ─────────────────────────────


async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
        now   = time.time()
        stale = [jid for jid, j in list(jobs.items())
                 if j.get("created_at", now) < now - JOB_TTL_SECONDS]
        for jid in stale:
            j = jobs.pop(jid, {})
            for key in ("output", "output_path"):
                p = j.get(key)
                if p:
                    Path(p).unlink(missing_ok=True)
        if stale:
            log.info("Cleaned %d stale jobs", len(stale))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_video(request: Request, background_tasks: BackgroundTasks,
                        file: UploadFile = File(...)):
    ip = _client_ip(request)

    # Rate limit uploads
    _check_rate_limit(_ip_uploads, ip, RATE_LIMIT_UPLOADS, label="uploads")

    # Global queue depth cap
    waiting = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    if waiting >= MAX_QUEUE_DEPTH:
        raise HTTPException(503, "Server is busy — please try again shortly")

    # Read first 16 bytes for magic-byte check before writing to disk
    header_bytes = await file.read(16)
    _validate_upload(file.filename or "", header_bytes)

    size_mb   = len(header_bytes) / 1024 / 1024
    safe_name = _safe_filename(file.filename or "upload.mp4")
    suffix    = Path(safe_name).suffix or ".mp4"

    input_path = UPLOAD_DIR / f"{uuid.uuid4()}_input{suffix}"
    out_path   = OUTPUT_DIR / f"{uuid.uuid4()}_output{suffix}"

    with open(input_path, "wb") as f_out:
        f_out.write(header_bytes)
        while chunk := await file.read(1024 * 1024):
            size_mb += len(chunk) / 1024 / 1024
            if size_mb > MAX_UPLOAD_MB:
                input_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit")
            f_out.write(chunk)
        f_out.flush()
        os.fsync(f_out.fileno())

    # Duration check via ffprobe
    duration: float | None = None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(input_path)],
            capture_output=True, text=True, timeout=15
        )
        duration = float(_json.loads(probe.stdout)["format"]["duration"])
        if duration < MIN_VIDEO_SECONDS:
            input_path.unlink(missing_ok=True)
            raise HTTPException(
                422,
                f"Video is {duration:.0f}s — minimum {MIN_VIDEO_SECONDS // 60} min required."
            )
    except HTTPException:
        raise
    except Exception as e:
        log.warning("ffprobe failed: %s", e)

    aai_key  = os.environ.get("ASSEMBLYAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    params   = {**BEST_PARAMS, "use_grok": bool(aai_key and groq_key),
                "aai_api_key": aai_key, "grok_api_key": groq_key}

    job_id = str(uuid.uuid4())
    # Issue a random token — client must present it to stream/download
    job_token = secrets.token_urlsafe(32)

    jobs[job_id] = {
        "status":     "queued",
        "queue":      asyncio.Queue(),
        "filename":   safe_name,
        "created_at": time.time(),
        "duration":   duration,
        "token":      job_token,
        "ip":         ip,
    }

    background_tasks.add_task(_run_job, job_id, str(input_path), str(out_path), params)
    log.info("Queued %s — %s (%.1f MB) from %s", job_id, safe_name, size_mb, ip)
    return {"job_id": job_id, "token": job_token, "duration": duration}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str, request: Request, token: str = ""):
    _validate_job_id(job_id)
    ip = _client_ip(request)
    _check_rate_limit(_ip_streams, ip, RATE_LIMIT_STREAM, label="stream requests")

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    _verify_job_token(job, token)

    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}")
def download(job_id: str, request: Request, background_tasks: BackgroundTasks,
             token: str = ""):
    _validate_job_id(job_id)
    ip = _client_ip(request)
    _check_rate_limit(_ip_streams, ip, RATE_LIMIT_STREAM, label="download requests")

    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Not ready")
    _verify_job_token(job, token)

    output = Path(job["output"])
    if not output.exists():
        raise HTTPException(410, "File already downloaded or expired")

    safe = _safe_filename(f"clipped_{job['filename']}")
    background_tasks.add_task(output.unlink, True)
    return FileResponse(path=str(output), filename=safe, media_type="video/mp4")


@app.get("/api/health")
def health():
    return {
        "status":   "ok",
        "queued":   sum(1 for j in jobs.values() if j["status"] == "queued"),
        "running":  sum(1 for j in jobs.values() if j["status"] == "running"),
        "capacity": MAX_CONCURRENT_JOBS,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    # uvloop is not available on Windows — fall back to default asyncio loop
    import sys
    loop = "uvloop" if sys.platform != "win32" else "asyncio"
    uvicorn.run("server:app", host="0.0.0.0", port=8000,
                workers=1, loop=loop, reload=False)
