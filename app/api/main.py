"""FastAPI Orchestrator API for the Video Processing Engine."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.celery_app import celery_app
from app.models import JobConfig, JobStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Processing Engine")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _get_redis() -> redis_lib.Redis:
    """Return a Redis client. Raises redis.ConnectionError if unavailable."""
    client = redis_lib.from_url(REDIS_URL, socket_connect_timeout=2)
    client.ping()  # raises ConnectionError / TimeoutError if down
    return client


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _read_job(client: redis_lib.Redis, job_id: str) -> dict[str, Any] | None:
    raw = client.get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def _write_job(client: redis_lib.Redis, job_id: str, state: dict[str, Any]) -> None:
    client.set(_job_key(job_id), json.dumps(state))


# ---------------------------------------------------------------------------
# Worker failure handler (task 13.5)
# ---------------------------------------------------------------------------

def mark_job_progress(job_id: str, progress: float, current_stage: str) -> None:
    """Update job progress and current stage in Redis.

    Intended to be called from Celery pipeline tasks as each stage completes.
    Silently tolerates Redis errors to avoid disrupting the pipeline.
    """
    try:
        client = _get_redis()
        state = _read_job(client, job_id) or {}
        state["progress"] = progress
        state["current_stage"] = current_stage
        if state.get("status") not in ("failed", "completed"):
            state["status"] = "processing"
        _write_job(client, job_id, state)
        logger.debug(
            "Job %s progress updated: %.1f%% (%s)", job_id, progress, current_stage
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to update progress for job %s: %s", job_id, exc
        )


def mark_job_failed(job_id: str, error_message: str) -> None:
    """Transition a job to 'failed' status and record the error details.

    Intended to be called from Celery task failure handlers:

        @app.task(bind=True)
        def my_task(self, job_id, ...):
            try:
                ...
            except Exception as exc:
                mark_job_failed(job_id, str(exc))
                raise
    """
    try:
        client = _get_redis()
        state = _read_job(client, job_id) or {}
        state["status"] = "failed"
        state["error"] = error_message
        _write_job(client, job_id, state)
        logger.error("Job %s marked as failed: %s", job_id, error_message)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to mark job %s as failed in Redis: %s", job_id, exc
        )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class JobSubmitRequest(BaseModel):
    video_url: str
    config: JobConfig


class JobSubmitResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# POST /jobs  (tasks 13.1, 13.8)
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobSubmitResponse, status_code=200)
def submit_job(request: JobSubmitRequest, response: Response) -> JobSubmitResponse:
    """Submit a new video processing job.

    - Validates JobConfig via Pydantic (FastAPI returns HTTP 422 on schema error).
    - Stores initial job state in Redis.
    - Enqueues the first pipeline task (transcriber) via Celery.
    - Returns HTTP 503 with Retry-After header if Redis is unavailable.
    """
    # Check Redis / queue availability (task 13.8)
    try:
        client = _get_redis()
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis unavailable on job submission: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Queue unavailable. Please retry later."},
            headers={"Retry-After": "30"},
        )

    job_id = str(uuid.uuid4())

    initial_state: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_stage": "queued",
        "output_urls": None,
        "error": None,
        "video_url": request.video_url,
        "config": request.config.model_dump(),
    }

    try:
        _write_job(client, job_id, initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write initial job state for %s: %s", job_id, exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Queue unavailable. Please retry later."},
            headers={"Retry-After": "30"},
        )

    # Enqueue the first pipeline task (transcribe_task from pipeline.py)
    try:
        celery_app.send_task(
            "app.workers.pipeline.transcribe_task",
            args=[job_id, request.video_url, request.config.model_dump()],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to enqueue transcribe_task for job %s: %s", job_id, exc)
        # Still return the job_id; the job is queued in Redis even if Celery is down

    return JobSubmitResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  (task 13.3)
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    """Return the current status of a job.

    Returns HTTP 404 if the job_id is not found.
    """
    try:
        client = _get_redis()
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis unavailable on status check: %s", exc)
        raise HTTPException(status_code=503, detail="Queue unavailable.")

    state = _read_job(client, job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    return JobStatus(
        job_id=state["job_id"],
        status=state["status"],
        progress=state.get("progress", 0.0),
        current_stage=state.get("current_stage"),
        output_urls=state.get("output_urls"),
        error=state.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /health  (task 13.7)
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Return health status for API, queue, and cloud storage.

    Always returns HTTP 200; degraded components are reflected in the body.
    """
    checks: dict[str, str] = {"api": "ok"}

    # Queue check: ping Redis
    try:
        client = _get_redis()
        checks["queue"] = "ok"
    except Exception:  # noqa: BLE001
        checks["queue"] = "error"

    # Storage check: verify env vars are configured
    storage_ok = bool(
        os.environ.get("AWS_S3_BUCKET")
        or os.environ.get("GCS_BUCKET")
        or os.environ.get("STORAGE_BUCKET")
    )
    checks["storage"] = "ok" if storage_ok else "error"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
