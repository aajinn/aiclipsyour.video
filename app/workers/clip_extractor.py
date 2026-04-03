"""Clip Extractor worker: per-segment FFmpeg extraction pipeline stage."""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

from app.models import Segment
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)


class FFmpegExtractionError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during clip extraction."""


def build_stream_copy_cmd(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
) -> list[str]:
    """Return the FFmpeg command for stream-copy extraction.

    Uses ``-ss`` / ``-to`` before ``-i`` for fast seeking, then ``-c copy``
    to avoid re-encoding.
    """
    return [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c", "copy",
        output_path,
    ]


def build_reencode_cmd(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
) -> list[str]:
    """Return the FFmpeg command for H.264 + AAC re-encode extraction."""
    return [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]


class ClipExtractor:
    """Reads segments.json and extracts one clip per segment via FFmpeg.

    Extraction strategy:
    1. Try stream-copy (``-c copy``) — fast, lossless.
    2. On non-zero exit, fall back to H.264 + AAC re-encode.
    3. On re-encode failure, log stderr, mark segment failed, continue.
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_clip(self, segment: Segment) -> str:
        """Extract a single clip for *segment* and upload to cloud storage.

        Returns the cloud storage key of the uploaded clip.

        Raises:
            FFmpegExtractionError: if both stream-copy and re-encode fail.
        """
        source_key = f"jobs/{self.job_id}/source.mp4"
        output_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "raw")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "source.mp4")
            output_path = str(tmp / "raw.mp4")

            logger.info(
                "Downloading source video %s for segment %s",
                source_key,
                segment.segment_id,
            )
            self.storage.download_file(source_key, input_path)

            # Attempt 1: stream-copy
            cmd = build_stream_copy_cmd(input_path, output_path, segment.start, segment.end)
            logger.debug("Running stream-copy FFmpeg command: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(
                    "Stream-copy failed (exit %d) for segment %s; stderr:\n%s",
                    result.returncode,
                    segment.segment_id,
                    result.stderr,
                )
                # Attempt 2: re-encode fallback
                cmd = build_reencode_cmd(input_path, output_path, segment.start, segment.end)
                logger.debug("Running re-encode FFmpeg command: %s", cmd)
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(
                        "Re-encode also failed (exit %d) for segment %s; stderr:\n%s",
                        result.returncode,
                        segment.segment_id,
                        result.stderr,
                    )
                    raise FFmpegExtractionError(
                        f"FFmpeg extraction failed for segment {segment.segment_id} "
                        f"(exit {result.returncode}). stderr: {result.stderr}"
                    )

            logger.info("Uploading clip to %s", output_key)
            self.storage.upload_file(output_key, output_path, content_type="video/mp4")

        return output_key

    def extract_all(self, segments: List[Segment]) -> dict[str, str | None]:
        """Extract clips for all *segments*, isolating per-segment failures.

        Returns a dict mapping ``segment_id`` → cloud storage key (or ``None``
        when that segment failed).
        """
        results: dict[str, str | None] = {}
        for segment in segments:
            try:
                key = self.extract_clip(segment)
                results[segment.segment_id] = key
            except FFmpegExtractionError as exc:
                logger.error(
                    "Segment %s failed; marking as failed and continuing: %s",
                    segment.segment_id,
                    exc,
                )
                self._mark_segment_failed(segment.segment_id, str(exc))
                results[segment.segment_id] = None
        return results

    def run(self) -> dict[str, str | None]:
        """Full pipeline: read segments.json → extract all clips.

        Returns the per-segment result dict (segment_id → key or None).
        """
        segments_key = CloudStorage.segments_key(self.job_id)
        logger.info("Reading segments from %s", segments_key)
        raw = self.storage.download(segments_key)
        segments_data = json.loads(raw.decode("utf-8"))
        segments = [Segment(**s) for s in segments_data]
        return self.extract_all(segments)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_segment_failed(self, segment_id: str, error: str) -> None:
        """Persist a failure record for *segment_id* in cloud storage."""
        key = f"jobs/{self.job_id}/clips/{segment_id}/error.json"
        data = json.dumps({"segment_id": segment_id, "error": error}).encode("utf-8")
        try:
            self.storage.upload(key, data, content_type="application/json")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not persist failure record for segment %s: %s",
                segment_id,
                exc,
            )


# ---------------------------------------------------------------------------
# Storage factory (shared with other workers)
# ---------------------------------------------------------------------------

def _make_storage() -> CloudStorage:
    import os

    from app.storage.gcs import GCSStorage
    from app.storage.s3 import S3Storage

    storage_backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
    if storage_backend == "gcs":
        bucket = os.environ.get("GCS_BUCKET", "video-processing-engine")
        return GCSStorage(bucket=bucket)
    bucket = os.environ.get("S3_BUCKET", "video-processing-engine")
    return S3Storage(bucket=bucket)


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------

try:
    from celery import group

    from app.celery_app import celery_app

    @celery_app.task(
        name="app.workers.clip_extractor.extract_clip",
        bind=True,
        max_retries=0,  # failure isolation: do not retry; mark segment failed instead
    )
    def extract_clip(self, job_id: str, segment_data: dict) -> dict:
        """Celery task: extract a single clip for one segment.

        *segment_data* is the dict representation of a :class:`~app.models.Segment`.

        Returns a dict with ``segment_id`` and ``clip_key`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        extractor = ClipExtractor(job_id=job_id, storage=storage)
        try:
            clip_key = extractor.extract_clip(segment)
            return {"segment_id": segment.segment_id, "clip_key": clip_key}
        except FFmpegExtractionError as exc:
            logger.error(
                "extract_clip task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            extractor._mark_segment_failed(segment.segment_id, str(exc))
            return {"segment_id": segment.segment_id, "error": str(exc)}

    @celery_app.task(
        name="app.workers.clip_extractor.extract_clips",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def extract_clips(self, job_id: str) -> dict:
        """Celery task: fan out one ``extract_clip`` subtask per segment.

        Reads ``jobs/{job_id}/segments.json`` from cloud storage and dispatches
        a :func:`extract_clip` subtask for each segment using a Celery group
        (parallel fan-out).

        Returns a dict with ``job_id`` and ``segment_count``.
        """
        storage = _make_storage()
        try:
            segments_key = CloudStorage.segments_key(job_id)
            raw = storage.download(segments_key)
            segments_data = json.loads(raw.decode("utf-8"))

            subtasks = group(
                extract_clip.s(job_id, seg_data) for seg_data in segments_data
            )
            subtasks.apply_async()

            return {"job_id": job_id, "segment_count": len(segments_data)}
        except Exception as exc:
            logger.error("extract_clips failed for job %s: %s", job_id, exc)
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; clip extractor tasks not registered.")
