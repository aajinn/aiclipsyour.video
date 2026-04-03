"""Export Engine worker: final H.264/AAC render, 60s trim, upload, signed URL."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

from app.models import Segment
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)

MAX_DURATION_SECONDS = 60


class ExportError(Exception):
    """Raised when FFmpeg final render fails."""


# ---------------------------------------------------------------------------
# Task 11.1 + 11.3: FFmpeg command builder
# ---------------------------------------------------------------------------

def build_export_cmd(
    input_path: str,
    output_path: str,
    duration: float,
) -> List[str]:
    """Build the FFmpeg command for final export.

    Produces H.264 + AAC, 1080x1920, 30 FPS, 8 Mbps video, 192k audio,
    with -movflags +faststart for web streaming.
    Trims to 60 seconds if duration > 60s (Task 11.3).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:v", "8M",
        "-b:a", "192k",
        "-r", "30",
        "-s", "1080x1920",
        "-movflags", "+faststart",
    ]

    # Task 11.3: trim to 60 seconds if segment is longer
    if duration > MAX_DURATION_SECONDS:
        cmd += ["-t", "60"]

    cmd.append(output_path)
    return cmd


# ---------------------------------------------------------------------------
# Main ExportEngine class
# ---------------------------------------------------------------------------

class ExportEngine:
    """Renders the final clip and uploads it to cloud storage.

    Reads:  jobs/{job_id}/clips/{segment_id}/audio_optimized.mp4
    Writes: jobs/{job_id}/clips/{segment_id}/final.mp4
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    def export(self, segment: Segment) -> str:
        """Render and upload the final clip for a single segment.

        Returns a signed URL valid for at least 24 hours.
        Raises ExportError on FFmpeg failure (intermediate files are retained).
        """
        audio_optimized_key = CloudStorage.clip_key(
            self.job_id, segment.segment_id, "audio_optimized"
        )
        final_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "final")
        duration = segment.end - segment.start

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "audio_optimized.mp4")
            output_path = str(tmp / "final.mp4")

            logger.info("Downloading audio-optimized clip %s", audio_optimized_key)
            self.storage.download_file(audio_optimized_key, input_path)

            cmd = build_export_cmd(input_path, output_path, duration)

            # Task 11.6: on failure, do NOT upload; raise ExportError with stderr
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ExportError(
                    f"FFmpeg final render failed for segment {segment.segment_id} "
                    f"(exit {result.returncode}). stderr: {result.stderr}"
                )

            logger.info("Uploading final clip to %s", final_key)
            self.storage.upload_file(final_key, output_path, content_type="video/mp4")

        # Task 11.5: generate signed URL valid for >= 24 hours
        signed_url = self.storage.generate_signed_url(final_key, expiration_seconds=86400)
        logger.info("Generated signed URL for %s", final_key)
        return signed_url


# ---------------------------------------------------------------------------
# Storage factory
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
# Celery task
# ---------------------------------------------------------------------------

try:
    from app.celery_app import celery_app

    @celery_app.task(
        name="app.workers.export_engine.export_clip",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def export_clip(
        self,
        job_id: str,
        segment_data: dict,
    ) -> dict:
        """Celery task: render and upload the final clip for a segment.

        Returns a dict with ``segment_id`` and ``signed_url`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        engine = ExportEngine(job_id=job_id, storage=storage)
        try:
            signed_url = engine.export(segment)
            return {
                "segment_id": segment.segment_id,
                "signed_url": signed_url,
            }
        except ExportError as exc:
            logger.error(
                "export_clip task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; export engine tasks not registered.")
