"""Pipeline orchestration: Celery task chain wiring all pipeline stages together.

Flow:
  transcribe_task → highlight_detect_task → group(process_segment_task, ...) → finalize_job_task

Requirements: 9.5, 10.1, 10.2, 9.3
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage progress weights (cumulative % at end of each stage)
# ---------------------------------------------------------------------------

PROGRESS_TRANSCRIPTION = 10.0       # after transcription completes
PROGRESS_HIGHLIGHT_DETECTION = 20.0  # after highlight detection completes
PROGRESS_CLIP_EXTRACTION = 30.0      # after all clips are extracted
PROGRESS_COMPLETED = 100.0           # after finalize


def _per_segment_progress(segment_index: int, total_segments: int) -> float:
    """Return cumulative progress after processing segment at *segment_index* (0-based).

    Per-segment processing spans 30% → 100% distributed evenly across segments.
    """
    if total_segments == 0:
        return PROGRESS_COMPLETED
    per_segment = (PROGRESS_COMPLETED - PROGRESS_CLIP_EXTRACTION) / total_segments
    return PROGRESS_CLIP_EXTRACTION + per_segment * (segment_index + 1)


# ---------------------------------------------------------------------------
# Storage factory
# ---------------------------------------------------------------------------

def _make_storage():
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
    from celery import chain, chord, group

    from app.api.main import mark_job_progress
    from app.celery_app import celery_app
    from app.models import JobConfig, Segment, Transcript
    from app.storage.base import CloudStorage
    from app.workers.audio_optimizer import AudioOptimizer
    from app.workers.caption_generator import CaptionGenerator
    from app.workers.clip_extractor import ClipExtractor
    from app.workers.export_engine import ExportEngine
    from app.workers.format_optimizer import FormatOptimizer
    from app.workers.highlight_detector import HighlightDetector
    from app.workers.transcriber import Transcriber
    from app.workers.visual_enhancer import VisualEnhancer

    # ------------------------------------------------------------------
    # Task 1: Transcribe
    # ------------------------------------------------------------------

    @celery_app.task(
        name="app.workers.pipeline.transcribe_task",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def transcribe_task(self, job_id: str, video_url: str, config_data: dict) -> dict:
        """Run Transcriber: extract audio + transcribe → produces transcript.json.

        Chains to highlight_detect_task on success.
        Updates Redis progress to 10% on completion.
        """
        mark_job_progress(job_id, 0.0, "transcription")
        storage = _make_storage()
        config = JobConfig(**config_data)

        try:
            transcriber = Transcriber(job_id=job_id, storage=storage)

            # Extract audio from source video
            video_key = f"jobs/{job_id}/source.mp4"
            audio_key = transcriber.extract_audio(video_key)

            # Transcribe audio
            transcript = transcriber.transcribe(audio_key, config)

            # Persist transcript.json to cloud storage
            transcript_key = CloudStorage.transcript_key(job_id)
            transcript_bytes = json.dumps(transcript.model_dump()).encode("utf-8")
            storage.upload(transcript_key, transcript_bytes, content_type="application/json")

            mark_job_progress(job_id, PROGRESS_TRANSCRIPTION, "transcription")
            logger.info("transcribe_task completed for job %s", job_id)

            # Chain to highlight detection
            highlight_detect_task.delay(job_id, config_data)

            return {"job_id": job_id, "transcript_key": transcript_key}

        except Exception as exc:
            logger.error("transcribe_task failed for job %s: %s", job_id, exc)
            _mark_failed(job_id, str(exc))
            raise self.retry(exc=exc)

    # ------------------------------------------------------------------
    # Task 2: Highlight Detection
    # ------------------------------------------------------------------

    @celery_app.task(
        name="app.workers.pipeline.highlight_detect_task",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def highlight_detect_task(self, job_id: str, config_data: dict) -> dict:
        """Run HighlightDetector: score segments → produces segments.json.

        Fans out to process_segment_task for each segment.
        Updates Redis progress to 20% on completion.
        """
        mark_job_progress(job_id, PROGRESS_TRANSCRIPTION, "highlight_detection")
        storage = _make_storage()

        try:
            detector = HighlightDetector(job_id=job_id, storage=storage)
            segments = detector.run()

            mark_job_progress(job_id, PROGRESS_HIGHLIGHT_DETECTION, "highlight_detection")
            logger.info(
                "highlight_detect_task completed for job %s: %d segments",
                job_id,
                len(segments),
            )

            if not segments:
                # No segments found — finalize immediately with empty results
                finalize_job_task.delay(job_id, [])
                return {"job_id": job_id, "segment_count": 0}

            # Fan out: extract clips then process each segment
            total = len(segments)

            # First extract all clips (sequential per segment, parallel across segments)
            extractor = ClipExtractor(job_id=job_id, storage=storage)
            clip_results: dict[str, str | None] = extractor.extract_all(segments)

            mark_job_progress(job_id, PROGRESS_CLIP_EXTRACTION, "clip_extraction")
            logger.info("Clip extraction complete for job %s", job_id)

            # Fan out per-segment processing as a Celery group
            segment_tasks = group(
                process_segment_task.s(
                    job_id,
                    seg.model_dump(),
                    config_data,
                    idx,
                    total,
                )
                for idx, seg in enumerate(segments)
                if clip_results.get(seg.segment_id) is not None
            )

            # Use chord: run all segment tasks, then finalize
            chord(segment_tasks)(finalize_job_task.s(job_id))

            return {"job_id": job_id, "segment_count": total}

        except Exception as exc:
            logger.error("highlight_detect_task failed for job %s: %s", job_id, exc)
            _mark_failed(job_id, str(exc))
            raise self.retry(exc=exc)

    # ------------------------------------------------------------------
    # Task 3: Per-segment processing
    # ------------------------------------------------------------------

    @celery_app.task(
        name="app.workers.pipeline.process_segment_task",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def process_segment_task(
        self,
        job_id: str,
        segment_data: dict,
        config_data: dict,
        segment_index: int,
        total_segments: int,
    ) -> dict:
        """Run the full per-segment chain: Format → Caption → Visual → Audio → Export.

        Updates Redis progress as each segment completes.
        Returns a dict with segment_id and signed_url (or error).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        config = JobConfig(**config_data)

        try:
            # Read transcript for caption generation
            transcript_key = CloudStorage.transcript_key(job_id)
            raw = storage.download(transcript_key)
            transcript = Transcript(**json.loads(raw.decode("utf-8")))

            # 1. Format Optimizer
            format_optimizer = FormatOptimizer(job_id=job_id, storage=storage)
            format_optimizer.optimize(segment, config)

            # 2. Caption Generator
            caption_gen = CaptionGenerator(job_id=job_id, storage=storage)
            caption_gen.generate(segment, transcript, config)

            # 3. Visual Enhancer
            visual_enhancer = VisualEnhancer(job_id=job_id, storage=storage)
            visual_enhancer.enhance(segment, config)

            # 4. Audio Optimizer
            audio_optimizer = AudioOptimizer(job_id=job_id, storage=storage)
            audio_optimizer.optimize(segment, config)

            # 5. Export Engine
            export_engine = ExportEngine(job_id=job_id, storage=storage)
            signed_url = export_engine.export(segment)

            # Update progress
            progress = _per_segment_progress(segment_index, total_segments)
            mark_job_progress(job_id, progress, "per_segment_processing")

            logger.info(
                "process_segment_task completed for job %s segment %s",
                job_id,
                segment.segment_id,
            )
            return {"segment_id": segment.segment_id, "signed_url": signed_url}

        except Exception as exc:
            logger.error(
                "process_segment_task failed for job %s segment %s: %s",
                job_id,
                segment.segment_id,
                exc,
            )
            return {"segment_id": segment.segment_id, "error": str(exc)}

    # ------------------------------------------------------------------
    # Task 4: Finalize
    # ------------------------------------------------------------------

    @celery_app.task(
        name="app.workers.pipeline.finalize_job_task",
        bind=True,
    )
    def finalize_job_task(self, results: list[dict] | None, job_id: str) -> dict:
        """Collect signed URLs from segment results and mark job completed.

        *results* is the list of dicts returned by process_segment_task (from chord).
        When called directly (no segments), *results* may be an empty list.
        """
        if results is None:
            results = []

        output_urls = [
            r["signed_url"]
            for r in results
            if isinstance(r, dict) and "signed_url" in r
        ]

        try:
            from app.api.main import _get_redis, _job_key, _read_job, _write_job

            client = _get_redis()
            state = _read_job(client, job_id) or {}
            state["status"] = "completed"
            state["progress"] = PROGRESS_COMPLETED
            state["current_stage"] = "completed"
            state["output_urls"] = output_urls
            _write_job(client, job_id, state)
            logger.info(
                "finalize_job_task: job %s completed with %d output URLs",
                job_id,
                len(output_urls),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("finalize_job_task failed to update Redis for job %s: %s", job_id, exc)

        return {"job_id": job_id, "output_urls": output_urls}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_failed(job_id: str, error: str) -> None:
        """Mark job as failed in Redis (best-effort)."""
        try:
            from app.api.main import mark_job_failed
            mark_job_failed(job_id, error)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not mark job %s as failed: %s", job_id, exc)

except ImportError as _import_err:
    logger.debug("Celery not available; pipeline tasks not registered. Error: %s", _import_err)
