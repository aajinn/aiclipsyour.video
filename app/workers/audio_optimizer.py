"""Audio Optimizer worker: loudness normalization, noise reduction, background music mixing."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from app.models import JobConfig, Segment
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)


class FFmpegAudioError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during audio optimization."""


# ---------------------------------------------------------------------------
# Task 10.1 + 10.2 + 10.4: FFmpeg command builder
# ---------------------------------------------------------------------------

def build_audio_cmd(
    input_path: str,
    output_path: str,
    config: JobConfig,
    music_path: Optional[str] = None,
) -> List[str]:
    """Build the FFmpeg command for audio optimization.

    Always applies loudnorm at -14 LUFS (EBU R128).
    Optionally prepends afftdn (noise reduction) before loudnorm.
    Optionally mixes background music at -12 dB (looped).
    Output: AAC 192k stereo.
    """
    # Build audio filter chain for the speech track
    audio_filters: List[str] = []

    # Task 10.2: noise reduction — afftdn BEFORE loudnorm
    if config.noise_reduction:
        audio_filters.append("afftdn")

    # Task 10.1: loudness normalization — always applied
    audio_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    speech_filter = ",".join(audio_filters)

    if music_path is None:
        # No background music — simple single-input command
        return [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", speech_filter,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            output_path,
        ]

    # Task 10.2: background music mixing
    # Loop music with -stream_loop -1, mix at -12 dB (volume=0.25)
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex",
        f"[0:a]{speech_filter}[speech];[1:a]volume=0.25[music];[speech][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        output_path,
    ]


# ---------------------------------------------------------------------------
# Main AudioOptimizer class
# ---------------------------------------------------------------------------

class AudioOptimizer:
    """Applies audio optimization to an enhanced clip.

    Reads:  jobs/{job_id}/clips/{segment_id}/enhanced.mp4
    Writes: jobs/{job_id}/clips/{segment_id}/audio_optimized.mp4
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    def optimize(self, segment: Segment, config: JobConfig) -> str:
        """Apply audio optimization to a single segment and upload the result.

        Returns the cloud storage key of the audio-optimized clip.
        """
        enhanced_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "enhanced")
        audio_optimized_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "audio_optimized")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "enhanced.mp4")
            output_path = str(tmp / "audio_optimized.mp4")

            logger.info("Downloading enhanced clip %s", enhanced_key)
            self.storage.download_file(enhanced_key, input_path)

            # Task 10.6: resolve background music, skip on failure with warning
            music_path: Optional[str] = None
            if config.background_music_url:
                music_local = str(tmp / "background_music")
                try:
                    self.storage.download_file(config.background_music_url, music_local)
                    if not Path(music_local).exists():
                        raise FileNotFoundError(
                            f"Music file not found after download: {config.background_music_url}"
                        )
                    music_path = music_local
                except Exception:
                    logger.warning(
                        "Background music file not found, skipping: %s",
                        config.background_music_url,
                    )

            cmd = build_audio_cmd(input_path, output_path, config, music_path)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegAudioError(
                    f"FFmpeg audio optimization failed for segment {segment.segment_id} "
                    f"(exit {result.returncode}). stderr: {result.stderr}"
                )

            logger.info("Uploading audio-optimized clip to %s", audio_optimized_key)
            self.storage.upload_file(audio_optimized_key, output_path, content_type="video/mp4")

        return audio_optimized_key


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
        name="app.workers.audio_optimizer.optimize_audio",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def optimize_audio(
        self,
        job_id: str,
        segment_data: dict,
        config_data: dict,
    ) -> dict:
        """Celery task: apply audio optimization to an enhanced clip.

        Returns a dict with ``segment_id`` and ``audio_optimized_key`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        config = JobConfig(**config_data)
        optimizer = AudioOptimizer(job_id=job_id, storage=storage)
        try:
            audio_optimized_key = optimizer.optimize(segment, config)
            return {
                "segment_id": segment.segment_id,
                "audio_optimized_key": audio_optimized_key,
            }
        except FFmpegAudioError as exc:
            logger.error(
                "optimize_audio task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; audio optimizer tasks not registered.")
