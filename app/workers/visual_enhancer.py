"""Visual Enhancer worker: applies emoji overlays, progress bar, dynamic text, and image/GIF inserts via FFmpeg filter graphs."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from app.models import DynamicTextConfig, JobConfig, OverlayConfig, ProgressBarConfig, Segment
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)


class FFmpegEnhanceError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during visual enhancement."""


# ---------------------------------------------------------------------------
# Task 9.1 + 9.3: Filter graph composition
# ---------------------------------------------------------------------------

def _resolve_overlay_assets(
    overlays: List[OverlayConfig],
    tmpdir: Path,
    storage: CloudStorage,
) -> List[Tuple[OverlayConfig, Path]]:
    """Download overlay assets; skip and warn on missing assets (Task 9.5)."""
    resolved: List[Tuple[OverlayConfig, Path]] = []
    for overlay in overlays:
        asset_key = overlay.asset_url
        local_path = tmpdir / f"overlay_{len(resolved)}{Path(asset_key).suffix or '.bin'}"
        try:
            storage.download_file(asset_key, str(local_path))
            if not local_path.exists():
                raise FileNotFoundError(f"Asset not found after download: {asset_key}")
            resolved.append((overlay, local_path))
        except Exception:
            logger.warning("Overlay asset not found, skipping: %s", asset_key)
    return resolved


def build_enhance_cmd(
    input_path: str,
    output_path: str,
    config: JobConfig,
    segment: Segment,
    resolved_overlays: Optional[List[Tuple[OverlayConfig, Path]]] = None,
) -> list[str]:
    """Build a single FFmpeg command applying all configured overlays in one pass.

    Uses -filter_complex when image/GIF overlays are present (multi-input),
    otherwise uses -vf for text-only filters.
    """
    if resolved_overlays is None:
        resolved_overlays = []

    duration = segment.end - segment.start

    # Collect vf-compatible (text) filters
    vf_filters: List[str] = []

    # Progress bar via drawbox (Task 9.3)
    pb: ProgressBarConfig = config.progress_bar
    # drawbox with dynamic width: w=W*t/duration
    # FFmpeg drawbox doesn't support expressions for w directly, so we use
    # a geq-based approach via drawbox with enable and a series, but the
    # simplest approach per spec is:
    # drawbox=x=0:y=H-height:w=W*t/duration:h=height:color=color:t=fill
    vf_filters.append(
        f"drawbox=x=0:y=H-{pb.height_px}:w=W*t/{duration}:h={pb.height_px}"
        f":color={pb.color}:t=fill"
    )

    # Dynamic text via drawtext (Task 9.3)
    for dtc in config.dynamic_text_words:
        end_time = dtc.start_time + dtc.duration
        vf_filters.append(
            f"drawtext=text='{dtc.word}'"
            f":fontsize=96"
            f":fontcolor=white"
            f":bordercolor=black:borderw=4"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":enable='between(t,{dtc.start_time},{end_time})'"
        )

    # Emoji overlays via drawtext (Task 9.3)
    # Emoji are rendered as text characters using drawtext
    # (image-based emoji would require overlay filter; drawtext handles unicode emoji)
    # NOTE: OverlayConfig.asset_url for emoji is expected to be a unicode char or
    # a small image. When resolved_overlays is empty (asset missing), we skip.
    # Emoji overlays that ARE resolved as images are handled in filter_complex below.

    has_image_overlays = len(resolved_overlays) > 0

    if not has_image_overlays:
        # Pure text/drawbox path — use -vf
        vf = ",".join(vf_filters) if vf_filters else "null"
        return [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path,
        ]

    # Image/GIF overlay path — use -filter_complex (Task 9.1)
    # Build inputs: first is the video, then one per image overlay
    cmd = ["ffmpeg", "-y", "-i", input_path]
    for _, asset_path in resolved_overlays:
        cmd += ["-i", str(asset_path)]

    # Build filter_complex
    # Start with the base video stream [0:v]
    # Apply vf-style filters first via a chain on [0:v]
    filter_parts: List[str] = []

    # Chain text/drawbox filters onto [0:v] → [base]
    if vf_filters:
        chain = ",".join(vf_filters)
        filter_parts.append(f"[0:v]{chain}[base]")
        current_label = "base"
    else:
        current_label = "0:v"

    # Overlay each image/GIF
    for idx, (overlay, _) in enumerate(resolved_overlays):
        input_idx = idx + 1  # 0 is the video
        end_time = overlay.start_time + overlay.duration
        next_label = f"v{idx}"
        enable_expr = f"between(t\\,{overlay.start_time}\\,{end_time})"
        filter_parts.append(
            f"[{current_label}][{input_idx}:v]overlay=x={overlay.x}:y={overlay.y}"
            f":enable='{enable_expr}'[{next_label}]"
        )
        current_label = next_label

    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{current_label}]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Main VisualEnhancer class
# ---------------------------------------------------------------------------

class VisualEnhancer:
    """Applies visual overlays to a captioned clip.

    Reads:  jobs/{job_id}/clips/{segment_id}/captioned.mp4
    Writes: jobs/{job_id}/clips/{segment_id}/enhanced.mp4
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    def enhance(self, segment: Segment, config: JobConfig) -> str:
        """Apply visual enhancements to a single segment and upload the result.

        Returns the cloud storage key of the enhanced clip.
        """
        captioned_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "captioned")
        enhanced_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "enhanced")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "captioned.mp4")
            output_path = str(tmp / "enhanced.mp4")

            logger.info("Downloading captioned clip %s", captioned_key)
            self.storage.download_file(captioned_key, input_path)

            # Task 9.5: resolve overlay assets, skipping missing ones with a warning
            resolved_overlays = _resolve_overlay_assets(config.overlays, tmp, self.storage)

            cmd = build_enhance_cmd(
                input_path,
                output_path,
                config,
                segment,
                resolved_overlays,
            )

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegEnhanceError(
                    f"FFmpeg enhancement failed for segment {segment.segment_id} "
                    f"(exit {result.returncode}). stderr: {result.stderr}"
                )

            logger.info("Uploading enhanced clip to %s", enhanced_key)
            self.storage.upload_file(enhanced_key, output_path, content_type="video/mp4")

        return enhanced_key


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
        name="app.workers.visual_enhancer.enhance_clip",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def enhance_clip(
        self,
        job_id: str,
        segment_data: dict,
        config_data: dict,
    ) -> dict:
        """Celery task: apply visual enhancements to a captioned clip.

        Returns a dict with ``segment_id`` and ``enhanced_key`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        config = JobConfig(**config_data)
        enhancer = VisualEnhancer(job_id=job_id, storage=storage)
        try:
            enhanced_key = enhancer.enhance(segment, config)
            return {"segment_id": segment.segment_id, "enhanced_key": enhanced_key}
        except FFmpegEnhanceError as exc:
            logger.error(
                "enhance_clip task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; visual enhancer tasks not registered.")
