"""Caption Generator worker: burns styled captions into formatted clips via FFmpeg drawtext."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from app.models import JobConfig, OverlayConfig, Segment, Transcript, WordToken
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)

# Output frame dimensions (must match Format Optimizer output)
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920

# Caption positioning: 65–85% of frame height
CAPTION_Y_MIN = int(FRAME_HEIGHT * 0.65)  # 1248
CAPTION_Y_MAX = int(FRAME_HEIGHT * 0.85)  # 1632
CAPTION_Y_DEFAULT = int(FRAME_HEIGHT * 0.75)  # 1440 — midpoint

# Minimum font size at 1080×1920
MIN_FONT_SIZE = 48

# Caption chunk size bounds
CHUNK_MIN_WORDS = 2
CHUNK_MAX_WORDS = 4


class FFmpegCaptionError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during caption generation."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CaptionChunk:
    """A 2–4 word display unit with timing."""

    def __init__(self, words: List[WordToken]) -> None:
        if not words:
            raise ValueError("CaptionChunk requires at least one word")
        self.words = words
        self.text = " ".join(w.word for w in words)
        self.start = words[0].start
        self.end = words[-1].end

    def __repr__(self) -> str:  # pragma: no cover
        return f"CaptionChunk({self.text!r}, {self.start:.3f}–{self.end:.3f})"


# ---------------------------------------------------------------------------
# Task 7.1: Transcript chunking
# ---------------------------------------------------------------------------

def chunk_transcript(transcript: Transcript) -> List[CaptionChunk]:
    """Break transcript words into 2–4 word display units with timing.

    Each chunk contains CHUNK_MIN_WORDS to CHUNK_MAX_WORDS words.
    The chunk start time is the first word's start; end time is the last word's end.
    If fewer than CHUNK_MIN_WORDS words remain at the end, they are merged into
    the previous chunk (up to CHUNK_MAX_WORDS) or emitted as-is when no previous
    chunk exists.
    """
    words = transcript.words
    if not words:
        return []

    chunks: List[CaptionChunk] = []
    i = 0
    while i < len(words):
        remaining = len(words) - i
        # If only 1 word left and we already have a chunk, merge into previous
        if remaining < CHUNK_MIN_WORDS and chunks:
            prev_words = chunks[-1].words
            merged = prev_words + list(words[i:])
            # Only merge if it doesn't exceed max; otherwise emit as standalone
            if len(merged) <= CHUNK_MAX_WORDS:
                chunks[-1] = CaptionChunk(merged)
                i += remaining
                continue
        # Normal chunk: take up to CHUNK_MAX_WORDS words
        end = min(i + CHUNK_MAX_WORDS, len(words))
        chunks.append(CaptionChunk(list(words[i:end])))
        i = end

    return chunks


# ---------------------------------------------------------------------------
# Task 7.5: Caption style presets
# ---------------------------------------------------------------------------

def _style_params(caption_style: str) -> dict:
    """Return FFmpeg drawtext parameter dict for the given style preset."""
    if caption_style == "highlight":
        return {
            "fontcolor": "yellow",
            "shadowcolor": "black",
            "shadowx": "2",
            "shadowy": "2",
        }
    # default
    return {
        "fontcolor": "white",
        "bordercolor": "black",
        "borderw": "3",
    }


# ---------------------------------------------------------------------------
# Task 7.7: Overlap detection
# ---------------------------------------------------------------------------

def compute_caption_y(
    default_y: int,
    chunk: CaptionChunk,
    overlays: List[OverlayConfig],
    font_size: int = MIN_FONT_SIZE,
) -> int:
    """Return the Y position for a caption chunk, shifted upward if it overlaps an overlay.

    Overlap is detected when:
    - The caption's time window intersects the overlay's time window, AND
    - The caption's vertical extent (default_y to default_y + font_size) intersects
      the overlay's vertical extent (overlay.y to overlay.y + some height).

    When overlap is detected, the caption is shifted upward by font_size + a small margin.
    """
    y = default_y
    caption_top = y
    caption_bottom = y + font_size

    for overlay in overlays:
        overlay_end = overlay.start_time + overlay.duration
        # Check time overlap
        if chunk.end <= overlay.start_time or chunk.start >= overlay_end:
            continue
        # Check vertical overlap (treat overlay as a region from overlay.y downward)
        # We use a conservative height estimate of font_size for the overlay region
        overlay_top = overlay.y
        overlay_bottom = overlay.y + font_size

        if caption_bottom > overlay_top and caption_top < overlay_bottom:
            # Shift caption upward above the overlay region
            y = overlay_top - font_size - 8  # 8px margin
            # Clamp to valid range
            y = max(CAPTION_Y_MIN, y)
            caption_top = y
            caption_bottom = y + font_size

    return y


# ---------------------------------------------------------------------------
# Task 7.3: FFmpeg drawtext command builder
# ---------------------------------------------------------------------------

def build_caption_cmd(
    input_path: str,
    output_path: str,
    chunks: List[CaptionChunk],
    caption_style: str = "default",
    overlays: Optional[List[OverlayConfig]] = None,
    font_size: int = MIN_FONT_SIZE,
) -> list[str]:
    """Build an FFmpeg command that burns captions into the video via drawtext filters.

    Each chunk produces one drawtext filter with:
    - enable='between(t,start,end)' for timing
    - fontsize >= 48
    - bold text (fontstyle=Bold or font=<bold-font>)
    - y position between 65% and 85% of frame height
    - style-specific color/shadow/border params
    """
    if overlays is None:
        overlays = []

    style = _style_params(caption_style)

    filter_parts: List[str] = []
    for chunk in chunks:
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, overlays, font_size)

        # Build drawtext option string
        opts: List[str] = [
            f"text='{chunk.text}'",
            f"fontsize={font_size}",
            f"fontcolor={style.get('fontcolor', 'white')}",
            f"x=(w-text_w)/2",
            f"y={y}",
            f"enable='between(t,{chunk.start},{chunk.end})'",
            "font=Arial:Bold",
        ]

        if "bordercolor" in style:
            opts.append(f"bordercolor={style['bordercolor']}")
            opts.append(f"borderw={style['borderw']}")

        if "shadowcolor" in style:
            opts.append(f"shadowcolor={style['shadowcolor']}")
            opts.append(f"shadowx={style['shadowx']}")
            opts.append(f"shadowy={style['shadowy']}")

        filter_parts.append("drawtext=" + ":".join(opts))

    vf = ",".join(filter_parts) if filter_parts else "null"

    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]


# ---------------------------------------------------------------------------
# Main CaptionGenerator class
# ---------------------------------------------------------------------------

class CaptionGenerator:
    """Burns styled captions into a formatted clip.

    Reads:  jobs/{job_id}/clips/{segment_id}/formatted.mp4
    Writes: jobs/{job_id}/clips/{segment_id}/captioned.mp4
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    def generate(
        self,
        segment: Segment,
        transcript: Transcript,
        config: JobConfig,
    ) -> str:
        """Generate captions for a single segment and upload the result.

        Returns the cloud storage key of the captioned clip.
        """
        formatted_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "formatted")
        captioned_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "captioned")

        # Filter transcript words to this segment's time range
        segment_words = [
            w for w in transcript.words
            if w.start >= segment.start and w.end <= segment.end
        ]
        segment_transcript = Transcript(
            job_id=transcript.job_id,
            words=segment_words,
            is_empty=len(segment_words) == 0,
        )

        chunks = chunk_transcript(segment_transcript)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "formatted.mp4")
            output_path = str(tmp / "captioned.mp4")

            logger.info("Downloading formatted clip %s", formatted_key)
            self.storage.download_file(formatted_key, input_path)

            cmd = build_caption_cmd(
                input_path,
                output_path,
                chunks,
                caption_style=config.caption_style,
                overlays=config.overlays,
            )

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegCaptionError(
                    f"FFmpeg caption generation failed for segment {segment.segment_id} "
                    f"(exit {result.returncode}). stderr: {result.stderr}"
                )

            logger.info("Uploading captioned clip to %s", captioned_key)
            self.storage.upload_file(captioned_key, output_path, content_type="video/mp4")

        return captioned_key


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
        name="app.workers.caption_generator.generate_captions",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def generate_captions(
        self,
        job_id: str,
        segment_data: dict,
        transcript_data: dict,
        config_data: dict,
    ) -> dict:
        """Celery task: burn captions into a formatted clip.

        Returns a dict with ``segment_id`` and ``captioned_key`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        transcript = Transcript(**transcript_data)
        config = JobConfig(**config_data)
        generator = CaptionGenerator(job_id=job_id, storage=storage)
        try:
            captioned_key = generator.generate(segment, transcript, config)
            return {"segment_id": segment.segment_id, "captioned_key": captioned_key}
        except FFmpegCaptionError as exc:
            logger.error(
                "generate_captions task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; caption generator tasks not registered.")
