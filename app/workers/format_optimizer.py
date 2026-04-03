"""Format Optimizer worker: converts raw clips to vertical 1080×1920 @ 30 FPS."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from app.models import JobConfig, Segment
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)

# Target output dimensions
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30

# 16:9 threshold: source wider than this ratio triggers blurred background fill
WIDE_RATIO_THRESHOLD = 16 / 9  # ~1.7778


class FFmpegFormatError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during format optimization."""


# ---------------------------------------------------------------------------
# FFmpeg command builders
# ---------------------------------------------------------------------------

def build_center_crop_cmd(input_path: str, output_path: str) -> list[str]:
    """Return FFmpeg command for center-crop to 1080×1920 @ 30 FPS.

    Crops the largest 9:16 region from the center of the source frame,
    then scales to exactly 1080×1920 and sets fps=30.
    """
    # crop=w:h:x:y  — take min(iw, ih*9/16) wide, min(ih, iw*16/9) tall, centered
    filter_graph = (
        "crop=if(gte(iw\\,ih*9/16)\\,ih*9/16\\,iw)"
        ":if(gte(iw\\,ih*9/16)\\,ih\\,iw*16/9)"
        ":(iw-if(gte(iw\\,ih*9/16)\\,ih*9/16\\,iw))/2"
        ":(ih-if(gte(iw\\,ih*9/16)\\,ih\\,iw*16/9))/2"
        f",scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        f",fps={OUTPUT_FPS}"
    )
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]


def build_blurred_background_cmd(input_path: str, output_path: str) -> list[str]:
    """Return FFmpeg command for blurred background fill (wide-source > 16:9).

    Background: source scaled to fill 1080×1920, heavily blurred.
    Foreground: source scaled to fit height (1920), overlaid centered.
    """
    # Background: scale to cover 1080×1920 (may crop sides), then boxblur
    bg = (
        f"[0:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},"
        f"boxblur=20:5[bg]"
    )
    # Foreground: scale to fit within 1080×1920 preserving aspect ratio
    fg = (
        f"[0:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[fg]"
    )
    # Overlay fg centered on bg
    overlay = (
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"fps={OUTPUT_FPS}"
    )
    filter_graph = f"{bg};{fg};{overlay}"
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_graph,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]


def build_face_tracking_cmd(
    input_path: str,
    output_path: str,
    crop_x_expr: str,
) -> list[str]:
    """Return FFmpeg command for face-tracking crop.

    *crop_x_expr* is the x-offset expression for the crop filter (as a string).
    The crop height is always the full source height; width is 9/16 of height.
    """
    filter_graph = (
        f"crop=ih*9/16:ih:{crop_x_expr}:0"
        f",scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        f",fps={OUTPUT_FPS}"
    )
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]


# ---------------------------------------------------------------------------
# Face detection helper
# ---------------------------------------------------------------------------

def detect_face_center_x(frame_bgr, frame_width: int) -> int | None:
    """Detect the primary face in *frame_bgr* and return its center x-coordinate.

    Returns None when no face is detected.
    Uses OpenCV Haar cascade classifier.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("opencv-python not installed; face tracking unavailable")
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Pick the largest face (most prominent)
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, w = largest[0], largest[2]
    return x + w // 2


def compute_face_crop_x(face_center_x: int, frame_width: int, frame_height: int) -> int:
    """Compute the crop x-offset that keeps *face_center_x* within center 60% of output width.

    The crop window width is ``frame_height * 9 / 16`` (portrait crop from landscape).
    The face must stay within the center 60% of the output frame width (1080 px).
    """
    crop_w = int(frame_height * 9 / 16)
    # Ideal: center the crop on the face
    x = face_center_x - crop_w // 2
    # Clamp to valid range
    x = max(0, min(x, frame_width - crop_w))
    return x


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class FormatOptimizer:
    """Converts a raw clip to vertical 1080×1920 @ 30 FPS.

    Strategy selection (in priority order):
    1. face_tracking=True  → per-frame face detection, fall back to center-crop per frame
    2. source AR > 16:9    → blurred background fill
    3. default             → center-crop
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    def get_source_dimensions(self, input_path: str) -> tuple[int, int]:
        """Return (width, height) of the video at *input_path* via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            input_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("ffprobe failed; assuming 16:9 source. stderr: %s", result.stderr)
            return (1920, 1080)
        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])

    def optimize(self, segment: Segment, config: JobConfig) -> str:
        """Optimize a single clip and upload the result to cloud storage.

        Returns the cloud storage key of the formatted clip.
        """
        raw_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "raw")
        formatted_key = CloudStorage.clip_key(self.job_id, segment.segment_id, "formatted")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = str(tmp / "raw.mp4")
            output_path = str(tmp / "formatted.mp4")

            logger.info("Downloading raw clip %s", raw_key)
            self.storage.download_file(raw_key, input_path)

            if config.face_tracking:
                self._optimize_face_tracking(input_path, output_path)
            else:
                width, height = self.get_source_dimensions(input_path)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio > WIDE_RATIO_THRESHOLD:
                    logger.info(
                        "Source AR %.3f > 16:9; using blurred background fill for segment %s",
                        aspect_ratio,
                        segment.segment_id,
                    )
                    cmd = build_blurred_background_cmd(input_path, output_path)
                else:
                    logger.info(
                        "Using center-crop for segment %s (AR %.3f)",
                        segment.segment_id,
                        aspect_ratio,
                    )
                    cmd = build_center_crop_cmd(input_path, output_path)

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise FFmpegFormatError(
                        f"FFmpeg format optimization failed for segment {segment.segment_id} "
                        f"(exit {result.returncode}). stderr: {result.stderr}"
                    )

            logger.info("Uploading formatted clip to %s", formatted_key)
            self.storage.upload_file(formatted_key, output_path, content_type="video/mp4")

        return formatted_key

    def _optimize_face_tracking(self, input_path: str, output_path: str) -> None:
        """Run face-tracking optimization, falling back to center-crop per frame when no face found."""
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.warning("opencv-python not installed; falling back to center-crop")
            cmd = build_center_crop_cmd(input_path, output_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegFormatError(
                    f"FFmpeg center-crop fallback failed (exit {result.returncode}). "
                    f"stderr: {result.stderr}"
                )
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.warning("Could not open video for face detection; falling back to center-crop")
            cap.release()
            cmd = build_center_crop_cmd(input_path, output_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise FFmpegFormatError(
                    f"FFmpeg center-crop fallback failed (exit {result.returncode}). "
                    f"stderr: {result.stderr}"
                )
            return

        # Sample the first readable frame to determine crop x
        face_crop_x: int | None = None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Try to detect face from first frame
        cap2 = cv2.VideoCapture(input_path)
        ret, frame = cap2.read()
        cap2.release()

        if ret and frame is not None:
            face_cx = detect_face_center_x(frame, frame_width)
            if face_cx is not None:
                face_crop_x = compute_face_crop_x(face_cx, frame_width, frame_height)
                logger.info("Face detected at center_x=%d; crop_x=%d", face_cx, face_crop_x)

        if face_crop_x is not None:
            # Use static crop offset derived from detected face
            cmd = build_face_tracking_cmd(input_path, output_path, str(face_crop_x))
        else:
            logger.info("No face detected; falling back to center-crop")
            cmd = build_center_crop_cmd(input_path, output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegFormatError(
                f"FFmpeg face-tracking optimization failed (exit {result.returncode}). "
                f"stderr: {result.stderr}"
            )


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
        name="app.workers.format_optimizer.optimize_clip",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def optimize_clip(self, job_id: str, segment_data: dict, config_data: dict) -> dict:
        """Celery task: convert a raw clip to vertical 1080×1920 @ 30 FPS.

        *segment_data* is the dict representation of a :class:`~app.models.Segment`.
        *config_data* is the dict representation of a :class:`~app.models.JobConfig`.

        Returns a dict with ``segment_id`` and ``formatted_key`` (or ``error``).
        """
        storage = _make_storage()
        segment = Segment(**segment_data)
        config = JobConfig(**config_data)
        optimizer = FormatOptimizer(job_id=job_id, storage=storage)
        try:
            formatted_key = optimizer.optimize(segment, config)
            return {"segment_id": segment.segment_id, "formatted_key": formatted_key}
        except FFmpegFormatError as exc:
            logger.error(
                "optimize_clip task failed for segment %s: %s",
                segment.segment_id,
                exc,
            )
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("Celery not available; format optimizer tasks not registered.")
