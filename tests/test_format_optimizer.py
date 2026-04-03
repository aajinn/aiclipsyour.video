"""Unit tests for FormatOptimizer (Tasks 6.1, 6.3, 6.5)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models import JobConfig, Segment
from app.workers.format_optimizer import (
    FFmpegFormatError,
    FormatOptimizer,
    OUTPUT_FPS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    WIDE_RATIO_THRESHOLD,
    build_blurred_background_cmd,
    build_center_crop_cmd,
    build_face_tracking_cmd,
    compute_face_crop_x,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(job_id: str = "job-fo-test") -> FormatOptimizer:
    storage = MagicMock()
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    return FormatOptimizer(job_id=job_id, storage=storage)


def _make_segment(
    segment_id: str = "seg-1",
    job_id: str = "job-fo-test",
) -> Segment:
    return Segment(
        segment_id=segment_id,
        job_id=job_id,
        start=0.0,
        end=30.0,
        score=0.8,
        llm_virality_score=0.7,
        rank=1,
    )


def _make_config(face_tracking: bool = False) -> JobConfig:
    return JobConfig(face_tracking=face_tracking)


def _ok_result() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    r.stdout = "1920,1080\n"
    return r


def _fail_result(stderr: str = "ffmpeg error") -> MagicMock:
    r = MagicMock()
    r.returncode = 1
    r.stderr = stderr
    r.stdout = ""
    return r


def _probe_result(width: int, height: int) -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    r.stdout = f"{width},{height}\n"
    return r


# ---------------------------------------------------------------------------
# Task 6.1: center-crop command
# ---------------------------------------------------------------------------

class TestBuildCenterCropCmd:
    def test_output_dimensions_in_filter(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        assert f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}" in vf

    def test_fps_in_filter(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        assert f"fps={OUTPUT_FPS}" in vf

    def test_crop_filter_present(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        assert "crop=" in vf

    def test_uses_libx264(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_uses_aac(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        assert "-c:a" in cmd
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    def test_input_output_paths(self):
        cmd = build_center_crop_cmd("/input.mp4", "/output.mp4")
        assert "/input.mp4" in cmd
        assert "/output.mp4" in cmd

    def test_overwrite_flag(self):
        cmd = build_center_crop_cmd("/in.mp4", "/out.mp4")
        assert "-y" in cmd


# ---------------------------------------------------------------------------
# Task 6.3: blurred background command
# ---------------------------------------------------------------------------

class TestBuildBlurredBackgroundCmd:
    def test_uses_filter_complex(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        assert "-filter_complex" in cmd

    def test_boxblur_in_filter(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "boxblur" in fc

    def test_overlay_in_filter(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "overlay" in fc

    def test_output_dimensions_in_filter(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert f"{OUTPUT_WIDTH}" in fc
        assert f"{OUTPUT_HEIGHT}" in fc

    def test_fps_in_filter(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert f"fps={OUTPUT_FPS}" in fc

    def test_uses_libx264(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_overwrite_flag(self):
        cmd = build_blurred_background_cmd("/in.mp4", "/out.mp4")
        assert "-y" in cmd


# ---------------------------------------------------------------------------
# Task 6.5: face-tracking command
# ---------------------------------------------------------------------------

class TestBuildFaceTrackingCmd:
    def test_crop_filter_present(self):
        cmd = build_face_tracking_cmd("/in.mp4", "/out.mp4", "100")
        vf = cmd[cmd.index("-vf") + 1]
        assert "crop=" in vf

    def test_crop_x_expr_in_filter(self):
        cmd = build_face_tracking_cmd("/in.mp4", "/out.mp4", "200")
        vf = cmd[cmd.index("-vf") + 1]
        assert "200" in vf

    def test_output_dimensions_in_filter(self):
        cmd = build_face_tracking_cmd("/in.mp4", "/out.mp4", "0")
        vf = cmd[cmd.index("-vf") + 1]
        assert f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}" in vf

    def test_fps_in_filter(self):
        cmd = build_face_tracking_cmd("/in.mp4", "/out.mp4", "0")
        vf = cmd[cmd.index("-vf") + 1]
        assert f"fps={OUTPUT_FPS}" in vf


# ---------------------------------------------------------------------------
# Task 6.5: compute_face_crop_x
# ---------------------------------------------------------------------------

class TestComputeFaceCropX:
    def test_face_centered_gives_centered_crop(self):
        # 1920x1080 source; face at center (960); crop_w = 1080*9/16 = 607
        x = compute_face_crop_x(960, 1920, 1080)
        crop_w = int(1080 * 9 / 16)
        assert x == max(0, min(960 - crop_w // 2, 1920 - crop_w))

    def test_face_at_left_edge_clamps_to_zero(self):
        x = compute_face_crop_x(0, 1920, 1080)
        assert x == 0

    def test_face_at_right_edge_clamps(self):
        crop_w = int(1080 * 9 / 16)
        x = compute_face_crop_x(1920, 1920, 1080)
        assert x == 1920 - crop_w

    def test_crop_x_non_negative(self):
        x = compute_face_crop_x(50, 1920, 1080)
        assert x >= 0

    def test_crop_window_stays_within_frame(self):
        crop_w = int(1080 * 9 / 16)
        x = compute_face_crop_x(1800, 1920, 1080)
        assert x + crop_w <= 1920


# ---------------------------------------------------------------------------
# Task 6.1: FormatOptimizer.optimize — default center-crop path
# ---------------------------------------------------------------------------

class TestFormatOptimizerCenterCrop:
    def test_center_crop_uploads_to_formatted_key(self):
        optimizer = _make_optimizer("job-cc-1")
        segment = _make_segment("seg-cc-1", "job-cc-1")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _ok_result()]):
            key = optimizer.optimize(segment, config)

        assert key == "jobs/job-cc-1/clips/seg-cc-1/formatted.mp4"
        optimizer.storage.upload_file.assert_called_once()
        assert optimizer.storage.upload_file.call_args[0][0] == key

    def test_center_crop_downloads_raw_from_correct_key(self):
        optimizer = _make_optimizer("job-cc-2")
        segment = _make_segment("seg-cc-2", "job-cc-2")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _ok_result()]):
            optimizer.optimize(segment, config)

        download_key = optimizer.storage.download_file.call_args[0][0]
        assert download_key == "jobs/job-cc-2/clips/seg-cc-2/raw.mp4"

    def test_center_crop_ffmpeg_uses_vf_flag(self):
        optimizer = _make_optimizer()
        segment = _make_segment()
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        # Second call is the FFmpeg encode call
        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "-vf" in ffmpeg_cmd

    def test_ffmpeg_failure_raises_format_error(self):
        optimizer = _make_optimizer()
        segment = _make_segment()
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _fail_result("bad input")]):
            with pytest.raises(FFmpegFormatError):
                optimizer.optimize(segment, config)

    def test_ffmpeg_failure_no_upload(self):
        optimizer = _make_optimizer()
        segment = _make_segment()
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _fail_result()]):
            with pytest.raises(FFmpegFormatError):
                optimizer.optimize(segment, config)

        optimizer.storage.upload_file.assert_not_called()


# ---------------------------------------------------------------------------
# Task 6.3: FormatOptimizer.optimize — blurred background path
# ---------------------------------------------------------------------------

class TestFormatOptimizerBlurredBackground:
    def test_wide_source_uses_filter_complex(self):
        """Source wider than 16:9 should trigger blurred background (filter_complex)."""
        optimizer = _make_optimizer("job-bb-1")
        segment = _make_segment("seg-bb-1", "job-bb-1")
        config = _make_config(face_tracking=False)

        # 2560x1080 is ~2.37:1, wider than 16:9
        with patch("subprocess.run", side_effect=[_probe_result(2560, 1080), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "-filter_complex" in ffmpeg_cmd

    def test_wide_source_filter_contains_boxblur(self):
        optimizer = _make_optimizer("job-bb-2")
        segment = _make_segment("seg-bb-2", "job-bb-2")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(2560, 1080), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        fc = ffmpeg_cmd[ffmpeg_cmd.index("-filter_complex") + 1]
        assert "boxblur" in fc

    def test_wide_source_filter_contains_overlay(self):
        optimizer = _make_optimizer("job-bb-3")
        segment = _make_segment("seg-bb-3", "job-bb-3")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(2560, 1080), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        fc = ffmpeg_cmd[ffmpeg_cmd.index("-filter_complex") + 1]
        assert "overlay" in fc

    def test_exactly_16_9_uses_center_crop_not_blur(self):
        """Exactly 16:9 should NOT trigger blurred background."""
        optimizer = _make_optimizer("job-bb-4")
        segment = _make_segment("seg-bb-4", "job-bb-4")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1920, 1080), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "-filter_complex" not in ffmpeg_cmd
        assert "-vf" in ffmpeg_cmd

    def test_portrait_source_uses_center_crop_not_blur(self):
        """Portrait source (9:16) should use center-crop, not blur."""
        optimizer = _make_optimizer("job-bb-5")
        segment = _make_segment("seg-bb-5", "job-bb-5")
        config = _make_config(face_tracking=False)

        with patch("subprocess.run", side_effect=[_probe_result(1080, 1920), _ok_result()]) as mock_run:
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "-filter_complex" not in ffmpeg_cmd


# ---------------------------------------------------------------------------
# Task 6.5: FormatOptimizer.optimize — face-tracking path
# ---------------------------------------------------------------------------

def _make_cv2_mock(faces=None, frame_width=1920, frame_height=1080):
    """Build a mock cv2 module for face-tracking tests."""
    cv2_mock = MagicMock()
    cv2_mock.CAP_PROP_FRAME_WIDTH = 3
    cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
    cv2_mock.COLOR_BGR2GRAY = 6

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: frame_width if prop == 3 else frame_height
    mock_cap.read.return_value = (True, MagicMock())
    cv2_mock.VideoCapture.return_value = mock_cap

    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = faces if faces is not None else []
    cv2_mock.CascadeClassifier.return_value = mock_cascade
    cv2_mock.data.haarcascades = ""

    return cv2_mock


class TestFormatOptimizerFaceTracking:
    def test_face_tracking_enabled_uses_vf_crop(self):
        """With face_tracking=True and a detected face, should use -vf with crop."""
        optimizer = _make_optimizer("job-ft-1")
        segment = _make_segment("seg-ft-1", "job-ft-1")
        config = _make_config(face_tracking=True)

        cv2_mock = _make_cv2_mock(faces=[(800, 100, 200, 200)])

        with patch("subprocess.run", return_value=_ok_result()) as mock_run, \
             patch.dict("sys.modules", {"cv2": cv2_mock}):
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args[0][0]
        assert "-vf" in ffmpeg_cmd
        vf = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
        assert "crop=" in vf

    def test_face_tracking_no_face_falls_back_to_center_crop(self):
        """With face_tracking=True but no face detected, should fall back to center-crop."""
        optimizer = _make_optimizer("job-ft-2")
        segment = _make_segment("seg-ft-2", "job-ft-2")
        config = _make_config(face_tracking=True)

        cv2_mock = _make_cv2_mock(faces=[])

        with patch("subprocess.run", return_value=_ok_result()) as mock_run, \
             patch.dict("sys.modules", {"cv2": cv2_mock}):
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args[0][0]
        assert "-vf" in ffmpeg_cmd
        # Center-crop uses -vf (not -filter_complex)
        assert "-filter_complex" not in ffmpeg_cmd

    def test_face_tracking_cv2_unavailable_falls_back_to_center_crop(self):
        """When cv2 is not installed, face tracking falls back to center-crop."""
        optimizer = _make_optimizer("job-ft-3")
        segment = _make_segment("seg-ft-3", "job-ft-3")
        config = _make_config(face_tracking=True)

        with patch("subprocess.run", return_value=_ok_result()) as mock_run, \
             patch.dict("sys.modules", {"cv2": None}):
            optimizer.optimize(segment, config)

        ffmpeg_cmd = mock_run.call_args[0][0]
        assert "-vf" in ffmpeg_cmd
        assert "-filter_complex" not in ffmpeg_cmd

    def test_face_tracking_uploads_formatted_key(self):
        optimizer = _make_optimizer("job-ft-4")
        segment = _make_segment("seg-ft-4", "job-ft-4")
        config = _make_config(face_tracking=True)

        cv2_mock = _make_cv2_mock(faces=[])

        with patch("subprocess.run", return_value=_ok_result()), \
             patch.dict("sys.modules", {"cv2": cv2_mock}):
            key = optimizer.optimize(segment, config)

        assert key == "jobs/job-ft-4/clips/seg-ft-4/formatted.mp4"
        optimizer.storage.upload_file.assert_called_once()
