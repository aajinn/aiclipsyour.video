"""Unit tests for ExportEngine (Tasks 11.1, 11.3, 11.5, 11.6)."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from app.models import Segment
from app.workers.export_engine import ExportEngine, ExportError, build_export_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment(
    segment_id: str = "seg-1",
    job_id: str = "job-1",
    start: float = 0.0,
    end: float = 30.0,
) -> Segment:
    return Segment(
        segment_id=segment_id,
        job_id=job_id,
        start=start,
        end=end,
        score=0.8,
        llm_virality_score=0.7,
        rank=1,
    )


def _ok_result() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


def _fail_result(stderr: str = "ffmpeg error") -> MagicMock:
    r = MagicMock()
    r.returncode = 1
    r.stderr = stderr
    return r


def _make_engine(job_id: str = "job-1") -> ExportEngine:
    storage = MagicMock()
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    storage.generate_signed_url.return_value = "https://signed.url/final.mp4"
    return ExportEngine(job_id=job_id, storage=storage)


# ---------------------------------------------------------------------------
# Task 11.1: FFmpeg codec and encoding flags
# ---------------------------------------------------------------------------

class TestFFmpegFlags:
    def test_h264_codec_flag(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-c:v" in cmd
        idx = cmd.index("-c:v")
        assert cmd[idx + 1] == "libx264"

    def test_aac_codec_flag(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-c:a" in cmd
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == "aac"

    def test_video_bitrate_8mbps(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "8M"

    def test_audio_bitrate_192k(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-b:a" in cmd
        idx = cmd.index("-b:a")
        assert cmd[idx + 1] == "192k"

    def test_framerate_30fps(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-r" in cmd
        idx = cmd.index("-r")
        assert cmd[idx + 1] == "30"

    def test_resolution_1080x1920(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-s" in cmd
        idx = cmd.index("-s")
        assert cmd[idx + 1] == "1080x1920"

    def test_faststart_flag(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-movflags" in cmd
        idx = cmd.index("-movflags")
        assert cmd[idx + 1] == "+faststart"


# ---------------------------------------------------------------------------
# Task 11.3: 60-second trim
# ---------------------------------------------------------------------------

class TestDurationTrim:
    def test_trim_flag_present_when_over_60s(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=61.0)
        assert "-t" in cmd
        idx = cmd.index("-t")
        assert cmd[idx + 1] == "60"

    def test_trim_flag_present_when_exactly_over_60s(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=60.1)
        assert "-t" in cmd

    def test_no_trim_flag_when_exactly_60s(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=60.0)
        assert "-t" not in cmd

    def test_no_trim_flag_when_under_60s(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=30.0)
        assert "-t" not in cmd

    def test_no_trim_flag_when_well_under_60s(self):
        cmd = build_export_cmd("/in.mp4", "/out.mp4", duration=10.0)
        assert "-t" not in cmd


# ---------------------------------------------------------------------------
# Task 11.5: Upload and signed URL
# ---------------------------------------------------------------------------

class TestSignedUrl:
    def test_signed_url_generated_with_86400_expiration(self):
        engine = _make_engine("job-url-1")
        seg = _segment("seg-url-1", "job-url-1", start=0.0, end=30.0)

        with patch("subprocess.run", return_value=_ok_result()):
            engine.export(seg)

        engine.storage.generate_signed_url.assert_called_once()
        call_kwargs = engine.storage.generate_signed_url.call_args
        # expiration_seconds must be >= 86400
        expiration = call_kwargs[1].get("expiration_seconds") or call_kwargs[0][1]
        assert expiration >= 86400

    def test_signed_url_returned(self):
        engine = _make_engine("job-url-2")
        seg = _segment("seg-url-2", "job-url-2", start=0.0, end=30.0)
        engine.storage.generate_signed_url.return_value = "https://example.com/signed"

        with patch("subprocess.run", return_value=_ok_result()):
            result = engine.export(seg)

        assert result == "https://example.com/signed"

    def test_signed_url_uses_final_key(self):
        engine = _make_engine("job-url-3")
        seg = _segment("seg-url-3", "job-url-3", start=0.0, end=30.0)

        with patch("subprocess.run", return_value=_ok_result()):
            engine.export(seg)

        signed_url_key = engine.storage.generate_signed_url.call_args[0][0]
        assert signed_url_key == "jobs/job-url-3/clips/seg-url-3/final.mp4"


# ---------------------------------------------------------------------------
# Task 11.6: Render failure handling
# ---------------------------------------------------------------------------

class TestRenderFailure:
    def test_export_error_raised_on_ffmpeg_failure(self):
        engine = _make_engine("job-fail-1")
        seg = _segment("seg-fail-1", "job-fail-1")

        with patch("subprocess.run", return_value=_fail_result("codec not found")):
            with pytest.raises(ExportError):
                engine.export(seg)

    def test_export_error_message_includes_stderr(self):
        engine = _make_engine("job-fail-2")
        seg = _segment("seg-fail-2", "job-fail-2")
        stderr_msg = "libx264 encoder not found"

        with patch("subprocess.run", return_value=_fail_result(stderr_msg)):
            with pytest.raises(ExportError, match=stderr_msg):
                engine.export(seg)

    def test_no_upload_on_render_failure(self):
        engine = _make_engine("job-fail-3")
        seg = _segment("seg-fail-3", "job-fail-3")

        with patch("subprocess.run", return_value=_fail_result()):
            with pytest.raises(ExportError):
                engine.export(seg)

        engine.storage.upload_file.assert_not_called()

    def test_no_signed_url_on_render_failure(self):
        engine = _make_engine("job-fail-4")
        seg = _segment("seg-fail-4", "job-fail-4")

        with patch("subprocess.run", return_value=_fail_result()):
            with pytest.raises(ExportError):
                engine.export(seg)

        engine.storage.generate_signed_url.assert_not_called()


# ---------------------------------------------------------------------------
# Storage keys
# ---------------------------------------------------------------------------

class TestStorageKeys:
    def test_reads_from_audio_optimized_key(self):
        engine = _make_engine("job-keys-1")
        seg = _segment("seg-keys-1", "job-keys-1")

        with patch("subprocess.run", return_value=_ok_result()):
            engine.export(seg)

        download_key = engine.storage.download_file.call_args_list[0][0][0]
        assert download_key == "jobs/job-keys-1/clips/seg-keys-1/audio_optimized.mp4"

    def test_uploads_to_final_key(self):
        engine = _make_engine("job-keys-2")
        seg = _segment("seg-keys-2", "job-keys-2")

        with patch("subprocess.run", return_value=_ok_result()):
            engine.export(seg)

        upload_key = engine.storage.upload_file.call_args[0][0]
        assert upload_key == "jobs/job-keys-2/clips/seg-keys-2/final.mp4"
