"""Unit tests for ClipExtractor (Task 5.1, 5.3, 5.4, 5.6)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from app.models import Segment
from app.workers.clip_extractor import (
    ClipExtractor,
    FFmpegExtractionError,
    build_reencode_cmd,
    build_stream_copy_cmd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(job_id: str = "job-ce-test") -> ClipExtractor:
    storage = MagicMock()
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    storage.upload.return_value = None
    return ClipExtractor(job_id=job_id, storage=storage)


def _make_segment(
    segment_id: str = "seg-1",
    job_id: str = "job-ce-test",
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


def _fail_result(stderr: str = "some ffmpeg error") -> MagicMock:
    r = MagicMock()
    r.returncode = 1
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# Task 5.1: FFmpeg command construction — stream-copy
# ---------------------------------------------------------------------------

class TestBuildStreamCopyCmd:
    def test_contains_c_copy_flag(self):
        cmd = build_stream_copy_cmd("/in.mp4", "/out.mp4", 5.0, 35.0)
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "copy"

    def test_contains_ss_and_to_flags(self):
        cmd = build_stream_copy_cmd("/in.mp4", "/out.mp4", 5.0, 35.0)
        assert "-ss" in cmd
        assert "-to" in cmd

    def test_start_end_values_present(self):
        cmd = build_stream_copy_cmd("/in.mp4", "/out.mp4", 10.5, 40.5)
        assert "10.5" in cmd
        assert "40.5" in cmd

    def test_input_and_output_paths_present(self):
        cmd = build_stream_copy_cmd("/input.mp4", "/output.mp4", 0.0, 30.0)
        assert "/input.mp4" in cmd
        assert "/output.mp4" in cmd

    def test_no_video_encoder_flag(self):
        """Stream-copy must not include -c:v."""
        cmd = build_stream_copy_cmd("/in.mp4", "/out.mp4", 0.0, 30.0)
        assert "-c:v" not in cmd

    def test_no_audio_encoder_flag(self):
        """Stream-copy must not include -c:a."""
        cmd = build_stream_copy_cmd("/in.mp4", "/out.mp4", 0.0, 30.0)
        assert "-c:a" not in cmd


# ---------------------------------------------------------------------------
# Task 5.3: FFmpeg command construction — re-encode fallback
# ---------------------------------------------------------------------------

class TestBuildReencodeCmd:
    def test_contains_libx264_flag(self):
        cmd = build_reencode_cmd("/in.mp4", "/out.mp4", 5.0, 35.0)
        assert "-c:v" in cmd
        idx = cmd.index("-c:v")
        assert cmd[idx + 1] == "libx264"

    def test_contains_aac_flag(self):
        cmd = build_reencode_cmd("/in.mp4", "/out.mp4", 5.0, 35.0)
        assert "-c:a" in cmd
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == "aac"

    def test_contains_ss_and_to_flags(self):
        cmd = build_reencode_cmd("/in.mp4", "/out.mp4", 5.0, 35.0)
        assert "-ss" in cmd
        assert "-to" in cmd

    def test_start_end_values_present(self):
        cmd = build_reencode_cmd("/in.mp4", "/out.mp4", 10.0, 50.0)
        assert "10.0" in cmd
        assert "50.0" in cmd


# ---------------------------------------------------------------------------
# Task 5.1: extract_clip — stream-copy success path
# ---------------------------------------------------------------------------

class TestExtractClipStreamCopy:
    def test_stream_copy_success_uploads_to_correct_key(self):
        extractor = _make_extractor(job_id="job-sc-1")
        segment = _make_segment(segment_id="seg-sc-1", job_id="job-sc-1")

        with patch("subprocess.run", return_value=_ok_result()):
            key = extractor.extract_clip(segment)

        assert key == "jobs/job-sc-1/clips/seg-sc-1/raw.mp4"
        extractor.storage.upload_file.assert_called_once()
        upload_key = extractor.storage.upload_file.call_args[0][0]
        assert upload_key == "jobs/job-sc-1/clips/seg-sc-1/raw.mp4"

    def test_stream_copy_downloads_source_from_correct_key(self):
        extractor = _make_extractor(job_id="job-sc-2")
        segment = _make_segment(segment_id="seg-sc-2", job_id="job-sc-2")

        with patch("subprocess.run", return_value=_ok_result()):
            extractor.extract_clip(segment)

        download_key = extractor.storage.download_file.call_args[0][0]
        assert download_key == "jobs/job-sc-2/source.mp4"

    def test_stream_copy_ffmpeg_called_once_on_success(self):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            extractor.extract_clip(segment)

        assert mock_run.call_count == 1

    def test_stream_copy_cmd_uses_c_copy(self):
        extractor = _make_extractor()
        segment = _make_segment(start=5.0, end=35.0)

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            extractor.extract_clip(segment)

        cmd = mock_run.call_args[0][0]
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"


# ---------------------------------------------------------------------------
# Task 5.3: extract_clip — re-encode fallback
# ---------------------------------------------------------------------------

class TestExtractClipReencodeFallback:
    def test_reencode_attempted_when_stream_copy_fails(self):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", side_effect=[_fail_result(), _ok_result()]) as mock_run:
            extractor.extract_clip(segment)

        assert mock_run.call_count == 2
        reencode_cmd = mock_run.call_args_list[1][0][0]
        assert "-c:v" in reencode_cmd
        assert "libx264" in reencode_cmd

    def test_reencode_success_uploads_clip(self):
        extractor = _make_extractor(job_id="job-re-1")
        segment = _make_segment(segment_id="seg-re-1", job_id="job-re-1")

        with patch("subprocess.run", side_effect=[_fail_result(), _ok_result()]):
            key = extractor.extract_clip(segment)

        assert key == "jobs/job-re-1/clips/seg-re-1/raw.mp4"
        extractor.storage.upload_file.assert_called_once()

    def test_both_fail_raises_ffmpeg_extraction_error(self):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", side_effect=[_fail_result("err1"), _fail_result("err2")]):
            with pytest.raises(FFmpegExtractionError):
                extractor.extract_clip(segment)

    def test_both_fail_no_upload(self):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", side_effect=[_fail_result(), _fail_result()]):
            with pytest.raises(FFmpegExtractionError):
                extractor.extract_clip(segment)

        extractor.storage.upload_file.assert_not_called()


# ---------------------------------------------------------------------------
# Task 5.6: Per-segment failure isolation
# ---------------------------------------------------------------------------

class TestSegmentFailureIsolation:
    def test_one_failing_segment_does_not_cancel_others(self):
        extractor = _make_extractor(job_id="job-iso-1")
        segments = [
            _make_segment("seg-a", "job-iso-1"),
            _make_segment("seg-b", "job-iso-1"),
            _make_segment("seg-c", "job-iso-1"),
        ]

        # seg-a succeeds (1 call), seg-b fails both attempts (2 calls), seg-c succeeds (1 call)
        side_effects = [_ok_result(), _fail_result(), _fail_result(), _ok_result()]

        with patch("subprocess.run", side_effect=side_effects):
            results = extractor.extract_all(segments)

        assert results["seg-a"] is not None
        assert results["seg-b"] is None
        assert results["seg-c"] is not None

    def test_failed_segment_result_is_none(self):
        extractor = _make_extractor()
        segment = _make_segment("seg-fail")

        with patch("subprocess.run", side_effect=[_fail_result(), _fail_result()]):
            results = extractor.extract_all([segment])

        assert results["seg-fail"] is None

    def test_failed_segment_error_persisted_to_storage(self):
        extractor = _make_extractor(job_id="job-err-persist")
        segment = _make_segment("seg-err", "job-err-persist")

        with patch("subprocess.run", side_effect=[_fail_result("bad codec"), _fail_result("bad codec")]):
            extractor.extract_all([segment])

        # error.json should have been uploaded
        upload_calls = extractor.storage.upload.call_args_list
        error_keys = [c[0][0] for c in upload_calls]
        assert any("seg-err/error.json" in k for k in error_keys)

    def test_stderr_logged_on_stream_copy_failure(self, caplog):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", side_effect=[_fail_result("stream copy stderr"), _ok_result()]):
            with caplog.at_level("WARNING", logger="app.workers.clip_extractor"):
                extractor.extract_clip(segment)

        assert "stream copy stderr" in caplog.text

    def test_stderr_logged_on_reencode_failure(self, caplog):
        extractor = _make_extractor()
        segment = _make_segment()

        with patch("subprocess.run", side_effect=[_fail_result("sc err"), _fail_result("reencode stderr")]):
            with caplog.at_level("ERROR", logger="app.workers.clip_extractor"):
                with pytest.raises(FFmpegExtractionError):
                    extractor.extract_clip(segment)

        assert "reencode stderr" in caplog.text

    def test_all_segments_succeed_returns_all_keys(self):
        extractor = _make_extractor(job_id="job-all-ok")
        segments = [
            _make_segment("seg-1", "job-all-ok"),
            _make_segment("seg-2", "job-all-ok"),
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            results = extractor.extract_all(segments)

        assert results["seg-1"] == "jobs/job-all-ok/clips/seg-1/raw.mp4"
        assert results["seg-2"] == "jobs/job-all-ok/clips/seg-2/raw.mp4"


# ---------------------------------------------------------------------------
# Task 5.4: run() — reads segments.json and extracts all
# ---------------------------------------------------------------------------

class TestClipExtractorRun:
    def test_run_reads_segments_from_correct_key(self):
        extractor = _make_extractor(job_id="job-run-1")
        segments = [_make_segment("seg-r1", "job-run-1")]
        extractor.storage.download.return_value = json.dumps(
            [s.model_dump() for s in segments]
        ).encode("utf-8")

        with patch("subprocess.run", return_value=_ok_result()):
            extractor.run()

        extractor.storage.download.assert_called_once_with("jobs/job-run-1/segments.json")

    def test_run_extracts_all_segments(self):
        extractor = _make_extractor(job_id="job-run-2")
        segments = [
            _make_segment("seg-r2a", "job-run-2"),
            _make_segment("seg-r2b", "job-run-2"),
        ]
        extractor.storage.download.return_value = json.dumps(
            [s.model_dump() for s in segments]
        ).encode("utf-8")

        with patch("subprocess.run", return_value=_ok_result()):
            results = extractor.run()

        assert len(results) == 2
        assert "seg-r2a" in results
        assert "seg-r2b" in results

    def test_run_returns_none_for_failed_segments(self):
        extractor = _make_extractor(job_id="job-run-3")
        segments = [_make_segment("seg-r3", "job-run-3")]
        extractor.storage.download.return_value = json.dumps(
            [s.model_dump() for s in segments]
        ).encode("utf-8")

        with patch("subprocess.run", side_effect=[_fail_result(), _fail_result()]):
            results = extractor.run()

        assert results["seg-r3"] is None
