"""Unit tests for VisualEnhancer (Tasks 9.1, 9.3, 9.5)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.models import (
    DynamicTextConfig,
    JobConfig,
    OverlayConfig,
    ProgressBarConfig,
    Segment,
)
from app.workers.visual_enhancer import (
    FFmpegEnhanceError,
    VisualEnhancer,
    build_enhance_cmd,
)


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


def _config(
    overlays=None,
    progress_bar=None,
    dynamic_text_words=None,
) -> JobConfig:
    kwargs = {}
    if overlays is not None:
        kwargs["overlays"] = overlays
    if progress_bar is not None:
        kwargs["progress_bar"] = progress_bar
    if dynamic_text_words is not None:
        kwargs["dynamic_text_words"] = dynamic_text_words
    return JobConfig(**kwargs)


def _overlay(asset_url="gs://bucket/emoji.png", x=100, y=200, start_time=1.0, duration=2.0) -> OverlayConfig:
    return OverlayConfig(asset_url=asset_url, x=x, y=y, start_time=start_time, duration=duration)


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


def _make_enhancer(job_id: str = "job-1") -> VisualEnhancer:
    storage = MagicMock()
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    return VisualEnhancer(job_id=job_id, storage=storage)


# ---------------------------------------------------------------------------
# Task 9.1: Single FFmpeg process regardless of overlay count
# ---------------------------------------------------------------------------

class TestSingleFFmpegProcess:
    def test_no_overlays_spawns_one_process(self):
        enhancer = _make_enhancer("job-single-1")
        seg = _segment("seg-1", "job-single-1")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            enhancer.enhance(seg, config)

        assert mock_run.call_count == 1

    def test_one_overlay_spawns_one_process(self, tmp_path):
        enhancer = _make_enhancer("job-single-2")
        seg = _segment("seg-1", "job-single-2")
        config = _config(overlays=[_overlay()])

        # Make download_file create the file so it's "found"
        def fake_download(key, local_path):
            Path(local_path).write_bytes(b"fake")

        enhancer.storage.download_file.side_effect = fake_download

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            enhancer.enhance(seg, config)

        assert mock_run.call_count == 1

    def test_three_overlays_spawns_one_process(self):
        enhancer = _make_enhancer("job-single-3")
        seg = _segment("seg-1", "job-single-3")
        overlays = [
            _overlay("gs://b/a.png", 10, 20, 0.0, 1.0),
            _overlay("gs://b/b.png", 30, 40, 2.0, 1.0),
            _overlay("gs://b/c.png", 50, 60, 4.0, 1.0),
        ]
        config = _config(overlays=overlays)

        def fake_download(key, local_path):
            Path(local_path).write_bytes(b"fake")

        enhancer.storage.download_file.side_effect = fake_download

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            enhancer.enhance(seg, config)

        assert mock_run.call_count == 1

    def test_no_overlays_uses_vf_not_filter_complex(self):
        seg = _segment()
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        assert "-vf" in cmd
        assert "-filter_complex" not in cmd

    def test_image_overlays_use_filter_complex(self, tmp_path):
        seg = _segment()
        overlay = _overlay()
        asset_path = tmp_path / "img.png"
        asset_path.write_bytes(b"fake")
        config = _config(overlays=[overlay])
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[(overlay, asset_path)])
        assert "-filter_complex" in cmd
        assert "-vf" not in cmd


# ---------------------------------------------------------------------------
# Task 9.3: Progress bar drawbox filter
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_drawbox_present_in_vf(self):
        seg = _segment()
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "drawbox" in vf

    def test_drawbox_at_bottom_of_frame(self):
        seg = _segment()
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "y=H-" in vf

    def test_drawbox_default_height_4px(self):
        seg = _segment()
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "h=4" in vf

    def test_drawbox_custom_height(self):
        seg = _segment()
        config = _config(progress_bar=ProgressBarConfig(height_px=8))
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "h=8" in vf

    def test_drawbox_custom_color(self):
        seg = _segment()
        config = _config(progress_bar=ProgressBarConfig(color="#FF0000"))
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "#FF0000" in vf

    def test_drawbox_width_uses_time_expression(self):
        seg = _segment(start=0.0, end=30.0)
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        # Width should be proportional to t/duration
        assert "W*t/" in vf

    def test_drawbox_fill_type(self):
        seg = _segment()
        config = _config()
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "t=fill" in vf


# ---------------------------------------------------------------------------
# Task 9.3: Dynamic text drawtext filter
# ---------------------------------------------------------------------------

class TestDynamicText:
    def test_drawtext_present_for_dynamic_word(self):
        seg = _segment()
        dtc = DynamicTextConfig(word="WOW", start_time=1.0, duration=0.5)
        config = _config(dynamic_text_words=[dtc])
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "drawtext=" in vf
        assert "WOW" in vf

    def test_drawtext_has_enable_expression(self):
        seg = _segment()
        dtc = DynamicTextConfig(word="FIRE", start_time=2.0, duration=0.5)
        config = _config(dynamic_text_words=[dtc])
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "between(t," in vf

    def test_drawtext_large_font(self):
        seg = _segment()
        dtc = DynamicTextConfig(word="BIG", start_time=0.0, duration=0.5)
        config = _config(dynamic_text_words=[dtc])
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        # Font size should be large (>=64)
        import re
        sizes = re.findall(r"fontsize=(\d+)", vf)
        assert any(int(s) >= 64 for s in sizes)

    def test_multiple_dynamic_words_all_present(self):
        seg = _segment()
        words = [
            DynamicTextConfig(word="ONE", start_time=0.0, duration=0.5),
            DynamicTextConfig(word="TWO", start_time=1.0, duration=0.5),
        ]
        config = _config(dynamic_text_words=words)
        cmd = build_enhance_cmd("/in.mp4", "/out.mp4", config, seg, resolved_overlays=[])
        vf = cmd[cmd.index("-vf") + 1]
        assert "ONE" in vf
        assert "TWO" in vf


# ---------------------------------------------------------------------------
# Task 9.5: Missing asset fault isolation
# ---------------------------------------------------------------------------

class TestMissingAssetFaultIsolation:
    def test_missing_asset_skips_overlay(self):
        enhancer = _make_enhancer("job-miss-1")
        seg = _segment("seg-1", "job-miss-1")
        config = _config(overlays=[_overlay("gs://bucket/missing.png")])

        # download_file raises to simulate missing asset
        enhancer.storage.download_file.side_effect = [
            None,  # first call: captioned.mp4 download succeeds
            FileNotFoundError("not found"),  # second call: overlay asset missing
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            key = enhancer.enhance(seg, config)

        # Should still complete successfully
        assert key == "jobs/job-miss-1/clips/seg-1/enhanced.mp4"

    def test_missing_asset_logs_warning(self, caplog):
        import logging

        enhancer = _make_enhancer("job-miss-2")
        seg = _segment("seg-1", "job-miss-2")
        config = _config(overlays=[_overlay("gs://bucket/missing.png")])

        enhancer.storage.download_file.side_effect = [
            None,
            FileNotFoundError("not found"),
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            with caplog.at_level(logging.WARNING, logger="app.workers.visual_enhancer"):
                enhancer.enhance(seg, config)

        assert any("missing.png" in r.message or "missing.png" in str(r.args) for r in caplog.records)

    def test_missing_asset_exactly_one_warning_per_asset(self, caplog):
        import logging

        enhancer = _make_enhancer("job-miss-3")
        seg = _segment("seg-1", "job-miss-3")
        config = _config(overlays=[
            _overlay("gs://bucket/a.png"),
            _overlay("gs://bucket/b.png"),
        ])

        enhancer.storage.download_file.side_effect = [
            None,  # captioned.mp4
            FileNotFoundError("not found"),  # a.png missing
            FileNotFoundError("not found"),  # b.png missing
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            with caplog.at_level(logging.WARNING, logger="app.workers.visual_enhancer"):
                enhancer.enhance(seg, config)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2

    def test_missing_asset_continues_with_remaining_overlays(self, tmp_path):
        enhancer = _make_enhancer("job-miss-4")
        seg = _segment("seg-1", "job-miss-4")
        config = _config(overlays=[
            _overlay("gs://bucket/missing.png"),
            _overlay("gs://bucket/present.png"),
        ])

        def fake_download(key, local_path):
            if "captioned" in local_path or "present" in key:
                Path(local_path).write_bytes(b"fake")
            else:
                raise FileNotFoundError("not found")

        enhancer.storage.download_file.side_effect = fake_download

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            enhancer.enhance(seg, config)

        # FFmpeg should still be called (with the one present overlay)
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert "-filter_complex" in cmd  # present.png triggers filter_complex


# ---------------------------------------------------------------------------
# Enhanced clip upload key
# ---------------------------------------------------------------------------

class TestEnhancedClipUpload:
    def test_enhanced_clip_uploaded_to_correct_key(self):
        enhancer = _make_enhancer("job-upload-1")
        seg = _segment("seg-upload-1", "job-upload-1")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            key = enhancer.enhance(seg, config)

        assert key == "jobs/job-upload-1/clips/seg-upload-1/enhanced.mp4"
        upload_key = enhancer.storage.upload_file.call_args[0][0]
        assert upload_key == "jobs/job-upload-1/clips/seg-upload-1/enhanced.mp4"

    def test_captioned_clip_downloaded_from_correct_key(self):
        enhancer = _make_enhancer("job-upload-2")
        seg = _segment("seg-upload-2", "job-upload-2")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            enhancer.enhance(seg, config)

        # First download_file call is for the captioned clip
        first_download_key = enhancer.storage.download_file.call_args_list[0][0][0]
        assert first_download_key == "jobs/job-upload-2/clips/seg-upload-2/captioned.mp4"

    def test_ffmpeg_failure_raises_enhance_error(self):
        enhancer = _make_enhancer("job-upload-3")
        seg = _segment("seg-upload-3", "job-upload-3")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result("bad filter")):
            with pytest.raises(FFmpegEnhanceError):
                enhancer.enhance(seg, config)

    def test_ffmpeg_failure_no_upload(self):
        enhancer = _make_enhancer("job-upload-4")
        seg = _segment("seg-upload-4", "job-upload-4")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result()):
            with pytest.raises(FFmpegEnhanceError):
                enhancer.enhance(seg, config)

        enhancer.storage.upload_file.assert_not_called()
