"""Unit tests for AudioOptimizer (Tasks 10.1, 10.2, 10.4, 10.6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models import JobConfig, Segment
from app.workers.audio_optimizer import (
    FFmpegAudioError,
    AudioOptimizer,
    build_audio_cmd,
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


def _config(**kwargs) -> JobConfig:
    return JobConfig(**kwargs)


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


def _make_optimizer(job_id: str = "job-1") -> AudioOptimizer:
    storage = MagicMock()
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    return AudioOptimizer(job_id=job_id, storage=storage)


# ---------------------------------------------------------------------------
# Task 10.1: loudnorm always present
# ---------------------------------------------------------------------------

class TestLoudnormAlwaysPresent:
    def test_loudnorm_in_cmd_no_noise_no_music(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd_str

    def test_loudnorm_in_cmd_with_noise_reduction(self):
        config = _config(noise_reduction=True)
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd_str

    def test_loudnorm_in_cmd_with_music(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd_str

    def test_loudnorm_targets_minus14_lufs(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "I=-14" in cmd_str

    def test_loudnorm_true_peak_minus1_5(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "TP=-1.5" in cmd_str


# ---------------------------------------------------------------------------
# Task 10.2: afftdn noise reduction
# ---------------------------------------------------------------------------

class TestNoiseReduction:
    def test_afftdn_present_when_noise_reduction_true(self):
        config = _config(noise_reduction=True)
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "afftdn" in cmd_str

    def test_afftdn_absent_when_noise_reduction_false(self):
        config = _config(noise_reduction=False)
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "afftdn" not in cmd_str

    def test_afftdn_before_loudnorm_in_filter_chain(self):
        config = _config(noise_reduction=True)
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        afftdn_pos = cmd_str.index("afftdn")
        loudnorm_pos = cmd_str.index("loudnorm")
        assert afftdn_pos < loudnorm_pos

    def test_afftdn_before_loudnorm_with_music(self):
        config = _config(noise_reduction=True, background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        afftdn_pos = cmd_str.index("afftdn")
        loudnorm_pos = cmd_str.index("loudnorm")
        assert afftdn_pos < loudnorm_pos


# ---------------------------------------------------------------------------
# Task 10.2: background music mixing
# ---------------------------------------------------------------------------

class TestBackgroundMusicMixing:
    def test_music_mixed_at_minus12db(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        # -12 dB ≈ volume=0.25
        assert "volume=0.25" in cmd_str

    def test_music_looped_with_stream_loop(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        assert "stream_loop" in cmd_str or "aloop" in cmd_str

    def test_stream_loop_value_is_minus1(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        # -stream_loop -1 should appear as consecutive args
        assert "-stream_loop" in cmd
        idx = cmd.index("-stream_loop")
        assert cmd[idx + 1] == "-1"

    def test_music_input_added_to_cmd(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        assert "/tmp/music.mp3" in cmd

    def test_no_music_input_when_music_path_none(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path=None)
        # Should not have a second -i for music
        i_indices = [i for i, x in enumerate(cmd) if x == "-i"]
        assert len(i_indices) == 1  # only the main input

    def test_amix_used_for_mixing(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        assert "amix" in cmd_str


# ---------------------------------------------------------------------------
# Task 10.4: AAC 192k stereo output
# ---------------------------------------------------------------------------

class TestAACOutput:
    def test_aac_codec_flag_present(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        assert "-c:a" in cmd
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == "aac"

    def test_192k_bitrate_flag_present(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        assert "-b:a" in cmd
        idx = cmd.index("-b:a")
        assert cmd[idx + 1] == "192k"

    def test_stereo_flag_present(self):
        config = _config()
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        assert "-ac" in cmd
        idx = cmd.index("-ac")
        assert cmd[idx + 1] == "2"

    def test_aac_192k_stereo_with_noise_reduction(self):
        config = _config(noise_reduction=True)
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config)
        cmd_str = " ".join(cmd)
        assert "aac" in cmd_str
        assert "192k" in cmd_str
        assert "-ac 2" in cmd_str

    def test_aac_192k_stereo_with_music(self):
        config = _config(background_music_url="music.mp3")
        cmd = build_audio_cmd("/in.mp4", "/out.mp4", config, music_path="/tmp/music.mp3")
        cmd_str = " ".join(cmd)
        assert "aac" in cmd_str
        assert "192k" in cmd_str
        assert "-ac 2" in cmd_str


# ---------------------------------------------------------------------------
# Task 10.6: Missing music file fault isolation
# ---------------------------------------------------------------------------

class TestMissingMusicFaultIsolation:
    def test_missing_music_logs_warning(self, caplog):
        import logging

        optimizer = _make_optimizer("job-miss-1")
        seg = _segment("seg-1", "job-miss-1")
        config = _config(background_music_url="gs://bucket/missing_music.mp3")

        optimizer.storage.download_file.side_effect = [
            None,  # enhanced.mp4 download succeeds
            FileNotFoundError("not found"),  # music download fails
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            with caplog.at_level(logging.WARNING, logger="app.workers.audio_optimizer"):
                optimizer.optimize(seg, config)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_missing_music_warning_mentions_url(self, caplog):
        import logging

        optimizer = _make_optimizer("job-miss-2")
        seg = _segment("seg-1", "job-miss-2")
        config = _config(background_music_url="gs://bucket/missing_music.mp3")

        optimizer.storage.download_file.side_effect = [
            None,
            FileNotFoundError("not found"),
        ]

        with patch("subprocess.run", return_value=_ok_result()):
            with caplog.at_level(logging.WARNING, logger="app.workers.audio_optimizer"):
                optimizer.optimize(seg, config)

        all_text = " ".join(str(r.message) + str(r.args) for r in caplog.records)
        assert "missing_music.mp3" in all_text

    def test_missing_music_continues_with_normalization(self):
        optimizer = _make_optimizer("job-miss-3")
        seg = _segment("seg-1", "job-miss-3")
        config = _config(background_music_url="gs://bucket/missing_music.mp3")

        optimizer.storage.download_file.side_effect = [
            None,
            FileNotFoundError("not found"),
        ]

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            key = optimizer.optimize(seg, config)

        assert key == "jobs/job-miss-3/clips/seg-1/audio_optimized.mp4"
        assert mock_run.call_count == 1

    def test_missing_music_no_music_in_ffmpeg_cmd(self):
        optimizer = _make_optimizer("job-miss-4")
        seg = _segment("seg-1", "job-miss-4")
        config = _config(background_music_url="gs://bucket/missing_music.mp3")

        optimizer.storage.download_file.side_effect = [
            None,
            FileNotFoundError("not found"),
        ]

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            optimizer.optimize(seg, config)

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        # No amix or stream_loop — music was skipped
        assert "amix" not in cmd_str
        assert "stream_loop" not in cmd_str

    def test_missing_music_loudnorm_still_applied(self):
        optimizer = _make_optimizer("job-miss-5")
        seg = _segment("seg-1", "job-miss-5")
        config = _config(background_music_url="gs://bucket/missing_music.mp3")

        optimizer.storage.download_file.side_effect = [
            None,
            FileNotFoundError("not found"),
        ]

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            optimizer.optimize(seg, config)

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd_str


# ---------------------------------------------------------------------------
# Storage keys
# ---------------------------------------------------------------------------

class TestStorageKeys:
    def test_reads_from_enhanced_mp4(self):
        optimizer = _make_optimizer("job-keys-1")
        seg = _segment("seg-keys-1", "job-keys-1")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            optimizer.optimize(seg, config)

        first_download_key = optimizer.storage.download_file.call_args_list[0][0][0]
        assert first_download_key == "jobs/job-keys-1/clips/seg-keys-1/enhanced.mp4"

    def test_writes_to_audio_optimized_mp4(self):
        optimizer = _make_optimizer("job-keys-2")
        seg = _segment("seg-keys-2", "job-keys-2")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            key = optimizer.optimize(seg, config)

        assert key == "jobs/job-keys-2/clips/seg-keys-2/audio_optimized.mp4"
        upload_key = optimizer.storage.upload_file.call_args[0][0]
        assert upload_key == "jobs/job-keys-2/clips/seg-keys-2/audio_optimized.mp4"

    def test_ffmpeg_failure_raises_audio_error(self):
        optimizer = _make_optimizer("job-keys-3")
        seg = _segment("seg-keys-3", "job-keys-3")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result("bad filter")):
            with pytest.raises(FFmpegAudioError):
                optimizer.optimize(seg, config)

    def test_ffmpeg_failure_no_upload(self):
        optimizer = _make_optimizer("job-keys-4")
        seg = _segment("seg-keys-4", "job-keys-4")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result()):
            with pytest.raises(FFmpegAudioError):
                optimizer.optimize(seg, config)

        optimizer.storage.upload_file.assert_not_called()
