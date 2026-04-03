"""Unit tests for CaptionGenerator (Tasks 7.1, 7.3, 7.5, 7.7)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models import JobConfig, OverlayConfig, Segment, Transcript, WordToken
from app.workers.caption_generator import (
    CAPTION_Y_DEFAULT,
    CAPTION_Y_MAX,
    CAPTION_Y_MIN,
    CHUNK_MAX_WORDS,
    CHUNK_MIN_WORDS,
    MIN_FONT_SIZE,
    CaptionChunk,
    CaptionGenerator,
    FFmpegCaptionError,
    build_caption_cmd,
    chunk_transcript,
    compute_caption_y,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(text: str, start: float, end: float) -> WordToken:
    return WordToken(word=text, start=start, end=end, confidence=1.0)


def _transcript(words: list[WordToken], job_id: str = "job-1") -> Transcript:
    return Transcript(job_id=job_id, words=words, is_empty=len(words) == 0)


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


def _config(style: str = "default", overlays: list | None = None) -> JobConfig:
    return JobConfig(caption_style=style, overlays=overlays or [])


def _overlay(x: int = 0, y: int = 1400, start_time: float = 0.0, duration: float = 5.0) -> OverlayConfig:
    return OverlayConfig(asset_url="http://example.com/img.png", x=x, y=y, start_time=start_time, duration=duration)


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


# ---------------------------------------------------------------------------
# Task 7.1: chunk_transcript — word count per chunk
# ---------------------------------------------------------------------------

class TestChunkTranscript:
    def test_empty_transcript_returns_no_chunks(self):
        t = _transcript([])
        assert chunk_transcript(t) == []

    def test_single_word_produces_one_chunk(self):
        t = _transcript([_word("hello", 0.0, 0.5)])
        chunks = chunk_transcript(t)
        assert len(chunks) == 1
        assert chunks[0].text == "hello"

    def test_two_words_produce_one_chunk(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        assert len(chunks) == 1
        assert len(chunks[0].words) == 2

    def test_four_words_produce_one_chunk(self):
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(4)]
        chunks = chunk_transcript(_transcript(words))
        assert len(chunks) == 1
        assert len(chunks[0].words) == 4

    def test_five_words_produce_two_chunks(self):
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(5)]
        chunks = chunk_transcript(_transcript(words))
        assert len(chunks) == 2
        total_words = sum(len(c.words) for c in chunks)
        assert total_words == 5

    def test_eight_words_produce_two_chunks_of_four(self):
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(8)]
        chunks = chunk_transcript(_transcript(words))
        assert len(chunks) == 2
        for c in chunks:
            assert len(c.words) == 4

    def test_all_chunks_have_2_to_4_words(self):
        # 10 words → chunks of 4, 4, 2
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(10)]
        chunks = chunk_transcript(_transcript(words))
        for c in chunks:
            assert CHUNK_MIN_WORDS <= len(c.words) <= CHUNK_MAX_WORDS

    def test_all_words_are_covered(self):
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(11)]
        chunks = chunk_transcript(_transcript(words))
        total = sum(len(c.words) for c in chunks)
        assert total == 11

    def test_chunk_timing_start_equals_first_word_start(self):
        words = [_word("a", 1.0, 1.5), _word("b", 1.6, 2.0), _word("c", 2.1, 2.5)]
        chunks = chunk_transcript(_transcript(words))
        assert chunks[0].start == pytest.approx(1.0)

    def test_chunk_timing_end_equals_last_word_end(self):
        words = [_word("a", 1.0, 1.5), _word("b", 1.6, 2.0), _word("c", 2.1, 2.5)]
        chunks = chunk_transcript(_transcript(words))
        assert chunks[0].end == pytest.approx(2.5)

    def test_timing_within_100ms_of_speech(self):
        """Chunk start/end must be within 100ms of the actual word boundaries."""
        words = [
            _word("hello", 0.000, 0.450),
            _word("world", 0.500, 0.950),
            _word("foo", 1.000, 1.450),
        ]
        chunks = chunk_transcript(_transcript(words))
        for chunk in chunks:
            # start should equal first word's start exactly (no rounding)
            assert abs(chunk.start - chunk.words[0].start) < 0.1
            assert abs(chunk.end - chunk.words[-1].end) < 0.1


# ---------------------------------------------------------------------------
# Task 7.5: Style presets
# ---------------------------------------------------------------------------

class TestStylePresets:
    def test_default_style_uses_white_text(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="default")
        vf = cmd[cmd.index("-vf") + 1]
        assert "fontcolor=white" in vf

    def test_default_style_has_black_border(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="default")
        vf = cmd[cmd.index("-vf") + 1]
        assert "bordercolor=black" in vf
        assert "borderw=3" in vf

    def test_default_style_no_shadow(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="default")
        vf = cmd[cmd.index("-vf") + 1]
        assert "shadowcolor" not in vf

    def test_highlight_style_uses_yellow_text(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="highlight")
        vf = cmd[cmd.index("-vf") + 1]
        assert "fontcolor=yellow" in vf

    def test_highlight_style_has_drop_shadow(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="highlight")
        vf = cmd[cmd.index("-vf") + 1]
        assert "shadowcolor=black" in vf
        assert "shadowx=2" in vf
        assert "shadowy=2" in vf

    def test_highlight_style_no_border(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, caption_style="highlight")
        vf = cmd[cmd.index("-vf") + 1]
        assert "bordercolor" not in vf


# ---------------------------------------------------------------------------
# Task 7.7: Overlap detection
# ---------------------------------------------------------------------------

class TestOverlapDetection:
    def test_no_overlap_returns_default_y(self):
        chunk = CaptionChunk([_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)])
        # Overlay far away in time
        overlay = _overlay(y=1400, start_time=5.0, duration=2.0)
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, [overlay])
        assert y == CAPTION_Y_DEFAULT

    def test_temporal_overlap_but_no_vertical_overlap_returns_default_y(self):
        chunk = CaptionChunk([_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)])
        # Overlay at same time but far above caption area
        overlay = _overlay(y=100, start_time=0.0, duration=2.0)
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, [overlay])
        assert y == CAPTION_Y_DEFAULT

    def test_overlap_shifts_caption_upward(self):
        chunk = CaptionChunk([_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)])
        # Overlay at same time and same vertical position as caption
        overlay = _overlay(y=CAPTION_Y_DEFAULT, start_time=0.0, duration=2.0)
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, [overlay])
        assert y < CAPTION_Y_DEFAULT

    def test_shifted_y_stays_within_valid_range(self):
        chunk = CaptionChunk([_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)])
        overlay = _overlay(y=CAPTION_Y_DEFAULT, start_time=0.0, duration=2.0)
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, [overlay])
        assert y >= CAPTION_Y_MIN

    def test_no_overlays_returns_default_y(self):
        chunk = CaptionChunk([_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)])
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, [])
        assert y == CAPTION_Y_DEFAULT

    def test_multiple_overlays_only_overlapping_one_triggers_shift(self):
        chunk = CaptionChunk([_word("hello", 1.0, 1.5), _word("world", 1.6, 2.0)])
        overlays = [
            _overlay(y=CAPTION_Y_DEFAULT, start_time=0.0, duration=0.5),  # no time overlap
            _overlay(y=CAPTION_Y_DEFAULT, start_time=1.2, duration=1.0),  # overlaps
        ]
        y = compute_caption_y(CAPTION_Y_DEFAULT, chunk, overlays)
        assert y < CAPTION_Y_DEFAULT


# ---------------------------------------------------------------------------
# Task 7.3: FFmpeg command construction
# ---------------------------------------------------------------------------

class TestBuildCaptionCmd:
    def test_drawtext_filter_present(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        vf = cmd[cmd.index("-vf") + 1]
        assert "drawtext=" in vf

    def test_font_size_at_least_48(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks, font_size=48)
        vf = cmd[cmd.index("-vf") + 1]
        assert "fontsize=48" in vf

    def test_y_position_in_valid_range(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        vf = cmd[cmd.index("-vf") + 1]
        # Extract y= value
        for part in vf.split(":"):
            if part.startswith("y=") and "between" not in part:
                y_val = int(part.split("=")[1])
                assert CAPTION_Y_MIN <= y_val <= CAPTION_Y_MAX
                break

    def test_bold_font_specified(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        vf = cmd[cmd.index("-vf") + 1]
        assert "Bold" in vf

    def test_timing_enable_expression_present(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        vf = cmd[cmd.index("-vf") + 1]
        assert "between(t," in vf

    def test_uses_libx264(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_uses_aac(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        assert "-c:a" in cmd
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    def test_overwrite_flag(self):
        words = [_word("hello", 0.0, 0.5), _word("world", 0.6, 1.0)]
        chunks = chunk_transcript(_transcript(words))
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        assert "-y" in cmd

    def test_empty_chunks_produces_null_filter(self):
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", [])
        vf = cmd[cmd.index("-vf") + 1]
        assert vf == "null"

    def test_multiple_chunks_produce_multiple_drawtext_filters(self):
        words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(6)]
        chunks = chunk_transcript(_transcript(words))
        assert len(chunks) >= 2
        cmd = build_caption_cmd("/in.mp4", "/out.mp4", chunks)
        vf = cmd[cmd.index("-vf") + 1]
        assert vf.count("drawtext=") == len(chunks)


# ---------------------------------------------------------------------------
# CaptionGenerator integration (mocked subprocess + storage)
# ---------------------------------------------------------------------------

class TestCaptionGenerator:
    def _make_generator(self, job_id: str = "job-cg-1") -> CaptionGenerator:
        storage = MagicMock()
        storage.download_file.return_value = None
        storage.upload_file.return_value = None
        return CaptionGenerator(job_id=job_id, storage=storage)

    def _make_transcript(self, job_id: str = "job-cg-1") -> Transcript:
        words = [
            _word("hello", 0.0, 0.5),
            _word("world", 0.6, 1.0),
            _word("foo", 1.1, 1.5),
            _word("bar", 1.6, 2.0),
        ]
        return _transcript(words, job_id=job_id)

    def test_generate_returns_captioned_key(self):
        gen = self._make_generator("job-cg-1")
        seg = _segment("seg-cg-1", "job-cg-1", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-1")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            key = gen.generate(seg, transcript, config)

        assert key == "jobs/job-cg-1/clips/seg-cg-1/captioned.mp4"

    def test_generate_downloads_formatted_key(self):
        gen = self._make_generator("job-cg-2")
        seg = _segment("seg-cg-2", "job-cg-2", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-2")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            gen.generate(seg, transcript, config)

        download_key = gen.storage.download_file.call_args[0][0]
        assert download_key == "jobs/job-cg-2/clips/seg-cg-2/formatted.mp4"

    def test_generate_uploads_captioned_key(self):
        gen = self._make_generator("job-cg-3")
        seg = _segment("seg-cg-3", "job-cg-3", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-3")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()):
            gen.generate(seg, transcript, config)

        upload_key = gen.storage.upload_file.call_args[0][0]
        assert upload_key == "jobs/job-cg-3/clips/seg-cg-3/captioned.mp4"

    def test_ffmpeg_failure_raises_caption_error(self):
        gen = self._make_generator("job-cg-4")
        seg = _segment("seg-cg-4", "job-cg-4", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-4")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result("bad filter")):
            with pytest.raises(FFmpegCaptionError):
                gen.generate(seg, transcript, config)

    def test_ffmpeg_failure_no_upload(self):
        gen = self._make_generator("job-cg-5")
        seg = _segment("seg-cg-5", "job-cg-5", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-5")
        config = _config()

        with patch("subprocess.run", return_value=_fail_result()):
            with pytest.raises(FFmpegCaptionError):
                gen.generate(seg, transcript, config)

        gen.storage.upload_file.assert_not_called()

    def test_generate_uses_drawtext_in_ffmpeg_cmd(self):
        gen = self._make_generator("job-cg-6")
        seg = _segment("seg-cg-6", "job-cg-6", 0.0, 30.0)
        transcript = self._make_transcript("job-cg-6")
        config = _config()

        with patch("subprocess.run", return_value=_ok_result()) as mock_run:
            gen.generate(seg, transcript, config)

        ffmpeg_cmd = mock_run.call_args[0][0]
        vf = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
        assert "drawtext=" in vf
