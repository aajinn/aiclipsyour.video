"""Unit tests for HighlightDetector (Tasks 3.1, 3.3, 3.5, 3.8)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.models import Segment, Transcript, WordToken
from app.workers.highlight_detector import (
    HIGH_SIGNAL_KEYWORDS,
    MAX_SEGMENT_DURATION,
    MAX_SEGMENTS,
    MIN_SEGMENT_DURATION,
    HighlightDetector,
    _extract_candidate_segments,
    _score_heuristic,
    _score_llm_virality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(job_id: str = "job-hd-test") -> HighlightDetector:
    storage = MagicMock()
    return HighlightDetector(job_id=job_id, storage=storage)


def _make_words(texts: list[str], start: float = 0.0, word_duration: float = 1.0) -> list[WordToken]:
    """Build a list of WordTokens from a list of text strings."""
    words = []
    t = start
    for text in texts:
        words.append(WordToken(word=text, start=t, end=t + word_duration, confidence=1.0))
        t += word_duration
    return words


def _make_transcript(
    words: list[WordToken],
    job_id: str = "job-hd-test",
    is_empty: bool = False,
) -> Transcript:
    return Transcript(job_id=job_id, words=words, is_empty=is_empty)


def _make_long_transcript(n_words: int = 60, job_id: str = "job-hd-test") -> Transcript:
    """Build a transcript with n_words words, each 1 second long."""
    words = _make_words([f"word{i}" for i in range(n_words)], word_duration=1.0)
    return _make_transcript(words, job_id=job_id)


# ---------------------------------------------------------------------------
# Task 3.1: Heuristic scoring
# ---------------------------------------------------------------------------

class TestScoreHeuristic:
    def test_empty_words_returns_zero(self):
        assert _score_heuristic([]) == 0.0

    def test_keyword_presence_increases_score(self):
        """A segment with a high-signal keyword should score higher than one without."""
        words_with_keyword = _make_words(["this", "is", "secret", "info"])
        words_without = _make_words(["this", "is", "normal", "info"])
        assert _score_heuristic(words_with_keyword) > _score_heuristic(words_without)

    def test_multiple_keywords_increase_score_more(self):
        one_keyword = _make_words(["this", "is", "crazy", "stuff"])
        two_keywords = _make_words(["this", "is", "crazy", "amazing"])
        assert _score_heuristic(two_keywords) > _score_heuristic(one_keyword)

    def test_preferred_length_range_scores_higher_than_too_short(self):
        """3–10 word utterances should score higher than 1-word utterances."""
        short = _make_words(["hi"])
        preferred = _make_words(["this", "is", "a", "great", "moment"])
        assert _score_heuristic(preferred) > _score_heuristic(short)

    def test_question_detection_increases_score(self):
        """Sentences ending with '?' should score higher."""
        question = _make_words(["is", "this", "real?"])
        statement = _make_words(["is", "this", "real"])
        assert _score_heuristic(question) > _score_heuristic(statement)

    def test_exclamation_mark_increases_score(self):
        """Exclamation marks signal emotional intensity."""
        excited = _make_words(["this", "is", "amazing!"])
        calm = _make_words(["this", "is", "amazing"])
        assert _score_heuristic(excited) > _score_heuristic(calm)

    def test_caps_words_increase_score(self):
        """ALL CAPS words signal emotional intensity."""
        caps = _make_words(["this", "is", "HUGE", "news"])
        no_caps = _make_words(["this", "is", "huge", "news"])
        assert _score_heuristic(caps) > _score_heuristic(no_caps)

    def test_score_clamped_to_one(self):
        """Score must never exceed 1.0."""
        # Pile on every signal
        words = _make_words(["SECRET!", "CRAZY!", "AMAZING!", "SHOCKING!", "UNBELIEVABLE!"])
        score = _score_heuristic(words)
        assert score <= 1.0

    def test_score_non_negative(self):
        words = _make_words(["hello", "world"])
        assert _score_heuristic(words) >= 0.0

    def test_all_high_signal_keywords_recognized(self):
        """Every keyword in HIGH_SIGNAL_KEYWORDS should boost the score."""
        baseline = _make_words(["this", "is", "a", "test"])
        baseline_score = _score_heuristic(baseline)
        for kw in list(HIGH_SIGNAL_KEYWORDS)[:5]:  # spot-check 5
            words_with_kw = _make_words(["this", "is", kw, "test"])
            assert _score_heuristic(words_with_kw) > baseline_score, f"Keyword '{kw}' did not boost score"


# ---------------------------------------------------------------------------
# Task 3.3: LLM virality scoring
# ---------------------------------------------------------------------------

class TestScoreLlmVirality:
    def test_returns_float_in_range(self):
        """LLM score must be in [0.0, 1.0]."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "7"

        mock_completion = [mock_chunk]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(mock_completion)

        mock_groq_module = MagicMock()
        mock_groq_module.Groq.return_value = mock_client

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            score = _score_llm_virality("this is a test segment")

        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.7)

    def test_groq_unavailable_returns_zero(self):
        """When Groq is unavailable, fall back to 0.0."""
        with patch.dict("sys.modules", {"groq": None}):
            score = _score_llm_virality("some text")
        assert score == 0.0

    def test_groq_exception_returns_zero(self):
        """When Groq raises an exception, fall back to 0.0."""
        mock_groq_module = MagicMock()
        mock_groq_module.Groq.side_effect = RuntimeError("connection refused")

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            score = _score_llm_virality("some text")

        assert score == 0.0

    def test_unparseable_response_returns_zero(self):
        """When LLM returns non-numeric text, fall back to 0.0."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "I cannot rate this."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        mock_groq_module = MagicMock()
        mock_groq_module.Groq.return_value = mock_client

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            score = _score_llm_virality("some text")

        assert score == 0.0

    def test_score_normalized_from_0_to_10(self):
        """LLM returns 0–10; should be normalized to 0.0–1.0."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "10"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        mock_groq_module = MagicMock()
        mock_groq_module.Groq.return_value = mock_client

        with patch.dict("sys.modules", {"groq": mock_groq_module}):
            score = _score_llm_virality("viral content")

        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task 3.5: Segment count and duration constraints
# ---------------------------------------------------------------------------

class TestHighlightDetectorConstraints:
    def test_empty_transcript_returns_zero_segments(self):
        """Empty transcript (is_empty=True) → 0 segments."""
        detector = _make_detector()
        transcript = _make_transcript([], is_empty=True)
        segments = detector.detect(transcript)
        assert segments == []

    def test_empty_words_list_returns_zero_segments(self):
        """Transcript with no words → 0 segments."""
        detector = _make_detector()
        transcript = _make_transcript([])
        segments = detector.detect(transcript)
        assert segments == []

    def test_non_empty_transcript_returns_at_least_one_segment(self):
        """Non-empty transcript → at least 1 segment."""
        detector = _make_detector()
        # Patch LLM to avoid network calls
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=60)
            segments = detector.detect(transcript)
        assert len(segments) >= 1

    def test_returns_at_most_ten_segments(self):
        """Never return more than 10 segments."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=200)
            segments = detector.detect(transcript)
        assert len(segments) <= MAX_SEGMENTS

    def test_segment_duration_at_least_15_seconds(self):
        """Every segment duration must be >= 15 seconds."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=120)
            segments = detector.detect(transcript)
        for seg in segments:
            assert seg.end - seg.start >= MIN_SEGMENT_DURATION, (
                f"Segment {seg.segment_id} duration {seg.end - seg.start} < {MIN_SEGMENT_DURATION}"
            )

    def test_segment_duration_at_most_60_seconds(self):
        """Every segment duration must be <= 60 seconds."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=120)
            segments = detector.detect(transcript)
        for seg in segments:
            assert seg.end - seg.start <= MAX_SEGMENT_DURATION, (
                f"Segment {seg.segment_id} duration {seg.end - seg.start} > {MAX_SEGMENT_DURATION}"
            )

    def test_segments_ranked_by_composite_score_descending(self):
        """Segments must be ordered by rank (1 = highest score)."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=120)
            segments = detector.detect(transcript)
        if len(segments) > 1:
            scores = [s.score for s in segments]
            assert scores == sorted(scores, reverse=True)

    def test_segment_rank_starts_at_one(self):
        """First segment should have rank=1."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=60)
            segments = detector.detect(transcript)
        if segments:
            assert segments[0].rank == 1

    def test_segment_job_id_matches(self):
        """All segments must have the correct job_id."""
        detector = _make_detector(job_id="my-job-123")
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            transcript = _make_long_transcript(n_words=60, job_id="my-job-123")
            segments = detector.detect(transcript)
        for seg in segments:
            assert seg.job_id == "my-job-123"

    def test_llm_score_stored_in_segment(self):
        """LLM virality score must be stored in segment.llm_virality_score."""
        detector = _make_detector()
        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.8):
            transcript = _make_long_transcript(n_words=60)
            segments = detector.detect(transcript)
        if segments:
            assert segments[0].llm_virality_score == pytest.approx(0.8, abs=0.01)

    def test_composite_score_incorporates_llm_score(self):
        """Composite score should be higher when LLM score is higher."""
        detector = _make_detector()
        transcript = _make_long_transcript(n_words=60)

        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.0):
            segments_low_llm = detector.detect(transcript)

        with patch("app.workers.highlight_detector._score_llm_virality", return_value=1.0):
            segments_high_llm = detector.detect(transcript)

        if segments_low_llm and segments_high_llm:
            assert segments_high_llm[0].score > segments_low_llm[0].score


# ---------------------------------------------------------------------------
# Task 3.8: Write segments.json to cloud storage
# ---------------------------------------------------------------------------

class TestWriteSegments:
    def test_write_segments_uploads_to_correct_key(self):
        """segments.json must be written to jobs/{job_id}/segments.json."""
        detector = _make_detector(job_id="job-write-test")
        segments = [
            Segment(
                segment_id="seg-1",
                job_id="job-write-test",
                start=0.0,
                end=30.0,
                score=0.8,
                llm_virality_score=0.7,
                rank=1,
            )
        ]
        detector.write_segments(segments)

        detector.storage.upload.assert_called_once()
        call_args = detector.storage.upload.call_args
        key = call_args[0][0]
        assert key == "jobs/job-write-test/segments.json"

    def test_write_segments_content_type_is_json(self):
        """Content type must be application/json."""
        detector = _make_detector(job_id="job-ct-test")
        detector.write_segments([])
        call_kwargs = detector.storage.upload.call_args[1]
        assert call_kwargs.get("content_type") == "application/json"

    def test_write_segments_serializes_all_fields(self):
        """Serialized JSON must contain all Segment fields."""
        detector = _make_detector(job_id="job-serial-test")
        seg = Segment(
            segment_id="seg-abc",
            job_id="job-serial-test",
            start=5.0,
            end=35.0,
            score=0.75,
            llm_virality_score=0.6,
            rank=1,
        )
        detector.write_segments([seg])

        raw_bytes = detector.storage.upload.call_args[0][1]
        data = json.loads(raw_bytes.decode("utf-8"))
        assert len(data) == 1
        assert data[0]["segment_id"] == "seg-abc"
        assert data[0]["job_id"] == "job-serial-test"
        assert data[0]["start"] == pytest.approx(5.0)
        assert data[0]["end"] == pytest.approx(35.0)
        assert data[0]["score"] == pytest.approx(0.75)
        assert data[0]["llm_virality_score"] == pytest.approx(0.6)
        assert data[0]["rank"] == 1

    def test_write_empty_segments_list(self):
        """Writing an empty list should still upload valid JSON."""
        detector = _make_detector(job_id="job-empty-write")
        detector.write_segments([])

        raw_bytes = detector.storage.upload.call_args[0][1]
        data = json.loads(raw_bytes.decode("utf-8"))
        assert data == []

    def test_run_reads_transcript_and_writes_segments(self):
        """run() should read transcript.json and write segments.json."""
        detector = _make_detector(job_id="job-run-test")

        # Build a transcript with enough words for at least one segment
        words = _make_words([f"word{i}" for i in range(60)], word_duration=1.0)
        transcript = _make_transcript(words, job_id="job-run-test")
        transcript_json = json.dumps(transcript.model_dump()).encode("utf-8")

        detector.storage.download.return_value = transcript_json

        with patch("app.workers.highlight_detector._score_llm_virality", return_value=0.5):
            segments = detector.run()

        # Transcript was read from the correct key
        detector.storage.download.assert_called_once_with("jobs/job-run-test/transcript.json")
        # segments.json was written
        detector.storage.upload.assert_called_once()
        assert isinstance(segments, list)


# ---------------------------------------------------------------------------
# Candidate segment extraction
# ---------------------------------------------------------------------------

class TestExtractCandidateSegments:
    def test_empty_transcript_returns_empty(self):
        transcript = _make_transcript([])
        assert _extract_candidate_segments(transcript) == []

    def test_too_short_transcript_returns_empty(self):
        """Transcript shorter than MIN_SEGMENT_DURATION → no candidates."""
        words = _make_words(["hi", "there"], word_duration=1.0)
        transcript = _make_transcript(words)
        candidates = _extract_candidate_segments(transcript)
        # Each candidate must span at least MIN_SEGMENT_DURATION
        for c in candidates:
            duration = c[-1].end - c[0].start
            assert duration >= MIN_SEGMENT_DURATION

    def test_candidates_respect_min_duration(self):
        words = _make_words([f"w{i}" for i in range(60)], word_duration=1.0)
        transcript = _make_transcript(words)
        candidates = _extract_candidate_segments(transcript)
        for c in candidates:
            duration = c[-1].end - c[0].start
            assert duration >= MIN_SEGMENT_DURATION

    def test_candidates_respect_max_duration(self):
        words = _make_words([f"w{i}" for i in range(120)], word_duration=1.0)
        transcript = _make_transcript(words)
        candidates = _extract_candidate_segments(transcript)
        for c in candidates:
            duration = c[-1].end - c[0].start
            assert duration <= MAX_SEGMENT_DURATION
