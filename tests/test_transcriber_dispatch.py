"""Unit tests for Transcriber.transcribe() provider dispatch (Task 2.4)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models import JobConfig, Transcript, WordToken
from app.workers.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_transcriber(job_id: str = "job-test") -> Transcriber:
    storage = MagicMock()
    storage.download_file.return_value = None  # no-op download
    return Transcriber(job_id=job_id, storage=storage)


def _whisper_result(words: list[dict]) -> dict:
    """Build a minimal Whisper result dict."""
    return {"segments": [{"words": words}]}


def _aai_word(text: str, start_ms: int, end_ms: int, confidence: float) -> MagicMock:
    w = MagicMock()
    w.text = text
    w.start = start_ms
    w.end = end_ms
    w.confidence = confidence
    return w


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

class TestTranscribeDispatch:
    def test_whisper_provider_is_called_for_whisper_config(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="whisper")

        with patch.object(t, "_transcribe_whisper", return_value=Transcript(job_id="job-test", words=[], is_empty=True)) as mock_w, \
             patch.object(t, "_transcribe_assemblyai") as mock_a:
            t.transcribe("jobs/job-test/audio.aac", config)

        mock_w.assert_called_once()
        mock_a.assert_not_called()

    def test_assemblyai_provider_is_called_for_assemblyai_config(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="assemblyai")

        with patch.object(t, "_transcribe_assemblyai", return_value=Transcript(job_id="job-test", words=[], is_empty=True)) as mock_a, \
             patch.object(t, "_transcribe_whisper") as mock_w:
            t.transcribe("jobs/job-test/audio.aac", config)

        mock_a.assert_called_once()
        mock_w.assert_not_called()

    def test_transcribe_downloads_audio_before_dispatch(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="whisper")

        with patch.object(t, "_transcribe_whisper", return_value=Transcript(job_id="job-test", words=[], is_empty=True)):
            t.transcribe("jobs/job-test/audio.aac", config)

        t.storage.download_file.assert_called_once()
        call_args = t.storage.download_file.call_args[0]
        assert call_args[0] == "jobs/job-test/audio.aac"


# ---------------------------------------------------------------------------
# Whisper provider
# ---------------------------------------------------------------------------

def _mock_whisper_module(model: MagicMock) -> dict:
    """Return a sys.modules patch dict for the whisper library."""
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = model
    return {"whisper": mock_whisper}


class TestTranscribeWhisper:
    def test_returns_transcript_with_word_tokens(self):
        t = _make_transcriber("job-w1")
        whisper_result = _whisper_result([
            {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
            {"word": "world", "start": 0.6, "end": 1.1, "probability": 0.95},
        ])

        mock_model = MagicMock()
        mock_model.transcribe.return_value = whisper_result

        with patch.dict("sys.modules", _mock_whisper_module(mock_model)):
            transcript = t._transcribe_whisper("/tmp/audio.aac")

        assert isinstance(transcript, Transcript)
        assert transcript.job_id == "job-w1"
        assert len(transcript.words) == 2
        assert transcript.is_empty is False

        assert transcript.words[0].word == "hello"
        assert transcript.words[0].start == pytest.approx(0.0)
        assert transcript.words[0].end == pytest.approx(0.5)
        assert transcript.words[0].confidence == pytest.approx(0.99)

    def test_empty_audio_returns_empty_transcript(self):
        t = _make_transcriber("job-w2")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        with patch.dict("sys.modules", _mock_whisper_module(mock_model)):
            transcript = t._transcribe_whisper("/tmp/silent.aac")

        assert transcript.is_empty is True
        assert transcript.words == []

    def test_whisper_called_with_word_timestamps(self):
        t = _make_transcriber("job-w3")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        with patch.dict("sys.modules", _mock_whisper_module(mock_model)):
            t._transcribe_whisper("/tmp/audio.aac")

        mock_model.transcribe.assert_called_once_with("/tmp/audio.aac", word_timestamps=True)

    def test_whisper_loads_base_model(self):
        t = _make_transcriber("job-w4")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t._transcribe_whisper("/tmp/audio.aac")

        mock_whisper.load_model.assert_called_once_with("base")

    def test_job_id_set_on_transcript(self):
        t = _make_transcriber("my-special-job")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        with patch.dict("sys.modules", _mock_whisper_module(mock_model)):
            transcript = t._transcribe_whisper("/tmp/audio.aac")

        assert transcript.job_id == "my-special-job"


# ---------------------------------------------------------------------------
# AssemblyAI provider
# ---------------------------------------------------------------------------

class TestTranscribeAssemblyAI:
    def test_returns_transcript_with_word_tokens(self):
        t = _make_transcriber("job-a1")
        aai_words = [
            _aai_word("hello", 0, 500, 0.98),
            _aai_word("world", 600, 1100, 0.93),
        ]
        mock_result = MagicMock()
        mock_result.words = aai_words

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        mock_aai = MagicMock()
        mock_aai.Transcriber.return_value = mock_transcriber
        mock_aai.TranscriptionConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            transcript = t._transcribe_assemblyai("/tmp/audio.aac")

        assert isinstance(transcript, Transcript)
        assert transcript.job_id == "job-a1"
        assert len(transcript.words) == 2
        assert transcript.is_empty is False

        # AssemblyAI ms → seconds conversion
        assert transcript.words[0].word == "hello"
        assert transcript.words[0].start == pytest.approx(0.0)
        assert transcript.words[0].end == pytest.approx(0.5)
        assert transcript.words[0].confidence == pytest.approx(0.98)

        assert transcript.words[1].start == pytest.approx(0.6)
        assert transcript.words[1].end == pytest.approx(1.1)

    def test_empty_audio_returns_empty_transcript(self):
        t = _make_transcriber("job-a2")
        mock_result = MagicMock()
        mock_result.words = []

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        mock_aai = MagicMock()
        mock_aai.Transcriber.return_value = mock_transcriber
        mock_aai.TranscriptionConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            transcript = t._transcribe_assemblyai("/tmp/silent.aac")

        assert transcript.is_empty is True
        assert transcript.words == []

    def test_assemblyai_configured_with_speaker_labels_false(self):
        t = _make_transcriber("job-a3")
        mock_result = MagicMock()
        mock_result.words = []

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        mock_aai = MagicMock()
        mock_aai.Transcriber.return_value = mock_transcriber

        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            t._transcribe_assemblyai("/tmp/audio.aac")

        mock_aai.TranscriptionConfig.assert_called_once_with(speaker_labels=False)

    def test_assemblyai_none_words_returns_empty_transcript(self):
        """result.words may be None when no speech detected."""
        t = _make_transcriber("job-a4")
        mock_result = MagicMock()
        mock_result.words = None

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        mock_aai = MagicMock()
        mock_aai.Transcriber.return_value = mock_transcriber
        mock_aai.TranscriptionConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            transcript = t._transcribe_assemblyai("/tmp/audio.aac")

        assert transcript.is_empty is True
        assert transcript.words == []


# ---------------------------------------------------------------------------
# Task 2.8: Empty transcript — is_empty=True and warning emitted
# Requirements: 1.4
# ---------------------------------------------------------------------------

class TestEmptyTranscript:
    """Requirement 1.4: IF the audio track contains no detectable speech,
    THEN the Transcriber SHALL mark the Transcript as empty and emit a
    warning event to the Orchestrator.
    """

    def test_whisper_silent_audio_sets_is_empty_true(self):
        """Silent audio (no segments) → Transcript.is_empty is True."""
        t = _make_transcriber("job-empty-1")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        with patch.dict("sys.modules", _mock_whisper_module(mock_model)):
            transcript = t._transcribe_whisper("/tmp/silent.aac")

        assert transcript.is_empty is True
        assert transcript.words == []

    def test_assemblyai_silent_audio_sets_is_empty_true(self):
        """Silent audio (empty words list) → Transcript.is_empty is True."""
        t = _make_transcriber("job-empty-2")
        mock_result = MagicMock()
        mock_result.words = []

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        mock_aai = MagicMock()
        mock_aai.Transcriber.return_value = mock_transcriber
        mock_aai.TranscriptionConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            transcript = t._transcribe_assemblyai("/tmp/silent.aac")

        assert transcript.is_empty is True
        assert transcript.words == []

    def test_transcribe_emits_warning_when_transcript_is_empty(self):
        """transcribe() must log a warning when the resulting transcript is empty."""
        t = _make_transcriber("job-empty-3")
        config = JobConfig(transcription_provider="whisper")
        empty_transcript = Transcript(job_id="job-empty-3", words=[], is_empty=True)

        with patch.object(t, "_transcribe_whisper", return_value=empty_transcript):
            import logging
            with patch.object(logging.getLogger("app.workers.transcriber"), "warning") as mock_warn:
                result = t.transcribe("jobs/job-empty-3/audio.aac", config)

        assert result.is_empty is True
        mock_warn.assert_called_once()
        warning_msg = mock_warn.call_args[0][0]
        assert "empty" in warning_msg.lower() or "Empty" in warning_msg

    def test_transcribe_no_warning_when_transcript_has_words(self):
        """transcribe() must NOT emit an empty-transcript warning when words are present."""
        t = _make_transcriber("job-nonempty-1")
        config = JobConfig(transcription_provider="whisper")
        non_empty = Transcript(
            job_id="job-nonempty-1",
            words=[WordToken(word="hello", start=0.0, end=0.5, confidence=0.99)],
            is_empty=False,
        )

        with patch.object(t, "_transcribe_whisper", return_value=non_empty):
            import logging
            with patch.object(logging.getLogger("app.workers.transcriber"), "warning") as mock_warn:
                result = t.transcribe("jobs/job-nonempty-1/audio.aac", config)

        assert result.is_empty is False
        mock_warn.assert_not_called()

    def test_transcribe_returns_empty_transcript_with_is_empty_true(self):
        """End-to-end: transcribe() returns Transcript with is_empty=True for silent audio."""
        t = _make_transcriber("job-empty-4")
        config = JobConfig(transcription_provider="assemblyai")
        empty_transcript = Transcript(job_id="job-empty-4", words=[], is_empty=True)

        with patch.object(t, "_transcribe_assemblyai", return_value=empty_transcript):
            result = t.transcribe("jobs/job-empty-4/audio.aac", config)

        assert result.is_empty is True
        assert result.words == []
        assert result.job_id == "job-empty-4"
