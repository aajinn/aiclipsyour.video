"""Unit and property-based tests for Transcriber retry logic (Task 2.6)."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import JobConfig, Transcript, WordToken
from app.workers.transcriber import (
    Transcriber,
    TranscriptionExhaustedError,
    _RETRY_DELAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcriber(job_id: str = "job-retry-test") -> Transcriber:
    storage = MagicMock()
    storage.download_file.return_value = None
    t = Transcriber(job_id=job_id, storage=storage)
    t._sleep = MagicMock()  # no-op sleep for tests
    return t


def _empty_transcript(job_id: str = "job-retry-test") -> Transcript:
    return Transcript(job_id=job_id, words=[], is_empty=True)


# ---------------------------------------------------------------------------
# Unit tests: _call_provider_with_retry
# ---------------------------------------------------------------------------

class TestCallProviderWithRetry:
    def test_succeeds_on_first_attempt(self):
        t = _make_transcriber()
        provider = MagicMock(return_value=_empty_transcript())

        result = t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert isinstance(result, Transcript)
        provider.assert_called_once_with("/tmp/audio.aac")
        t._sleep.assert_not_called()

    def test_retries_on_failure_and_succeeds_on_second_attempt(self):
        t = _make_transcriber()
        provider = MagicMock(side_effect=[RuntimeError("fail"), _empty_transcript()])

        result = t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert isinstance(result, Transcript)
        assert provider.call_count == 2
        # Sleep called once before the retry with 1s delay
        t._sleep.assert_called_once_with(1)

    def test_retries_on_failure_and_succeeds_on_third_attempt(self):
        t = _make_transcriber()
        provider = MagicMock(
            side_effect=[RuntimeError("fail1"), RuntimeError("fail2"), _empty_transcript()]
        )

        result = t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert isinstance(result, Transcript)
        assert provider.call_count == 3
        assert t._sleep.call_count == 2
        t._sleep.assert_any_call(1)
        t._sleep.assert_any_call(2)

    def test_raises_exhausted_after_three_failures(self):
        t = _make_transcriber()
        provider = MagicMock(side_effect=RuntimeError("always fails"))

        with pytest.raises(TranscriptionExhaustedError):
            t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert provider.call_count == 3

    def test_provider_called_exactly_three_times_before_exhaustion(self):
        """Provider must be called exactly 3 times total, not 4."""
        t = _make_transcriber()
        provider = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(TranscriptionExhaustedError):
            t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert provider.call_count == 3

    def test_retry_delays_are_exactly_1_2_4(self):
        """Delays must be exactly [1, 2, 4] seconds."""
        t = _make_transcriber()
        provider = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(TranscriptionExhaustedError):
            t._call_provider_with_retry(provider, "/tmp/audio.aac")

        sleep_calls = [c.args[0] for c in t._sleep.call_args_list]
        assert sleep_calls == [1, 2]  # sleep called before attempt 2 and 3

    def test_no_sleep_before_first_attempt(self):
        t = _make_transcriber()
        provider = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(TranscriptionExhaustedError):
            t._call_provider_with_retry(provider, "/tmp/audio.aac")

        # Only 2 sleeps for 3 attempts (no sleep before first attempt)
        assert t._sleep.call_count == 2

    def test_exhausted_error_chains_original_exception(self):
        t = _make_transcriber()
        original = RuntimeError("root cause")
        provider = MagicMock(side_effect=original)

        with pytest.raises(TranscriptionExhaustedError) as exc_info:
            t._call_provider_with_retry(provider, "/tmp/audio.aac")

        assert exc_info.value.__cause__ is original

    def test_retry_delays_constant_matches_spec(self):
        """_RETRY_DELAYS must be [1, 2, 4] as per the spec."""
        assert _RETRY_DELAYS == [1, 2, 4]


# ---------------------------------------------------------------------------
# Unit tests: transcribe() uses retry
# ---------------------------------------------------------------------------

class TestTranscribeUsesRetry:
    def test_transcribe_retries_whisper_on_error(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="whisper")
        success = _empty_transcript()

        with patch.object(
            t, "_transcribe_whisper", side_effect=[RuntimeError("fail"), success]
        ) as mock_w:
            result = t.transcribe("jobs/job-retry-test/audio.aac", config)

        assert result is success
        assert mock_w.call_count == 2
        t._sleep.assert_called_once_with(1)

    def test_transcribe_retries_assemblyai_on_error(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="assemblyai")
        success = _empty_transcript()

        with patch.object(
            t, "_transcribe_assemblyai", side_effect=[RuntimeError("fail"), success]
        ) as mock_a:
            result = t.transcribe("jobs/job-retry-test/audio.aac", config)

        assert result is success
        assert mock_a.call_count == 2

    def test_transcribe_raises_exhausted_after_three_failures(self):
        t = _make_transcriber()
        config = JobConfig(transcription_provider="whisper")

        with patch.object(t, "_transcribe_whisper", side_effect=RuntimeError("fail")):
            with pytest.raises(TranscriptionExhaustedError):
                t.transcribe("jobs/job-retry-test/audio.aac", config)


# ---------------------------------------------------------------------------
# Property-based test: Property 3 — Retry on provider error
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

# Feature: video-processing-engine, Property 3: Retry on provider error
@given(
    job_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-"),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_property_3_retry_on_provider_error(job_id: str):
    """For any transcription provider that always errors, the Transcriber invokes it
    exactly 3 times before raising TranscriptionExhaustedError.

    **Validates: Requirements 1.5**
    """
    storage = MagicMock()
    storage.download_file.return_value = None
    t = Transcriber(job_id=job_id, storage=storage)
    t._sleep = MagicMock()  # no-op sleep

    call_count = 0

    def always_fails(audio_path: str) -> Transcript:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("provider error")

    with pytest.raises(TranscriptionExhaustedError):
        t._call_provider_with_retry(always_fails, "/tmp/audio.aac")

    assert call_count == 3, f"Expected 3 provider calls, got {call_count}"

    sleep_calls = [c.args[0] for c in t._sleep.call_args_list]
    assert sleep_calls == [1, 2], f"Expected sleep delays [1, 2], got {sleep_calls}"
