"""Transcriber worker: audio extraction and transcription pipeline stage."""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from app.models import JobConfig, Transcript, WordToken
from app.storage.base import CloudStorage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when a transcription provider returns an error."""


class TranscriptionExhaustedError(Exception):
    """Raised after all retry attempts for a transcription provider are exhausted."""


class AudioExtractionError(Exception):
    """Raised when FFmpeg exits with a non-zero return code during audio extraction."""


_RETRY_DELAYS = [1, 2, 4]  # exponential backoff: 2^0, 2^1, 2^2 seconds


def build_audio_extraction_cmd(input_path: str, output_path: str) -> list[str]:
    """Return the FFmpeg command list for audio-only extraction.

    Flags used:
      -y            overwrite output without prompting
      -i <input>    input file
      -vn           disable video recording (no video stream in output)
      -acodec copy  copy audio stream without re-encoding

    Deliberately omits any -c:v flag so the video stream is never re-encoded.
    """
    return ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "copy", output_path]


class Transcriber:
    """Downloads a source video, extracts its audio track, and uploads the result.

    Later tasks (2.4, 2.6, 2.8) will extend this class with:
      - configurable transcription provider dispatch (Whisper / AssemblyAI)
      - exponential-backoff retry logic
      - empty-transcript detection and warning emission
    """

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage
        self._sleep = time.sleep  # can be replaced in tests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_audio(self, video_key: str) -> str:
        """Download *video_key* from cloud storage, extract audio, upload result.

        Returns the cloud storage key where the extracted audio was written
        (``jobs/{job_id}/audio.aac``).

        Raises:
            AudioExtractionError: if FFmpeg exits with a non-zero return code.
        """
        audio_key = CloudStorage.audio_key(self.job_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = str(tmp / "source_video")
            audio_path = str(tmp / "audio.aac")

            logger.info("Downloading video %s for job %s", video_key, self.job_id)
            self.storage.download_file(video_key, video_path)

            self._run_ffmpeg(video_path, audio_path)

            logger.info("Uploading extracted audio to %s", audio_key)
            self.storage.upload_file(audio_key, audio_path, content_type="audio/aac")

        return audio_key

    def transcribe(self, audio_key: str, config: JobConfig) -> Transcript:
        """Download *audio_key* from cloud storage and transcribe using the configured provider.

        Dispatches to Whisper or AssemblyAI based on ``config.transcription_provider``.
        Retries up to 3 times with exponential backoff (1s, 2s, 4s) on provider error.

        Returns:
            A :class:`~app.models.Transcript` with word-level timestamps.

        Raises:
            TranscriptionExhaustedError: if all 3 retry attempts fail.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "audio.aac")
            logger.info("Downloading audio %s for job %s", audio_key, self.job_id)
            self.storage.download_file(audio_key, audio_path)

            if config.transcription_provider == "whisper":
                provider_fn = self._transcribe_whisper
            elif config.transcription_provider == "assemblyai":
                provider_fn = self._transcribe_assemblyai
            else:
                raise ValueError(
                    f"Unknown transcription provider: {config.transcription_provider!r}"
                )

            transcript = self._call_provider_with_retry(provider_fn, audio_path)

        if transcript.is_empty:
            logger.warning(
                "Empty transcript for job %s: no detectable speech found in audio. "
                "Emitting warning to orchestrator.",
                self.job_id,
            )

        return transcript

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _call_provider_with_retry(
        self, provider_fn: Callable[[str], Transcript], audio_path: str
    ) -> Transcript:
        """Call *provider_fn(audio_path)* up to 3 times with exponential backoff.

        Delays between attempts: 1s, 2s, 4s (applied *before* each retry, not before
        the first attempt).

        Raises:
            TranscriptionExhaustedError: after all 3 attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_DELAYS)):
            if attempt > 0:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "Transcription attempt %d failed for job %s; retrying in %ds",
                    attempt,
                    self.job_id,
                    delay,
                )
                self._sleep(delay)
            try:
                return provider_fn(audio_path)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.error(
                    "Transcription provider error on attempt %d for job %s: %s",
                    attempt + 1,
                    self.job_id,
                    exc,
                )

        raise TranscriptionExhaustedError(
            f"Transcription failed after {len(_RETRY_DELAYS)} attempts for job {self.job_id}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Private provider implementations
    # ------------------------------------------------------------------

    def _transcribe_whisper(self, audio_path: str) -> Transcript:
        """Transcribe *audio_path* using the local Whisper model.

        Uses ``whisper.load_model("base")`` with ``word_timestamps=True``.
        """
        import whisper  # type: ignore[import]

        logger.info("Transcribing with Whisper for job %s", self.job_id)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)

        words: list[WordToken] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(
                    WordToken(
                        word=w["word"],
                        start=float(w["start"]),
                        end=float(w["end"]),
                        confidence=float(w.get("probability", 1.0)),
                    )
                )

        return Transcript(
            job_id=self.job_id,
            words=words,
            is_empty=len(words) == 0,
        )

    def _transcribe_assemblyai(self, audio_path: str) -> Transcript:
        """Transcribe *audio_path* using the AssemblyAI API.

        Uses ``assemblyai.Transcriber`` with ``speaker_labels=False``.
        """
        import assemblyai  # type: ignore[import]

        logger.info("Transcribing with AssemblyAI for job %s", self.job_id)
        aai_config = assemblyai.TranscriptionConfig(speaker_labels=False)
        transcriber = assemblyai.Transcriber()
        result = transcriber.transcribe(audio_path, config=aai_config)

        words: list[WordToken] = []
        for w in result.words or []:
            words.append(
                WordToken(
                    word=w.text,
                    start=w.start / 1000.0,  # AssemblyAI returns milliseconds
                    end=w.end / 1000.0,
                    confidence=float(w.confidence),
                )
            )

        return Transcript(
            job_id=self.job_id,
            words=words,
            is_empty=len(words) == 0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, input_path: str, output_path: str) -> None:
        """Build and execute the FFmpeg audio-extraction command.

        Raises:
            AudioExtractionError: if FFmpeg returns a non-zero exit code.
        """
        cmd = build_audio_extraction_cmd(input_path, output_path)
        logger.debug("Running FFmpeg command: %s", cmd)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(
                "FFmpeg failed (exit %d) for job %s:\n%s",
                result.returncode,
                self.job_id,
                result.stderr,
            )
            raise AudioExtractionError(
                f"FFmpeg exited with code {result.returncode} for job {self.job_id}. "
                f"stderr: {result.stderr}"
            )
