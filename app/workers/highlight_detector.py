"""Highlight Detector worker: segment scoring and ranking pipeline stage."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional

from app.models import Segment, Transcript, WordToken
from app.storage.base import CloudStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic scoring constants
# ---------------------------------------------------------------------------

HIGH_SIGNAL_KEYWORDS = {
    "secret", "crazy", "important", "shocking", "unbelievable",
    "never", "always", "best", "worst", "amazing", "incredible",
    "insane", "huge", "massive", "critical", "urgent", "breaking",
    "exclusive", "revealed", "truth", "lie", "fake", "real",
    "dangerous", "powerful", "ultimate", "perfect", "terrible",
}

STRONG_ADJECTIVES = {
    "incredible", "unbelievable", "extraordinary", "phenomenal",
    "outstanding", "remarkable", "spectacular", "magnificent",
    "devastating", "catastrophic", "revolutionary", "groundbreaking",
}

# Preferred sentence length range (word count)
_MIN_PREFERRED_WORDS = 3
_MAX_PREFERRED_WORDS = 10

# Segment duration constraints
MIN_SEGMENT_DURATION = 15.0   # seconds
MAX_SEGMENT_DURATION = 60.0   # seconds

# Composite score weights
_HEURISTIC_WEIGHT = 0.5
_LLM_WEIGHT = 0.5

# Max segments to return
MAX_SEGMENTS = 10


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

def _score_heuristic(words: List[WordToken]) -> float:
    """Score a list of WordTokens using heuristic signals.

    Signals:
    - Keyword presence: high-signal words boost score
    - Sentence length: prefer 3–10 word utterances
    - Question detection: sentences ending with "?" score higher
    - Emotional intensity: exclamation marks, capitalized words, strong adjectives

    Returns a float in [0.0, 1.0].
    """
    if not words:
        return 0.0

    score = 0.0
    word_texts = [w.word for w in words]
    full_text = " ".join(word_texts)
    word_count = len(word_texts)

    # --- Keyword presence ---
    keyword_hits = sum(
        1 for w in word_texts if w.lower().strip(".,!?;:\"'") in HIGH_SIGNAL_KEYWORDS
    )
    # Each keyword hit adds 0.15, capped at 0.45
    score += min(keyword_hits * 0.15, 0.45)

    # --- Sentence length ---
    if _MIN_PREFERRED_WORDS <= word_count <= _MAX_PREFERRED_WORDS:
        score += 0.20
    elif word_count < _MIN_PREFERRED_WORDS:
        # Very short utterances are less useful
        score += 0.05
    else:
        # Long utterances get a small bonus (still content-rich)
        score += 0.10

    # --- Question detection ---
    stripped = full_text.strip()
    if stripped.endswith("?") or any(w.strip().endswith("?") for w in word_texts):
        score += 0.15

    # --- Emotional intensity ---
    # Exclamation marks
    exclamation_count = full_text.count("!")
    score += min(exclamation_count * 0.05, 0.15)

    # Capitalized words (ALL CAPS, not just title-case start of sentence)
    caps_words = [
        w for w in word_texts
        if len(w) > 1 and w.isupper() and w.isalpha()
    ]
    score += min(len(caps_words) * 0.05, 0.10)

    # Strong adjectives
    strong_adj_hits = sum(
        1 for w in word_texts if w.lower().strip(".,!?;:\"'") in STRONG_ADJECTIVES
    )
    score += min(strong_adj_hits * 0.10, 0.20)

    # Clamp to [0.0, 1.0]
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Candidate segment extraction
# ---------------------------------------------------------------------------

def _extract_candidate_segments(
    transcript: Transcript,
    target_duration: float = 30.0,
) -> List[List[WordToken]]:
    """Slide a window over the transcript words to produce candidate segments.

    Each candidate is a list of WordTokens whose span is between
    MIN_SEGMENT_DURATION and MAX_SEGMENT_DURATION seconds.

    Uses a greedy approach: start a new window at each word and extend it
    until the duration hits the target, then record it.
    """
    words = transcript.words
    if not words:
        return []

    candidates: List[List[WordToken]] = []
    n = len(words)

    i = 0
    while i < n:
        window: List[WordToken] = []
        j = i
        while j < n:
            window.append(words[j])
            duration = words[j].end - words[i].start
            if duration >= MIN_SEGMENT_DURATION:
                # Record this window if it doesn't exceed max
                if duration <= MAX_SEGMENT_DURATION:
                    candidates.append(list(window))
                break
            j += 1

        # Advance by roughly half the window to get overlapping candidates
        step = max(1, len(window) // 2) if window else 1
        i += step

    return candidates


# ---------------------------------------------------------------------------
# LLM virality scoring
# ---------------------------------------------------------------------------

def _score_llm_virality(text: str) -> float:
    """Call the Groq API to score *text* for virality.

    Returns a float in [0.0, 1.0].
    Falls back to 0.0 if the Groq API is unavailable or returns an error.
    """
    try:
        from groq import Groq  # type: ignore[import]

        client = Groq()
        prompt = (
            "You are a social media virality expert. "
            "Rate the following transcript segment for its potential to go viral "
            "on short-form video platforms (Instagram Reels, TikTok, YouTube Shorts). "
            "Respond with ONLY a single number between 0 and 10, where 0 is not viral "
            "and 10 is extremely viral. Do not include any other text.\n\n"
            f"Transcript segment:\n{text}"
        )

        completion = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
            compound_custom={
                "tools": {
                    "enabled_tools": ["web_search", "code_interpreter", "visit_website"]
                }
            },
        )

        response_text = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                response_text += delta

        # Parse the numeric score from the response
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response_text.strip())
        if match:
            raw_score = float(match.group(1))
            # Normalize from 0–10 to 0.0–1.0
            return min(max(raw_score / 10.0, 0.0), 1.0)

        logger.warning("Could not parse LLM virality score from response: %r", response_text)
        return 0.0

    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq API unavailable or error; falling back to 0.0 LLM score: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main HighlightDetector class
# ---------------------------------------------------------------------------

class HighlightDetector:
    """Reads a Transcript, scores candidate segments, and returns ranked Segments."""

    def __init__(self, job_id: str, storage: CloudStorage) -> None:
        self.job_id = job_id
        self.storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, transcript: Transcript) -> List[Segment]:
        """Detect and rank highlight segments from *transcript*.

        Returns:
            A list of 0–10 :class:`~app.models.Segment` objects ranked by
            composite score descending.  Returns an empty list when the
            transcript is empty.
        """
        if transcript.is_empty or not transcript.words:
            logger.warning(
                "Empty transcript for job %s: returning 0 segments and notifying orchestrator.",
                self.job_id,
            )
            return []

        candidates = _extract_candidate_segments(transcript)
        if not candidates:
            logger.warning(
                "No valid candidate segments found for job %s (transcript too short?).",
                self.job_id,
            )
            return []

        scored: List[tuple[float, float, List[WordToken]]] = []
        for word_window in candidates:
            heuristic = _score_heuristic(word_window)
            text = " ".join(w.word for w in word_window)
            llm_score = _score_llm_virality(text)
            composite = _HEURISTIC_WEIGHT * heuristic + _LLM_WEIGHT * llm_score
            scored.append((composite, llm_score, word_window))

        # Sort by composite score descending
        scored.sort(key=lambda t: t[0], reverse=True)

        # Take top MAX_SEGMENTS, ensure at least 1
        top = scored[:MAX_SEGMENTS]

        segments: List[Segment] = []
        for rank, (composite, llm_score, word_window) in enumerate(top, start=1):
            start = word_window[0].start
            end = word_window[-1].end
            duration = end - start

            # Clamp duration to [MIN, MAX] — should already be in range from
            # candidate extraction, but enforce defensively
            if duration < MIN_SEGMENT_DURATION:
                end = start + MIN_SEGMENT_DURATION
            elif duration > MAX_SEGMENT_DURATION:
                end = start + MAX_SEGMENT_DURATION

            segments.append(
                Segment(
                    segment_id=str(uuid.uuid4()),
                    job_id=self.job_id,
                    start=start,
                    end=end,
                    score=round(composite, 4),
                    llm_virality_score=round(llm_score, 4),
                    rank=rank,
                )
            )

        return segments

    def write_segments(self, segments: List[Segment]) -> str:
        """Serialize *segments* to JSON and write to cloud storage.

        Returns the cloud storage key where the data was written
        (``jobs/{job_id}/segments.json``).
        """
        key = CloudStorage.segments_key(self.job_id)
        data = json.dumps([s.model_dump() for s in segments], indent=2).encode("utf-8")
        self.storage.upload(key, data, content_type="application/json")
        logger.info("Wrote %d segments to %s", len(segments), key)
        return key

    def run(self) -> List[Segment]:
        """Full pipeline: read transcript → detect segments → write segments.json.

        Returns the list of detected segments.
        """
        transcript_key = CloudStorage.transcript_key(self.job_id)
        logger.info("Reading transcript from %s", transcript_key)
        raw = self.storage.download(transcript_key)
        transcript_data = json.loads(raw.decode("utf-8"))
        transcript = Transcript(**transcript_data)

        segments = self.detect(transcript)
        self.write_segments(segments)
        return segments


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

def _make_storage() -> CloudStorage:
    """Instantiate the configured cloud storage backend from environment variables."""
    import os

    from app.storage.gcs import GCSStorage
    from app.storage.s3 import S3Storage

    storage_backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
    if storage_backend == "gcs":
        bucket = os.environ.get("GCS_BUCKET", "video-processing-engine")
        return GCSStorage(bucket=bucket)
    bucket = os.environ.get("S3_BUCKET", "video-processing-engine")
    return S3Storage(bucket=bucket)


try:
    from app.celery_app import celery_app

    @celery_app.task(
        name="app.workers.highlight_detector.detect_highlights",
        bind=True,
        max_retries=3,
        default_retry_delay=1,
    )
    def detect_highlights(self, job_id: str) -> dict:
        """Celery task: detect highlight segments for *job_id*.

        Reads ``jobs/{job_id}/transcript.json`` from cloud storage, scores
        candidate segments, and writes ``jobs/{job_id}/segments.json``.

        Returns a dict with ``job_id`` and ``segment_count``.
        """
        storage = _make_storage()
        try:
            detector = HighlightDetector(job_id=job_id, storage=storage)
            segments = detector.run()
            return {"job_id": job_id, "segment_count": len(segments)}
        except Exception as exc:
            logger.error("Highlight detection failed for job %s: %s", job_id, exc)
            raise self.retry(exc=exc)

except ImportError:
    # Celery not available (e.g., during unit tests without the celery package)
    logger.debug("Celery not available; detect_highlights task not registered.")
