from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class CloudStorage(ABC):
    """Abstract cloud storage interface.

    All artifact keys follow the convention:
        jobs/{job_id}/...
    e.g.
        jobs/{job_id}/audio.aac
        jobs/{job_id}/transcript.json
        jobs/{job_id}/segments.json
        jobs/{job_id}/clips/{segment_id}/raw.mp4
        jobs/{job_id}/clips/{segment_id}/formatted.mp4
        jobs/{job_id}/clips/{segment_id}/captioned.mp4
        jobs/{job_id}/clips/{segment_id}/enhanced.mp4
        jobs/{job_id}/clips/{segment_id}/audio_optimized.mp4
        jobs/{job_id}/clips/{segment_id}/final.mp4
    """

    # ------------------------------------------------------------------ #
    # Key helpers                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def audio_key(job_id: str) -> str:
        return f"jobs/{job_id}/audio.aac"

    @staticmethod
    def transcript_key(job_id: str) -> str:
        return f"jobs/{job_id}/transcript.json"

    @staticmethod
    def segments_key(job_id: str) -> str:
        return f"jobs/{job_id}/segments.json"

    @staticmethod
    def clip_key(job_id: str, segment_id: str, stage: str) -> str:
        """Return the cloud storage key for a clip at a given pipeline stage.

        stage must be one of: raw, formatted, captioned, enhanced,
        audio_optimized, final.
        """
        return f"jobs/{job_id}/clips/{segment_id}/{stage}.mp4"

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload *data* to *key*."""

    @abstractmethod
    def upload_file(self, key: str, local_path: str, content_type: str = "application/octet-stream") -> None:
        """Upload a local file at *local_path* to *key*."""

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download and return the raw bytes stored at *key*."""

    @abstractmethod
    def download_file(self, key: str, local_path: str) -> None:
        """Download the object at *key* to *local_path*."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object at *key*."""

    @abstractmethod
    def generate_signed_url(self, key: str, expiration_seconds: int = 86400) -> str:
        """Return a signed URL for *key* valid for *expiration_seconds* seconds.

        Default expiration is 24 hours (86 400 s) to satisfy Requirement 8.5.
        """
