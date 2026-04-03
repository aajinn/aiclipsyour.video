"""Property-based test for cloud storage path convention (Task 14.1, Property 29).

**Validates: Requirements 10.4**
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import JobConfig, Segment
from app.storage.base import CloudStorage
from app.workers.audio_optimizer import AudioOptimizer
from app.workers.caption_generator import CaptionGenerator
from app.workers.clip_extractor import ClipExtractor
from app.workers.export_engine import ExportEngine
from app.workers.format_optimizer import FormatOptimizer
from app.workers.highlight_detector import HighlightDetector
from app.workers.transcriber import Transcriber
from app.workers.visual_enhancer import VisualEnhancer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOB_ID_ALPHABET = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"),
    whitelist_characters="-",
)

_job_id_st = st.text(alphabet=_JOB_ID_ALPHABET, min_size=1, max_size=30)
_segment_id_st = st.text(alphabet=_JOB_ID_ALPHABET, min_size=1, max_size=30)

_CLOUD_KEY_PATTERN = re.compile(r"^jobs/[^/]+/")


def _make_storage() -> MagicMock:
    """Return a mock CloudStorage that records all key arguments."""
    storage = MagicMock(spec=CloudStorage)
    storage.download_file.return_value = None
    storage.upload_file.return_value = None
    storage.download.return_value = b"[]"
    storage.upload.return_value = None
    storage.generate_signed_url.return_value = "https://example.com/signed"
    return storage


def _segment(job_id: str, segment_id: str) -> Segment:
    return Segment(
        segment_id=segment_id,
        job_id=job_id,
        start=0.0,
        end=30.0,
        score=0.8,
        llm_virality_score=0.7,
        rank=1,
    )


def _config(**kwargs) -> JobConfig:
    return JobConfig(**kwargs)


def _ok_run() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


def _collect_storage_keys(storage: MagicMock) -> list[str]:
    """Collect all key arguments passed to storage read/write methods."""
    keys: list[str] = []
    for method_name in ("download_file", "upload_file", "download", "upload"):
        method = getattr(storage, method_name)
        for call in method.call_args_list:
            # First positional arg is always the key
            if call.args:
                keys.append(call.args[0])
    return keys


# ---------------------------------------------------------------------------
# Property 29: All artifacts use cloud storage paths
# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
# ---------------------------------------------------------------------------

# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_transcriber_audio_key(job_id: str, segment_id: str):
    """Transcriber.extract_audio writes to a jobs/{job_id}/... key.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    t = Transcriber(job_id=job_id, storage=storage)

    with patch("subprocess.run", return_value=_ok_run()):
        t.extract_audio(f"jobs/{job_id}/source.mp4")

    keys = _collect_storage_keys(storage)
    # All keys must match the convention
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"Transcriber used non-convention key: {key!r}"
        )
    # The written audio key must be jobs/{job_id}/audio.aac
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/audio.aac" for k in upload_keys), (
        f"Expected jobs/{job_id}/audio.aac in upload keys, got {upload_keys}"
    )


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_clip_extractor_keys(job_id: str, segment_id: str):
    """ClipExtractor reads source and writes raw clip using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    extractor = ClipExtractor(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)

    with patch("subprocess.run", return_value=_ok_run()):
        extractor.extract_clip(seg)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"ClipExtractor used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/clips/{segment_id}/raw.mp4" for k in upload_keys), (
        f"Expected raw.mp4 key in uploads, got {upload_keys}"
    )


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_format_optimizer_keys(job_id: str, segment_id: str):
    """FormatOptimizer reads raw and writes formatted clip using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    optimizer = FormatOptimizer(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)
    config = _config()

    with patch("subprocess.run", return_value=_ok_run()):
        optimizer.optimize(seg, config)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"FormatOptimizer used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/clips/{segment_id}/formatted.mp4" for k in upload_keys), (
        f"Expected formatted.mp4 key in uploads, got {upload_keys}"
    )


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_audio_optimizer_keys(job_id: str, segment_id: str):
    """AudioOptimizer reads enhanced and writes audio_optimized using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    optimizer = AudioOptimizer(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)
    config = _config()

    with patch("subprocess.run", return_value=_ok_run()):
        optimizer.optimize(seg, config)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"AudioOptimizer used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(
        k == f"jobs/{job_id}/clips/{segment_id}/audio_optimized.mp4" for k in upload_keys
    ), f"Expected audio_optimized.mp4 key in uploads, got {upload_keys}"


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_export_engine_keys(job_id: str, segment_id: str):
    """ExportEngine reads audio_optimized and writes final clip using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    engine = ExportEngine(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)

    with patch("subprocess.run", return_value=_ok_run()):
        engine.export(seg)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"ExportEngine used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/clips/{segment_id}/final.mp4" for k in upload_keys), (
        f"Expected final.mp4 key in uploads, got {upload_keys}"
    )


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_visual_enhancer_keys(job_id: str, segment_id: str):
    """VisualEnhancer reads captioned and writes enhanced clip using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    storage = _make_storage()
    enhancer = VisualEnhancer(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)
    config = _config()

    with patch("subprocess.run", return_value=_ok_run()):
        enhancer.enhance(seg, config)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"VisualEnhancer used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/clips/{segment_id}/enhanced.mp4" for k in upload_keys), (
        f"Expected enhanced.mp4 key in uploads, got {upload_keys}"
    )


# Feature: video-processing-engine, Property 29: All artifacts use cloud storage paths
@given(job_id=_job_id_st, segment_id=_segment_id_st)
@settings(max_examples=100)
def test_property_29_caption_generator_keys(job_id: str, segment_id: str):
    """CaptionGenerator reads formatted and writes captioned clip using jobs/{job_id}/... keys.

    **Validates: Requirements 10.4**
    """
    from app.models import Transcript

    storage = _make_storage()
    generator = CaptionGenerator(job_id=job_id, storage=storage)
    seg = _segment(job_id, segment_id)
    transcript = Transcript(job_id=job_id, words=[], is_empty=True)
    config = _config()

    with patch("subprocess.run", return_value=_ok_run()):
        generator.generate(seg, transcript, config)

    keys = _collect_storage_keys(storage)
    for key in keys:
        assert _CLOUD_KEY_PATTERN.match(key), (
            f"CaptionGenerator used non-convention key: {key!r}"
        )
    upload_keys = [c.args[0] for c in storage.upload_file.call_args_list]
    assert any(k == f"jobs/{job_id}/clips/{segment_id}/captioned.mp4" for k in upload_keys), (
        f"Expected captioned.mp4 key in uploads, got {upload_keys}"
    )


# ---------------------------------------------------------------------------
# CloudStorage key helper tests (unit)
# ---------------------------------------------------------------------------

class TestCloudStorageKeyHelpers:
    """Verify the static key helpers always produce jobs/{job_id}/... paths."""

    def test_audio_key_convention(self):
        key = CloudStorage.audio_key("my-job")
        assert key == "jobs/my-job/audio.aac"
        assert _CLOUD_KEY_PATTERN.match(key)

    def test_transcript_key_convention(self):
        key = CloudStorage.transcript_key("my-job")
        assert key == "jobs/my-job/transcript.json"
        assert _CLOUD_KEY_PATTERN.match(key)

    def test_segments_key_convention(self):
        key = CloudStorage.segments_key("my-job")
        assert key == "jobs/my-job/segments.json"
        assert _CLOUD_KEY_PATTERN.match(key)

    @pytest.mark.parametrize("stage", ["raw", "formatted", "captioned", "enhanced", "audio_optimized", "final"])
    def test_clip_key_convention(self, stage: str):
        key = CloudStorage.clip_key("my-job", "seg-1", stage)
        assert key == f"jobs/my-job/clips/seg-1/{stage}.mp4"
        assert _CLOUD_KEY_PATTERN.match(key)
