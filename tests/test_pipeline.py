"""Unit tests for app/workers/pipeline.py — Tasks 15.1 and 15.2."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

JOB_ID = "test-job-pipeline-001"

CONFIG_DATA = {
    "transcription_provider": "whisper",
    "face_tracking": False,
    "noise_reduction": False,
    "caption_style": "default",
    "overlays": [],
    "progress_bar": {"color": "#FFFFFF", "height_px": 4},
    "dynamic_text_words": [],
}

SEGMENT_DATA = {
    "segment_id": "seg-001",
    "job_id": JOB_ID,
    "start": 0.0,
    "end": 30.0,
    "score": 0.8,
    "llm_virality_score": 0.7,
    "rank": 1,
}

TRANSCRIPT_DATA = {
    "job_id": JOB_ID,
    "words": [
        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
        {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.98},
    ],
    "is_empty": False,
}


def _make_redis_store(initial: dict | None = None) -> MagicMock:
    """Return a mock Redis client backed by an in-memory dict."""
    store: dict[str, str] = {}
    if initial:
        for k, v in initial.items():
            store[k] = json.dumps(v) if isinstance(v, dict) else v

    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.side_effect = lambda k: store[k].encode() if k in store else None
    mock.set.side_effect = lambda k, v: store.update({k: v if isinstance(v, str) else v.decode()})
    mock._store = store
    return mock


# ---------------------------------------------------------------------------
# Tests for transcribe_task (15.1)
# ---------------------------------------------------------------------------

class TestTranscribeTask:
    """transcribe_task calls Transcriber and chains to highlight_detect_task."""

    def test_calls_transcriber_extract_audio(self):
        """transcribe_task invokes Transcriber.extract_audio with the source video key."""
        redis_mock = _make_redis_store({f"job:{JOB_ID}": {"status": "queued", "progress": 0.0}})

        mock_transcriber = MagicMock()
        mock_transcriber.extract_audio.return_value = f"jobs/{JOB_ID}/audio.aac"
        mock_transcriber.transcribe.return_value = MagicMock(
            model_dump=lambda: TRANSCRIPT_DATA, is_empty=False
        )

        mock_storage = MagicMock()
        mock_storage.upload.return_value = None

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.Transcriber", return_value=mock_transcriber), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.highlight_detect_task") as mock_chain, \
             patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import transcribe_task
            transcribe_task(JOB_ID, "https://example.com/video.mp4", CONFIG_DATA)

        mock_transcriber.extract_audio.assert_called_once_with(f"jobs/{JOB_ID}/source.mp4")

    def test_calls_transcriber_transcribe(self):
        """transcribe_task invokes Transcriber.transcribe after audio extraction."""
        redis_mock = _make_redis_store({f"job:{JOB_ID}": {"status": "queued", "progress": 0.0}})

        mock_transcriber = MagicMock()
        mock_transcriber.extract_audio.return_value = f"jobs/{JOB_ID}/audio.aac"
        mock_transcriber.transcribe.return_value = MagicMock(
            model_dump=lambda: TRANSCRIPT_DATA, is_empty=False
        )
        mock_storage = MagicMock()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.Transcriber", return_value=mock_transcriber), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.highlight_detect_task"), \
             patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import transcribe_task
            transcribe_task(JOB_ID, "https://example.com/video.mp4", CONFIG_DATA)

        mock_transcriber.transcribe.assert_called_once()

    def test_chains_to_highlight_detect_task(self):
        """transcribe_task dispatches highlight_detect_task after transcription."""
        redis_mock = _make_redis_store({f"job:{JOB_ID}": {"status": "queued", "progress": 0.0}})

        mock_transcriber = MagicMock()
        mock_transcriber.extract_audio.return_value = f"jobs/{JOB_ID}/audio.aac"
        mock_transcriber.transcribe.return_value = MagicMock(
            model_dump=lambda: TRANSCRIPT_DATA, is_empty=False
        )
        mock_storage = MagicMock()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.Transcriber", return_value=mock_transcriber), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.highlight_detect_task") as mock_hd, \
             patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import transcribe_task
            transcribe_task(JOB_ID, "https://example.com/video.mp4", CONFIG_DATA)

        mock_hd.delay.assert_called_once_with(JOB_ID, CONFIG_DATA)

    def test_updates_progress_to_10_percent(self):
        """transcribe_task updates Redis progress to 10% on completion."""
        redis_mock = _make_redis_store({f"job:{JOB_ID}": {"status": "queued", "progress": 0.0}})

        mock_transcriber = MagicMock()
        mock_transcriber.extract_audio.return_value = f"jobs/{JOB_ID}/audio.aac"
        mock_transcriber.transcribe.return_value = MagicMock(
            model_dump=lambda: TRANSCRIPT_DATA, is_empty=False
        )
        mock_storage = MagicMock()

        progress_calls = []

        def capture_progress(job_id, progress, stage):
            progress_calls.append((job_id, progress, stage))

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.Transcriber", return_value=mock_transcriber), \
             patch("app.workers.pipeline.mark_job_progress", side_effect=capture_progress), \
             patch("app.workers.pipeline.highlight_detect_task"), \
             patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import transcribe_task
            transcribe_task(JOB_ID, "https://example.com/video.mp4", CONFIG_DATA)

        # Should have called mark_job_progress with 10.0 at some point
        assert any(p == 10.0 for _, p, _ in progress_calls)


# ---------------------------------------------------------------------------
# Tests for highlight_detect_task (15.1)
# ---------------------------------------------------------------------------

class TestHighlightDetectTask:
    """highlight_detect_task calls HighlightDetector and fans out to process_segment_task."""

    def _make_segment(self, seg_id: str = "seg-001"):
        from app.models import Segment
        return Segment(
            segment_id=seg_id,
            job_id=JOB_ID,
            start=0.0,
            end=30.0,
            score=0.8,
            llm_virality_score=0.7,
            rank=1,
        )

    def test_calls_highlight_detector_run(self):
        """highlight_detect_task invokes HighlightDetector.run()."""
        seg = self._make_segment()
        mock_detector = MagicMock()
        mock_detector.run.return_value = [seg]

        mock_extractor = MagicMock()
        mock_extractor.extract_all.return_value = {"seg-001": "jobs/x/clips/seg-001/raw.mp4"}

        mock_storage = MagicMock()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.HighlightDetector", return_value=mock_detector), \
             patch("app.workers.pipeline.ClipExtractor", return_value=mock_extractor), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.group"), \
             patch("app.workers.pipeline.chord") as mock_chord, \
             patch("app.workers.pipeline.process_segment_task"):
            mock_chord.return_value = MagicMock()
            from app.workers.pipeline import highlight_detect_task
            highlight_detect_task(JOB_ID, CONFIG_DATA)

        mock_detector.run.assert_called_once()

    def test_fans_out_process_segment_task_per_segment(self):
        """highlight_detect_task creates one process_segment_task per segment."""
        segments = [self._make_segment(f"seg-{i:03d}") for i in range(3)]

        mock_detector = MagicMock()
        mock_detector.run.return_value = segments

        mock_extractor = MagicMock()
        mock_extractor.extract_all.return_value = {
            seg.segment_id: f"jobs/x/clips/{seg.segment_id}/raw.mp4"
            for seg in segments
        }

        mock_storage = MagicMock()
        s_call_count = []

        def capture_group(gen):
            # Consume the generator so .s() calls are made
            items = list(gen)
            s_call_count.append(len(items))
            return MagicMock()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.HighlightDetector", return_value=mock_detector), \
             patch("app.workers.pipeline.ClipExtractor", return_value=mock_extractor), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.group", side_effect=capture_group), \
             patch("app.workers.pipeline.chord") as mock_chord, \
             patch("app.workers.pipeline.process_segment_task") as mock_pst:
            mock_chord.return_value = MagicMock()
            mock_pst.s = MagicMock(return_value=MagicMock())
            from app.workers.pipeline import highlight_detect_task
            highlight_detect_task(JOB_ID, CONFIG_DATA)

        # group was called with 3 items (one per segment)
        assert sum(s_call_count) == 3

    def test_updates_progress_to_20_percent(self):
        """highlight_detect_task updates Redis progress to 20% after detection."""
        seg = self._make_segment()
        mock_detector = MagicMock()
        mock_detector.run.return_value = [seg]

        mock_extractor = MagicMock()
        mock_extractor.extract_all.return_value = {"seg-001": "jobs/x/clips/seg-001/raw.mp4"}

        mock_storage = MagicMock()
        progress_calls = []

        def capture_progress(job_id, progress, stage):
            progress_calls.append((job_id, progress, stage))

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.HighlightDetector", return_value=mock_detector), \
             patch("app.workers.pipeline.ClipExtractor", return_value=mock_extractor), \
             patch("app.workers.pipeline.mark_job_progress", side_effect=capture_progress), \
             patch("app.workers.pipeline.group"), \
             patch("app.workers.pipeline.chord") as mock_chord, \
             patch("app.workers.pipeline.process_segment_task"):
            mock_chord.return_value = MagicMock()
            from app.workers.pipeline import highlight_detect_task
            highlight_detect_task(JOB_ID, CONFIG_DATA)

        assert any(p == 20.0 for _, p, _ in progress_calls)

    def test_finalizes_immediately_when_no_segments(self):
        """highlight_detect_task calls finalize_job_task directly when no segments found."""
        mock_detector = MagicMock()
        mock_detector.run.return_value = []

        mock_storage = MagicMock()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.HighlightDetector", return_value=mock_detector), \
             patch("app.workers.pipeline.mark_job_progress"), \
             patch("app.workers.pipeline.finalize_job_task") as mock_finalize:
            from app.workers.pipeline import highlight_detect_task
            highlight_detect_task(JOB_ID, CONFIG_DATA)

        mock_finalize.delay.assert_called_once_with(JOB_ID, [])


# ---------------------------------------------------------------------------
# Tests for process_segment_task (15.1)
# ---------------------------------------------------------------------------

class TestProcessSegmentTask:
    """process_segment_task runs all per-segment stages in order."""

    def _make_storage_with_transcript(self):
        mock_storage = MagicMock()
        mock_storage.download.return_value = json.dumps(TRANSCRIPT_DATA).encode("utf-8")
        return mock_storage

    def test_runs_all_per_segment_stages_in_order(self):
        """process_segment_task calls Format → Caption → Visual → Audio → Export in order."""
        call_order = []

        mock_storage = self._make_storage_with_transcript()

        mock_format = MagicMock()
        mock_format.optimize.side_effect = lambda *a, **kw: call_order.append("format")

        mock_caption = MagicMock()
        mock_caption.generate.side_effect = lambda *a, **kw: call_order.append("caption")

        mock_visual = MagicMock()
        mock_visual.enhance.side_effect = lambda *a, **kw: call_order.append("visual")

        mock_audio = MagicMock()
        mock_audio.optimize.side_effect = lambda *a, **kw: call_order.append("audio")

        mock_export = MagicMock()
        mock_export.export.side_effect = lambda *a, **kw: (
            call_order.append("export") or "https://signed.url/clip.mp4"
        )

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.FormatOptimizer", return_value=mock_format), \
             patch("app.workers.pipeline.CaptionGenerator", return_value=mock_caption), \
             patch("app.workers.pipeline.VisualEnhancer", return_value=mock_visual), \
             patch("app.workers.pipeline.AudioOptimizer", return_value=mock_audio), \
             patch("app.workers.pipeline.ExportEngine", return_value=mock_export), \
             patch("app.workers.pipeline.mark_job_progress"):
            from app.workers.pipeline import process_segment_task
            result = process_segment_task(JOB_ID, SEGMENT_DATA, CONFIG_DATA, 0, 1)

        assert call_order == ["format", "caption", "visual", "audio", "export"]

    def test_returns_signed_url_on_success(self):
        """process_segment_task returns dict with segment_id and signed_url."""
        mock_storage = self._make_storage_with_transcript()
        signed_url = "https://storage.example.com/clip.mp4?sig=abc"

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.FormatOptimizer") as FO, \
             patch("app.workers.pipeline.CaptionGenerator") as CG, \
             patch("app.workers.pipeline.VisualEnhancer") as VE, \
             patch("app.workers.pipeline.AudioOptimizer") as AO, \
             patch("app.workers.pipeline.ExportEngine") as EE, \
             patch("app.workers.pipeline.mark_job_progress"):
            EE.return_value.export.return_value = signed_url
            from app.workers.pipeline import process_segment_task
            result = process_segment_task(JOB_ID, SEGMENT_DATA, CONFIG_DATA, 0, 1)

        assert result["segment_id"] == "seg-001"
        assert result["signed_url"] == signed_url

    def test_returns_error_on_stage_failure(self):
        """process_segment_task returns error dict when a stage raises an exception."""
        mock_storage = self._make_storage_with_transcript()

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.FormatOptimizer") as FO, \
             patch("app.workers.pipeline.mark_job_progress"):
            FO.return_value.optimize.side_effect = RuntimeError("FFmpeg failed")
            from app.workers.pipeline import process_segment_task
            result = process_segment_task(JOB_ID, SEGMENT_DATA, CONFIG_DATA, 0, 1)

        assert result["segment_id"] == "seg-001"
        assert "error" in result

    def test_updates_progress_after_segment_completes(self):
        """process_segment_task updates Redis progress after each segment."""
        mock_storage = self._make_storage_with_transcript()
        progress_calls = []

        def capture_progress(job_id, progress, stage):
            progress_calls.append((job_id, progress, stage))

        with patch("app.workers.pipeline._make_storage", return_value=mock_storage), \
             patch("app.workers.pipeline.FormatOptimizer"), \
             patch("app.workers.pipeline.CaptionGenerator"), \
             patch("app.workers.pipeline.VisualEnhancer"), \
             patch("app.workers.pipeline.AudioOptimizer"), \
             patch("app.workers.pipeline.ExportEngine") as EE, \
             patch("app.workers.pipeline.mark_job_progress", side_effect=capture_progress):
            EE.return_value.export.return_value = "https://signed.url/clip.mp4"
            from app.workers.pipeline import process_segment_task
            process_segment_task(JOB_ID, SEGMENT_DATA, CONFIG_DATA, 0, 2)

        # Progress should be updated (30% + 35% = 65% for first of 2 segments)
        assert len(progress_calls) > 0
        last_progress = progress_calls[-1][1]
        assert last_progress > 30.0  # beyond clip extraction stage


# ---------------------------------------------------------------------------
# Tests for finalize_job_task (15.1)
# ---------------------------------------------------------------------------

class TestFinalizeJobTask:
    """finalize_job_task marks job as completed with output_urls."""

    def test_marks_job_completed(self):
        """finalize_job_task sets job status to 'completed' in Redis."""
        initial = {
            "job_id": JOB_ID,
            "status": "processing",
            "progress": 90.0,
            "current_stage": "per_segment_processing",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        results = [
            {"segment_id": "seg-001", "signed_url": "https://cdn.example.com/clip1.mp4"},
            {"segment_id": "seg-002", "signed_url": "https://cdn.example.com/clip2.mp4"},
        ]

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import finalize_job_task
            result = finalize_job_task(results, JOB_ID)

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["status"] == "completed"

    def test_stores_output_urls(self):
        """finalize_job_task stores all signed URLs as output_urls."""
        initial = {
            "job_id": JOB_ID,
            "status": "processing",
            "progress": 90.0,
            "current_stage": "per_segment_processing",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        urls = [
            "https://cdn.example.com/clip1.mp4",
            "https://cdn.example.com/clip2.mp4",
        ]
        results = [
            {"segment_id": "seg-001", "signed_url": urls[0]},
            {"segment_id": "seg-002", "signed_url": urls[1]},
        ]

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import finalize_job_task
            result = finalize_job_task(results, JOB_ID)

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["output_urls"] == urls
        assert result["output_urls"] == urls

    def test_handles_empty_results(self):
        """finalize_job_task handles empty results list (no segments)."""
        initial = {
            "job_id": JOB_ID,
            "status": "processing",
            "progress": 20.0,
            "current_stage": "highlight_detection",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import finalize_job_task
            result = finalize_job_task([], JOB_ID)

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["status"] == "completed"
        assert stored["output_urls"] == []

    def test_skips_failed_segments_in_output_urls(self):
        """finalize_job_task excludes segments that returned errors from output_urls."""
        initial = {
            "job_id": JOB_ID,
            "status": "processing",
            "progress": 90.0,
            "current_stage": "per_segment_processing",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        results = [
            {"segment_id": "seg-001", "signed_url": "https://cdn.example.com/clip1.mp4"},
            {"segment_id": "seg-002", "error": "FFmpeg failed"},
        ]

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import finalize_job_task
            result = finalize_job_task(results, JOB_ID)

        assert len(result["output_urls"]) == 1
        assert result["output_urls"][0] == "https://cdn.example.com/clip1.mp4"

    def test_sets_progress_to_100(self):
        """finalize_job_task sets progress to 100% on completion."""
        initial = {
            "job_id": JOB_ID,
            "status": "processing",
            "progress": 90.0,
            "current_stage": "per_segment_processing",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.workers.pipeline import finalize_job_task
            finalize_job_task([], JOB_ID)

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["progress"] == 100.0


# ---------------------------------------------------------------------------
# Tests for mark_job_progress (15.2)
# ---------------------------------------------------------------------------

class TestMarkJobProgress:
    """Progress is updated in Redis at each stage."""

    def test_updates_progress_and_stage(self):
        """mark_job_progress writes progress and current_stage to Redis."""
        initial = {
            "job_id": JOB_ID,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "queued",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.api.main import mark_job_progress
            mark_job_progress(JOB_ID, 10.0, "transcription")

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["progress"] == 10.0
        assert stored["current_stage"] == "transcription"

    def test_sets_status_to_processing(self):
        """mark_job_progress transitions status from queued to processing."""
        initial = {
            "job_id": JOB_ID,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "queued",
            "output_urls": None,
            "error": None,
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.api.main import mark_job_progress
            mark_job_progress(JOB_ID, 10.0, "transcription")

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["status"] == "processing"

    def test_does_not_overwrite_failed_status(self):
        """mark_job_progress does not change status when job is already failed."""
        initial = {
            "job_id": JOB_ID,
            "status": "failed",
            "progress": 10.0,
            "current_stage": "transcription",
            "output_urls": None,
            "error": "Something went wrong",
        }
        redis_mock = _make_redis_store({f"job:{JOB_ID}": initial})

        with patch("app.api.main._get_redis", return_value=redis_mock):
            from app.api.main import mark_job_progress
            mark_job_progress(JOB_ID, 20.0, "highlight_detection")

        stored = json.loads(redis_mock._store[f"job:{JOB_ID}"])
        assert stored["status"] == "failed"

    def test_tolerates_redis_error(self):
        """mark_job_progress does not raise when Redis is unavailable."""
        with patch("app.api.main._get_redis", side_effect=Exception("Redis down")):
            from app.api.main import mark_job_progress
            # Should not raise
            mark_job_progress(JOB_ID, 10.0, "transcription")
