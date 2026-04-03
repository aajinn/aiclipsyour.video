"""Unit tests for the FastAPI Orchestrator API (Tasks 13.1, 13.3, 13.5, 13.7, 13.8)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, mark_job_failed

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JOB_PAYLOAD = {
    "video_url": "https://example.com/video.mp4",
    "config": {
        "transcription_provider": "whisper",
        "face_tracking": False,
        "noise_reduction": False,
        "caption_style": "default",
    },
}


def _make_redis_mock(job_state: dict | None = None) -> MagicMock:
    """Return a mock Redis client that stores/retrieves a single job state."""
    store: dict[str, bytes] = {}

    mock = MagicMock()
    mock.ping.return_value = True

    def _set(key: str, value: str) -> None:
        store[key] = value.encode() if isinstance(value, str) else value

    def _get(key: str) -> bytes | None:
        if job_state is not None and key.startswith("job:"):
            return json.dumps(job_state).encode()
        return store.get(key)

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    return mock


# ---------------------------------------------------------------------------
# POST /jobs — happy path (task 13.1)
# ---------------------------------------------------------------------------

class TestSubmitJob:
    def test_valid_config_returns_200_with_job_id(self):
        redis_mock = _make_redis_mock()
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch("app.api.main.celery_app") as celery_mock:
            celery_mock.send_task.return_value = MagicMock()
            response = client.post("/jobs", json=VALID_JOB_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_valid_config_enqueues_transcriber_task(self):
        redis_mock = _make_redis_mock()
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch("app.api.main.celery_app") as celery_mock:
            celery_mock.send_task.return_value = MagicMock()
            client.post("/jobs", json=VALID_JOB_PAYLOAD)

        celery_mock.send_task.assert_called_once()
        task_name = celery_mock.send_task.call_args[0][0]
        assert "transcrib" in task_name.lower()

    def test_valid_config_stores_initial_state_in_redis(self):
        redis_mock = _make_redis_mock()
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch("app.api.main.celery_app") as celery_mock:
            celery_mock.send_task.return_value = MagicMock()
            response = client.post("/jobs", json=VALID_JOB_PAYLOAD)

        job_id = response.json()["job_id"]
        # Verify Redis.set was called with the job key
        redis_mock.set.assert_called_once()
        key_arg = redis_mock.set.call_args[0][0]
        assert key_arg == f"job:{job_id}"

        # Verify stored state has expected fields
        stored_json = redis_mock.set.call_args[0][1]
        stored = json.loads(stored_json)
        assert stored["status"] == "queued"
        assert stored["progress"] == 0.0
        assert stored["current_stage"] == "queued"

    def test_each_submission_returns_unique_job_id(self):
        redis_mock = _make_redis_mock()
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch("app.api.main.celery_app") as celery_mock:
            celery_mock.send_task.return_value = MagicMock()
            r1 = client.post("/jobs", json=VALID_JOB_PAYLOAD)
            r2 = client.post("/jobs", json=VALID_JOB_PAYLOAD)

        assert r1.json()["job_id"] != r2.json()["job_id"]


# ---------------------------------------------------------------------------
# POST /jobs — invalid config returns 422 (task 13.1 / requirement 9.2)
# ---------------------------------------------------------------------------

class TestSubmitJobValidation:
    def test_invalid_transcription_provider_returns_422(self):
        payload = {
            "video_url": "https://example.com/video.mp4",
            "config": {"transcription_provider": "invalid_provider"},
        }
        response = client.post("/jobs", json=payload)
        assert response.status_code == 422

    def test_missing_video_url_returns_422(self):
        payload = {"config": {"transcription_provider": "whisper"}}
        response = client.post("/jobs", json=payload)
        assert response.status_code == 422

    def test_invalid_caption_style_returns_422(self):
        payload = {
            "video_url": "https://example.com/video.mp4",
            "config": {"caption_style": "neon"},
        }
        response = client.post("/jobs", json=payload)
        assert response.status_code == 422

    def test_422_response_contains_field_level_errors(self):
        payload = {
            "video_url": "https://example.com/video.mp4",
            "config": {"transcription_provider": "bad"},
        }
        response = client.post("/jobs", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# POST /jobs — Redis unavailable returns 503 (task 13.8)
# ---------------------------------------------------------------------------

class TestSubmitJobRedisUnavailable:
    def test_redis_unavailable_returns_503(self):
        with patch("app.api.main._get_redis", side_effect=Exception("Connection refused")):
            response = client.post("/jobs", json=VALID_JOB_PAYLOAD)

        assert response.status_code == 503

    def test_503_includes_retry_after_header(self):
        with patch("app.api.main._get_redis", side_effect=Exception("Connection refused")):
            response = client.post("/jobs", json=VALID_JOB_PAYLOAD)

        assert response.status_code == 503
        assert "retry-after" in {k.lower() for k in response.headers}

    def test_redis_unavailable_does_not_enqueue_task(self):
        with patch("app.api.main._get_redis", side_effect=Exception("Connection refused")), \
             patch("app.api.main.celery_app") as celery_mock:
            client.post("/jobs", json=VALID_JOB_PAYLOAD)

        celery_mock.send_task.assert_not_called()


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} (task 13.3)
# ---------------------------------------------------------------------------

class TestGetJobStatus:
    def _queued_state(self, job_id: str) -> dict:
        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "queued",
            "output_urls": None,
            "error": None,
        }

    def test_returns_job_status_shape(self):
        job_id = "test-job-123"
        state = self._queued_state(job_id)
        redis_mock = _make_redis_mock(job_state=state)

        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in {"queued", "processing", "completed", "failed"}
        assert 0.0 <= data["progress"] <= 100.0
        assert "current_stage" in data

    def test_returns_correct_status_values(self):
        job_id = "test-job-456"
        state = self._queued_state(job_id)
        redis_mock = _make_redis_mock(job_state=state)

        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get(f"/jobs/{job_id}")

        data = response.json()
        assert data["status"] == "queued"
        assert data["progress"] == 0.0
        assert data["current_stage"] == "queued"

    def test_completed_job_includes_output_urls(self):
        job_id = "test-job-done"
        state = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "current_stage": "export",
            "output_urls": ["https://storage.example.com/clip1.mp4"],
            "error": None,
        }
        redis_mock = _make_redis_mock(job_state=state)

        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get(f"/jobs/{job_id}")

        data = response.json()
        assert data["status"] == "completed"
        assert isinstance(data["output_urls"], list)
        assert len(data["output_urls"]) > 0

    def test_failed_job_includes_error(self):
        job_id = "test-job-fail"
        state = {
            "job_id": job_id,
            "status": "failed",
            "progress": 10.0,
            "current_stage": "transcriber",
            "output_urls": None,
            "error": "Transcription provider timed out",
        }
        redis_mock = _make_redis_mock(job_state=state)

        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get(f"/jobs/{job_id}")

        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Transcription provider timed out"

    def test_unknown_job_id_returns_404(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None  # job not found

        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get("/jobs/nonexistent-job-id")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Worker failure handler (task 13.5)
# ---------------------------------------------------------------------------

class TestMarkJobFailed:
    def test_sets_status_to_failed(self):
        job_id = "fail-job-1"
        initial_state = {
            "job_id": job_id,
            "status": "processing",
            "progress": 20.0,
            "current_stage": "transcriber",
            "output_urls": None,
            "error": None,
        }
        store: dict[str, str] = {f"job:{job_id}": json.dumps(initial_state)}

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = lambda k: store.get(k, "").encode() if store.get(k) else None
        mock_client.set.side_effect = lambda k, v: store.update({k: v})

        with patch("app.api.main._get_redis", return_value=mock_client):
            mark_job_failed(job_id, "FFmpeg crashed with exit code 1")

        stored = json.loads(store[f"job:{job_id}"])
        assert stored["status"] == "failed"

    def test_records_error_message(self):
        job_id = "fail-job-2"
        initial_state = {
            "job_id": job_id,
            "status": "processing",
            "progress": 50.0,
            "current_stage": "clip_extractor",
            "output_urls": None,
            "error": None,
        }
        store: dict[str, str] = {f"job:{job_id}": json.dumps(initial_state)}

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = lambda k: store.get(k, "").encode() if store.get(k) else None
        mock_client.set.side_effect = lambda k, v: store.update({k: v})

        error_msg = "Segment extraction failed: codec not supported"
        with patch("app.api.main._get_redis", return_value=mock_client):
            mark_job_failed(job_id, error_msg)

        stored = json.loads(store[f"job:{job_id}"])
        assert stored["error"] == error_msg

    def test_failure_handler_tolerates_redis_error(self):
        """mark_job_failed should not raise even if Redis is down."""
        with patch("app.api.main._get_redis", side_effect=Exception("Redis down")):
            # Should not raise
            mark_job_failed("some-job", "some error")

    def test_error_message_is_non_empty(self):
        job_id = "fail-job-3"
        initial_state = {
            "job_id": job_id,
            "status": "processing",
            "progress": 30.0,
            "current_stage": "format_optimizer",
            "output_urls": None,
            "error": None,
        }
        store: dict[str, str] = {f"job:{job_id}": json.dumps(initial_state)}

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = lambda k: store.get(k, "").encode() if store.get(k) else None
        mock_client.set.side_effect = lambda k, v: store.update({k: v})

        with patch("app.api.main._get_redis", return_value=mock_client):
            mark_job_failed(job_id, "Non-empty error detail")

        stored = json.loads(store[f"job:{job_id}"])
        assert stored["error"]  # non-empty


# ---------------------------------------------------------------------------
# GET /health (task 13.7)
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_200(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch.dict("os.environ", {"AWS_S3_BUCKET": "my-bucket"}):
            response = client.get("/health")

        assert response.status_code == 200

    def test_response_contains_checks_dict(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch.dict("os.environ", {"AWS_S3_BUCKET": "my-bucket"}):
            response = client.get("/health")

        data = response.json()
        assert "status" in data
        assert "checks" in data
        checks = data["checks"]
        assert "api" in checks
        assert "queue" in checks
        assert "storage" in checks

    def test_api_check_always_ok(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get("/health")

        assert response.json()["checks"]["api"] == "ok"

    def test_queue_ok_when_redis_reachable(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock):
            response = client.get("/health")

        assert response.json()["checks"]["queue"] == "ok"

    def test_queue_error_when_redis_unreachable(self):
        with patch("app.api.main._get_redis", side_effect=Exception("Connection refused")):
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["checks"]["queue"] == "error"

    def test_storage_ok_when_env_var_set(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch.dict("os.environ", {"AWS_S3_BUCKET": "my-bucket"}, clear=False):
            response = client.get("/health")

        assert response.json()["checks"]["storage"] == "ok"

    def test_storage_error_when_no_env_vars(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        env_without_storage = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("AWS_S3_BUCKET", "GCS_BUCKET", "STORAGE_BUCKET")
        }
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch.dict("os.environ", env_without_storage, clear=True):
            response = client.get("/health")

        assert response.json()["checks"]["storage"] == "error"

    def test_degraded_status_when_queue_down(self):
        with patch("app.api.main._get_redis", side_effect=Exception("down")):
            response = client.get("/health")

        assert response.json()["status"] == "degraded"

    def test_ok_status_when_all_healthy(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        with patch("app.api.main._get_redis", return_value=redis_mock), \
             patch.dict("os.environ", {"AWS_S3_BUCKET": "bucket"}, clear=False):
            response = client.get("/health")

        assert response.json()["status"] == "ok"
