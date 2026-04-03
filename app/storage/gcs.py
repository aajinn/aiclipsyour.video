from __future__ import annotations

import datetime
import os

from google.cloud import storage as gcs_lib

from app.storage.base import CloudStorage


class GCSStorage(CloudStorage):
    """Google Cloud Storage implementation of CloudStorage."""

    def __init__(
        self,
        bucket: str,
        project: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        self.bucket_name = bucket
        if credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            self._client = gcs_lib.Client(project=project)
        else:
            self._client = gcs_lib.Client(project=project)
        self._bucket = self._client.bucket(bucket)

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(data, content_type=content_type)

    def upload_file(self, key: str, local_path: str, content_type: str = "application/octet-stream") -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_filename(local_path, content_type=content_type)

    def download(self, key: str) -> bytes:
        blob = self._bucket.blob(key)
        return blob.download_as_bytes()

    def download_file(self, key: str, local_path: str) -> None:
        blob = self._bucket.blob(key)
        blob.download_to_filename(local_path)

    def exists(self, key: str) -> bool:
        blob = self._bucket.blob(key)
        return blob.exists()

    def delete(self, key: str) -> None:
        blob = self._bucket.blob(key)
        blob.delete()

    def generate_signed_url(self, key: str, expiration_seconds: int = 86400) -> str:
        blob = self._bucket.blob(key)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=expiration_seconds),
            method="GET",
            version="v4",
        )
