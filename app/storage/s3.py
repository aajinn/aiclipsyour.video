from __future__ import annotations

import os

import boto3
from botocore.exceptions import ClientError

from app.storage.base import CloudStorage


class S3Storage(CloudStorage):
    """AWS S3 implementation of CloudStorage."""

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def upload_file(self, key: str, local_path: str, content_type: str = "application/octet-stream") -> None:
        self._client.upload_file(
            Filename=local_path,
            Bucket=self.bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )

    def download(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def download_file(self, key: str, local_path: str) -> None:
        self._client.download_file(Bucket=self.bucket, Key=key, Filename=local_path)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def generate_signed_url(self, key: str, expiration_seconds: int = 86400) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiration_seconds,
        )
