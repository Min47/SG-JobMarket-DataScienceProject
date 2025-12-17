"""GCS helpers for uploading and downloading files from Google Cloud Storage.

This module provides a client for interacting with GCS, including:
- Upload files with compression
- Download files
- List blobs
- Check blob existence
- Path helper utilities
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from google.api_core import retry

from utils.retry import RetryPolicy, retry_call


logger = logging.getLogger(__name__)


class GCSError(Exception):
    """Base exception for GCS operations."""


class GCSPermissionError(GCSError):
    """Raised when IAM permissions are insufficient."""


class GCSNotFoundError(GCSError):
    """Raised when bucket or blob is not found."""


class GCSClient:
    """Client for Google Cloud Storage operations with retry logic."""

    def __init__(self, project_id: Optional[str] = None) -> None:
        """Initialize GCS client.
        
        Args:
            project_id: GCP project ID (optional, uses default from environment)
        """
        try:
            self._client = storage.Client(project=project_id)
            logger.info(f"[GCS] Initialized client for project: {project_id or 'default'}")
        except Exception as e:
            logger.error(f"[GCS] Failed to initialize client: {e}")
            raise GCSError(f"Failed to initialize GCS client: {e}") from e

    def upload_file(
        self,
        local_path: str | Path,
        gcs_uri: str,
        compress: bool = False,
    ) -> dict[str, str]:
        """Upload a file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_uri: Destination GCS URI (gs://bucket/path/to/file)
            compress: If True, compress with gzip before upload
            
        Returns:
            Dict with blob metadata (size, created, gs_uri)
            
        Raises:
            GCSPermissionError: If IAM permissions are insufficient
            GCSError: For other upload failures
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Prepare upload (with optional compression)
            if compress:
                # Compress to temporary file
                compressed_path = local_path.with_suffix(local_path.suffix + ".gz")
                with open(local_path, "rb") as f_in:
                    with gzip.open(compressed_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                upload_path = compressed_path
                # Update blob name to include .gz
                blob = bucket.blob(blob_name + ".gz")
                logger.info(f"[GCS] Compressed {local_path.stat().st_size} bytes to {upload_path.stat().st_size} bytes")
            else:
                upload_path = local_path
            
            # Upload with retry
            def _upload() -> None:
                blob.upload_from_filename(
                    str(upload_path),
                    retry=retry.Retry(deadline=300.0),  # 5 min timeout
                )
            
            retry_call(
                _upload,
                policy=RetryPolicy(max_attempts=3, base_delay_seconds=2.0),
                retry_on=(Exception,),
                on_retry=lambda attempt, exc: logger.warning(f"[GCS] Upload retry {attempt}: {exc}"),
            )
            
            # Clean up compressed temp file
            if compress and compressed_path.exists():
                compressed_path.unlink()
            
            # Get metadata
            blob.reload()
            
            result = {
                "size": str(blob.size),
                "created": blob.time_created.isoformat() if blob.time_created else "unknown",
                "gs_uri": f"gs://{bucket_name}/{blob.name}",
            }
            
            logger.info(f"[GCS] Uploaded {local_path.name} to {result['gs_uri']} ({result['size']} bytes)")
            return result
            
        except Forbidden as e:
            raise GCSPermissionError(
                f"Permission denied uploading to gs://{bucket_name}/{blob_name}. "
                "Check IAM role: roles/storage.objectAdmin"
            ) from e
        except Exception as e:
            logger.error(f"[GCS] Upload failed: {e}")
            raise GCSError(f"Failed to upload {local_path} to {gcs_uri}: {e}") from e

    def upload_jsonl(
        self,
        local_path: str | Path,
        gcs_uri: str,
        compress: bool = True,
    ) -> dict[str, str]:
        """Upload a JSONL file to GCS with optional compression.
        
        This is a convenience wrapper around upload_file() with compression enabled by default.
        
        Args:
            local_path: Path to local JSONL file
            gcs_uri: Destination GCS URI (gs://bucket/path/to/file.jsonl)
            compress: If True (default), compress with gzip
            
        Returns:
            Dict with blob metadata
        """
        return self.upload_file(local_path, gcs_uri, compress=compress)

    def download_file(
        self,
        gcs_uri: str,
        local_path: str | Path,
    ) -> Path:
        """Download a file from GCS to local filesystem.
        
        Args:
            gcs_uri: Source GCS URI (gs://bucket/path/to/file)
            local_path: Destination local path
            
        Returns:
            Path to downloaded file
            
        Raises:
            GCSNotFoundError: If blob doesn't exist
            GCSPermissionError: If IAM permissions are insufficient
        """
        local_path = Path(local_path)
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        
        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Download with retry
            def _download() -> None:
                blob.download_to_filename(
                    str(local_path),
                    retry=retry.Retry(deadline=300.0),
                )
            
            retry_call(
                _download,
                policy=RetryPolicy(max_attempts=3, base_delay_seconds=2.0),
                retry_on=(Exception,),
                on_retry=lambda attempt, exc: logger.warning(f"[GCS] Download retry {attempt}: {exc}"),
            )
            
            logger.info(f"[GCS] Downloaded {gcs_uri} to {local_path} ({local_path.stat().st_size} bytes)")
            return local_path
            
        except NotFound as e:
            raise GCSNotFoundError(f"Blob not found: {gcs_uri}") from e
        except Forbidden as e:
            raise GCSPermissionError(
                f"Permission denied downloading from {gcs_uri}. "
                "Check IAM role: roles/storage.objectViewer"
            ) from e
        except Exception as e:
            logger.error(f"[GCS] Download failed: {e}")
            raise GCSError(f"Failed to download {gcs_uri} to {local_path}: {e}") from e

    def list_blobs(
        self,
        bucket_name: str,
        prefix: str = "",
        extension: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """List blobs in a bucket with optional prefix and extension filter.
        
        Args:
            bucket_name: Name of the GCS bucket
            prefix: Filter by prefix (e.g., "raw/jobstreet/")
            extension: Filter by file extension (e.g., ".jsonl")
            
        Returns:
            List of dicts with blob metadata (name, size, updated)
        """
        try:
            bucket = self._client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            results = []
            for blob in blobs:
                # Apply extension filter if specified
                if extension and not blob.name.endswith(extension):
                    continue
                
                results.append({
                    "name": blob.name,
                    "size": str(blob.size),
                    "updated": blob.updated.isoformat() if blob.updated else "unknown",
                })
            
            logger.info(f"[GCS] Listed {len(results)} blobs in gs://{bucket_name}/{prefix}")
            return results
            
        except Exception as e:
            logger.error(f"[GCS] List blobs failed: {e}")
            raise GCSError(f"Failed to list blobs in gs://{bucket_name}/{prefix}: {e}") from e

    def exists(self, gcs_uri: str) -> bool:
        """Check if a blob exists in GCS.
        
        Args:
            gcs_uri: GCS URI to check (gs://bucket/path/to/file)
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            bucket_name, blob_name = parse_gcs_uri(gcs_uri)
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.warning(f"[GCS] Error checking existence of {gcs_uri}: {e}")
            return False


# =============================================================================
# Path Helper Functions
# =============================================================================

def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a GCS URI into bucket name and blob name.
    
    Args:
        gcs_uri: GCS URI in format gs://bucket/path/to/blob
        
    Returns:
        Tuple of (bucket_name, blob_name)
        
    Raises:
        ValueError: If URI format is invalid
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (must start with gs://): {gcs_uri}")
    
    parts = gcs_uri[5:].split("/", 1)
    
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid GCS URI format (expected gs://bucket/path): {gcs_uri}")
    
    return parts[0], parts[1]


def validate_gcs_uri(gcs_uri: str) -> bool:
    """Validate GCS URI format.
    
    Args:
        gcs_uri: URI to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_gcs_uri(gcs_uri)
        return True
    except ValueError:
        return False


def build_raw_path(bucket: str, source: str, timestamp: str, filename: str = "dump.jsonl") -> str:
    """Build GCS path for raw scraper output.
    
    Args:
        bucket: GCS bucket name
        source: Source name (jobstreet, mcf)
        timestamp: Run timestamp (YYYY-MM-DD_HHMMSS)
        filename: Output filename (default: dump.jsonl)
        
    Returns:
        Complete GCS URI
    """
    return f"gs://{bucket}/raw/{source}/{timestamp}/{filename}"


def build_model_path(bucket: str, model_name: str, version: str) -> str:
    """Build GCS path for ML model storage.
    
    Args:
        bucket: GCS bucket name
        model_name: Name of the model
        version: Model version string
        
    Returns:
        Complete GCS URI (directory path)
    """
    return f"gs://{bucket}/models/{model_name}/{version}/"

