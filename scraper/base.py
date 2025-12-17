"""Base classes for site scrapers.

This module defines the core interface for async scrapers that can run in Cloud Run.
Concrete site scrapers should implement `fetch()` and `parse()`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

import aiohttp

from dotenv import load_dotenv

from utils.retry import RetryPolicy, retry_async_call
from utils.schemas import RawJob
from utils.config import Settings

# Load environment variables (critical for Docker)
load_dotenv()

# Configuration
MAX_RETAINED_RUNS = 10  # Keep only N most recent run folders per source
LOCAL_RETENTION_DAYS = int(os.getenv("LOCAL_RETENTION_DAYS", "30"))  # Keep local files for N days
GCS_UPLOAD_ENABLED = os.getenv("GCS_UPLOAD_ENABLED", "false").lower() == "true"

# User agents loaded from environment variable
# Format: comma-separated list in .env as SCRAPER_USER_AGENTS="agent1,agent2,agent3"
def _load_user_agents() -> list[str]:
    """Load user agents from environment variable."""
    agents_str = os.getenv("SCRAPER_USER_AGENTS", "")
    if agents_str:
        return [agent.strip() for agent in agents_str.split(",") if agent.strip()]
    # Fallback default
    return ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"]

USER_AGENTS = _load_user_agents()


@dataclass(frozen=True, slots=True)
class ScrapeContext:
    """Run context for a scraper execution."""

    run_timestamp: str  # YYYY-MM-DD_HHMMSS for unique filenames
    source: str
    output_dir: str = "data/raw"


class BaseScraper(ABC):
    """Async scraper base class.

    Contract:
    - Output must be deterministic given the same inputs.
    - Must not keep state between runs (Cloud Run is stateless).
    - Must yield `RawJob` records (schema-contract).
    """

    def __init__(self, *, context: ScrapeContext) -> None:
        self._context = context
        self._logger = logging.getLogger(f"scraper.{context.source}")
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def context(self) -> ScrapeContext:
        return self._context

    async def __aenter__(self) -> BaseScraper:
        """Initialize resources (aiohttp session)."""
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup resources."""
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the active session, raising error if not initialized."""
        if self._session is None:
            raise RuntimeError("Scraper not initialized. Use 'async with scraper: ...'")
        return self._session

    async def _fetch_url(self, url: str, method: str = "GET", **kwargs: Any) -> str:
        """Fetch a URL with retries and rotation."""
        
        async def _request() -> str:
            # Random sleep to avoid rate limits
            await asyncio.sleep(random.uniform(1.0, 2.5))
            
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.text()

        return await retry_async_call(
            _request,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=2.0),
            retry_on=(aiohttp.ClientError, asyncio.TimeoutError),
            on_retry=lambda attempt, exc: self._logger.warning(
                f"[ScraperBase] Retry {attempt} for {url}: {exc}"
            ),
        )

    @abstractmethod
    async def fetch(self) -> AsyncIterator[str]:
        """Yield raw HTML/JSON payloads from the source."""

    @abstractmethod
    def parse(self, payload: str) -> Iterable[RawJob]:
        """Parse a payload into one or more `RawJob` records."""

    async def run(self) -> None:
        """Execute the full scrape pipeline and write to disk."""
        output_path = self._get_output_path()
        self._logger.info(f"[ScraperBase] Starting scrape for {self.context.source} -> {output_path}")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            async for payload in self.fetch():
                for job in self.parse(payload):
                    # Validate it's a RawJob (runtime check)
                    if not isinstance(job, RawJob):
                        self._logger.error(f"[ScraperBase] Parser returned invalid type: {type(job)}")
                        continue
                    
                    # Write as JSONL
                    f.write(json.dumps(asdict(job), ensure_ascii=False) + "\n")
                    count += 1
                    
                    # if count % 10 == 0:
                    #     self._logger.info(f"[ScraperBase] Scraped {count} jobs...")

        self._logger.info(f"[ScraperBase] Finished. Total jobs: {count}. Saved to {output_path}")
        
        # Quick validation check
        if count == 0:
            self._logger.warning("[ScraperBase] No jobs were scraped")
        else:
            # Upload to GCS if enabled
            self._upload_to_gcs_if_enabled(output_path)
        
        # Cleanup old runs (keep only MAX_RETAINED_RUNS most recent)
        self._cleanup_old_runs()

    def _get_output_path(self) -> Path:
        """Construct the output file path: data/raw/{source}/{timestamp}/dump.jsonl"""
        return (
            Path(self.context.output_dir)
            / self.context.source
            / self.context.run_timestamp
            / "dump.jsonl"
        )

    def _upload_to_gcs_if_enabled(self, local_path: Path) -> None:
        """Upload scraped data to GCS if enabled via environment variable.
        
        Args:
            local_path: Path to the local JSONL file to upload
        """
        if not GCS_UPLOAD_ENABLED:
            self._logger.debug("[ScraperBase] GCS upload disabled (GCS_UPLOAD_ENABLED=false)")
            return
        
        try:
            # Load settings and validate bucket
            settings = Settings.load()
            
            if not settings.gcs_bucket:
                self._logger.warning("[ScraperBase] GCS upload enabled but GCS_BUCKET not set in .env")
                return
            
            # Import GCS client (lazy import to avoid dependency issues)
            from utils.gcs import GCSClient, build_raw_path
            
            # Build GCS path
            gcs_uri = build_raw_path(
                bucket=settings.gcs_bucket,
                source=self.context.source,
                timestamp=self.context.run_timestamp,
                filename=local_path.name,
            )
            
            # Upload with compression
            self._logger.info(f"[ScraperBase] Uploading to GCS: {gcs_uri}")
            gcs_client = GCSClient(project_id=settings.gcp_project_id)
            metadata = gcs_client.upload_jsonl(
                local_path=local_path,
                gcs_uri=gcs_uri,
                compress=True,  # Save bandwidth and storage
            )
            
            self._logger.info(
                f"[ScraperBase] Successfully uploaded to {metadata['gs_uri']} "
                f"({metadata['size']} bytes, created: {metadata['created']})"
            )
            
        except Exception as e:
            # Log error but don't fail the scrape (local backup still exists)
            self._logger.error(f"[ScraperBase] Failed to upload to GCS: {e}", exc_info=True)
            self._logger.warning("[ScraperBase] Continuing with local file as backup")
    
    def _cleanup_old_runs(self) -> None:
        """Remove old run folders based on retention policy.
        
        Keeps runs from last LOCAL_RETENTION_DAYS OR last MAX_RETAINED_RUNS (whichever is more).
        This provides a safety net while optimizing storage costs.
        """
        source_dir = Path(self.context.output_dir) / self.context.source
        
        if not source_dir.exists():
            return
        
        # Get all timestamp folders (YYYY-MM-DD_HHMMSS format)
        run_folders = [
            d for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        if len(run_folders) <= MAX_RETAINED_RUNS:
            return
        
        # Sort by folder name (timestamp) descending (newest first)
        run_folders.sort(key=lambda x: x.name, reverse=True)
        
        # Calculate date threshold for retention
        from datetime import datetime, timedelta
        date_threshold = datetime.now() - timedelta(days=LOCAL_RETENTION_DAYS)
        
        folders_to_delete = []
        for i, folder in enumerate(run_folders):
            # Always keep most recent MAX_RETAINED_RUNS
            if i < MAX_RETAINED_RUNS:
                continue
            
            # Parse timestamp from folder name (YYYY-MM-DD_HHMMSS)
            try:
                folder_date = datetime.strptime(folder.name, "%Y-%m-%d_%H%M%S")
                if folder_date < date_threshold:
                    folders_to_delete.append(folder)
            except ValueError:
                # Invalid timestamp format, delete to clean up
                folders_to_delete.append(folder)
        
        # Delete old folders
        total_freed = 0
        for folder in folders_to_delete:
            try:
                import shutil
                folder_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                shutil.rmtree(folder)
                total_freed += folder_size
                self._logger.info(
                    f"[ScraperBase] Cleaned up old run: {folder.name} "
                    f"({folder_size / 1024 / 1024:.2f} MB)"
                )
            except Exception as e:
                self._logger.warning(f"[ScraperBase] Failed to delete old run {folder.name}: {e}")
        
        if total_freed > 0:
            self._logger.info(
                f"[ScraperBase] Total space freed: {total_freed / 1024 / 1024:.2f} MB "
                f"({len(folders_to_delete)} runs deleted, keeping {LOCAL_RETENTION_DAYS} days)"
            )
    
    # ========================================================================
    # Checkpoint Management (Shared across all scrapers)
    # ========================================================================
    
    def _get_checkpoint_path(self, filename: str = "checkpoint.txt") -> Path:
        """Get path for checkpoint file.
        
        Args:
            filename: Name of checkpoint file (default: checkpoint.txt)
                     JobStreet uses: job_ids.txt
                     MCF uses: job_uuids.txt
        """
        return (
            Path(self.context.output_dir)
            / self.context.source
            / self.context.run_timestamp
            / filename
        )
    
    def _load_checkpoint(self, filename: str = "checkpoint.txt") -> list[str]:
        """Load checkpoint data from file.
        
        Looks in current timestamp folder first, then falls back to most recent previous run.
        
        Args:
            filename: Name of checkpoint file to load
            
        Returns:
            List of lines from checkpoint file (stripped, non-empty)
        """
        checkpoint_path = self._get_checkpoint_path(filename)
        
        # Try current timestamp folder first
        if checkpoint_path.exists():
            lines = [
                line.strip() 
                for line in checkpoint_path.read_text(encoding="utf-8").splitlines() 
                if line.strip()
            ]
            self._logger.info(f"[{self.context.source}] Loaded checkpoint from current run ({len(lines)} items)")
            return lines
        
        # Fall back to most recent previous run
        source_dir = Path(self.context.output_dir) / self.context.source
        if source_dir.exists():
            # Get all timestamp folders, sorted newest first
            run_folders = sorted(
                [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
                key=lambda x: x.name,
                reverse=True
            )
            
            # Try each folder until we find a checkpoint
            for folder in run_folders:
                alt_checkpoint = folder / filename
                if alt_checkpoint.exists():
                    lines = [
                        line.strip() 
                        for line in alt_checkpoint.read_text(encoding="utf-8").splitlines() 
                        if line.strip()
                    ]
                    self._logger.info(f"[{self.context.source}] Loaded checkpoint from previous run: {folder.name} ({len(lines)} items)")
                    return lines
        
        self._logger.error(f"[{self.context.source}] No checkpoint file '{filename}' found in any previous runs")
        return []

