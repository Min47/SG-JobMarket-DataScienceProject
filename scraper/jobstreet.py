"""JobStreet scraper implementation.

Two-phase strategy:
1. Fetch job IDs from listing API (v5)
2. Fetch individual job details via GraphQL
"""

from __future__ import annotations

import asyncio
import datetime
import json
import random
from typing import Any, AsyncIterator, Dict, Iterable, List
from urllib.parse import urlencode

import aiohttp

from scraper.base import BaseScraper
from scraper.jobstreet_queries import GRAPHQL_FRAGMENT
from utils.retry import RetryPolicy, retry_async_call
from utils.schemas import RawJob

# Configuration
BATCH_SIZE = 32  # Exceeding 32 will cause 'Query complexity limit exceeded' errors
LISTING_PAGE_DELAY = 2.0  # Seconds between listing API calls
GRAPHQL_BATCH_DELAY = 5.0  # Seconds between batch GraphQL calls (increased to avoid rate limiting)
DATE_RANGE = "2"  # Days: 2

# Phase control (for testing/debugging)
RUN_PHASE_1 = True  # Fetch job IDs from listing API
RUN_PHASE_2 = True  # Fetch details via GraphQL
DEBUGGING = False  # If True, limits to first 2 pages of listings

# JobStreet SG API Endpoints
LISTING_API_URL = "https://sg.jobstreet.com/api/jobsearch/v5/search"
GRAPHQL_API_URL = "https://sg.jobstreet.com/graphql"


class JobStreetScraper(BaseScraper):
    """Scraper for JobStreet Singapore using listing + GraphQL strategy."""

    async def fetch(self) -> AsyncIterator[str]:
        """Fetch job listings then fetch details individually."""
        job_ids: List[str] = []
        
        # Phase 1: Collect job IDs from listing pages
        if RUN_PHASE_1:
            self._logger.info("[JobStreet] \n")
            self._logger.info("[JobStreet] Phase 1: Fetching job IDs from listing API")
            job_ids = await self._fetch_job_ids()
            self._logger.info(f"[JobStreet] Phase 1 Complete: Collected {len(job_ids)} job IDs")
            # Always save final checkpoint after Phase 1 completes
            self._save_job_ids_checkpoint(job_ids)
        else:
            self._logger.info("[JobStreet] Phase 1 SKIPPED: Loading job IDs from checkpoint")
            job_ids = self._load_job_ids_checkpoint()
            if job_ids:
                self._logger.info(f"[JobStreet] Loaded {len(job_ids)} job IDs from checkpoint")
        
        # Validate we have IDs before Phase 2
        if not job_ids:
            self._logger.warning("[JobStreet] No job IDs available - cannot proceed to Phase 2")
            # Still save empty checkpoint if Phase 1 ran
            if RUN_PHASE_1:
                self._save_job_ids_checkpoint([])
            return
        
        # Phase 2: Batch fetch details via GraphQL (40 jobs per query)
        if RUN_PHASE_2:
            self._logger.info("[JobStreet] \n")
            self._logger.info(f"[JobStreet] Phase 2: Fetching {len(job_ids)} job details in batches of {BATCH_SIZE}")
            
            # Rate limit tracking across all batches (CARRIES FORWARD)
            global_rate_limit_level = 0  # Global escalation: 0=none, 1=10min, 2=30min, 3=1hr, 4=abort
            total_rate_limit_count = 0  # Total number of rate limits hit
            
            for i in range(0, len(job_ids), BATCH_SIZE):
                batch = job_ids[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(job_ids) + BATCH_SIZE - 1) // BATCH_SIZE
                self._logger.info(f"[JobStreet] > Fetching Batch: {batch_num} / {total_batches} ({len(batch)} jobs)")
                
                try:
                    await asyncio.sleep(GRAPHQL_BATCH_DELAY)
                    
                    payload = await self._fetch_graphql_batch(batch)
                    
                    # Progressive rate limit retry: 10min -> 30min -> 1hr -> end Phase 2
                    # Level carries forward across ALL batches
                    while self._is_rate_limited(payload):
                        global_rate_limit_level += 1  # Escalate globally (never resets)
                        total_rate_limit_count += 1
                        
                        if global_rate_limit_level == 1:
                            # First rate limit ANYWHERE: wait 10 minutes
                            self._logger.warning(
                                f"[JobStreet] Rate limit #{total_rate_limit_count} on batch {batch_num}. "
                                f"Level 1: Pausing for 10 minutes..."
                            )
                            await asyncio.sleep(600)  # 10 minutes
                            
                        elif global_rate_limit_level == 2:
                            # Second rate limit ANYWHERE: wait 30 minutes
                            self._logger.warning(
                                f"[JobStreet] Rate limit #{total_rate_limit_count} on batch {batch_num}. "
                                f"Level 2: Pausing for 30 minutes..."
                            )
                            await asyncio.sleep(1800)  # 30 minutes
                            
                        elif global_rate_limit_level == 3:
                            # Third rate limit ANYWHERE: wait 1 hour (SEVERE)
                            self._logger.error(
                                f"[JobStreet] Rate limit #{total_rate_limit_count} on batch {batch_num}. "
                                f"Level 3 SEVERE: Pausing for 1 HOUR... (FINAL WARNING)"
                            )
                            await asyncio.sleep(3600)  # 1 hour
                            
                        else:
                            # Fourth rate limit ANYWHERE: abort Phase 2 entirely
                            self._logger.error(
                                f"[JobStreet] Rate limit #{total_rate_limit_count} on batch {batch_num}. "
                                f"Level 4: ABORTING PHASE 2 - exceeded retry limit. "
                                f"Successfully processed {batch_num - 1}/{total_batches} batches."
                            )
                            return  # End Phase 2 completely
                        
                        # Retry the batch after waiting
                        self._logger.info(f"[JobStreet] Resuming and retrying batch {batch_num} (Level {global_rate_limit_level})...")
                        payload = await self._fetch_graphql_batch(batch)
                    
                    yield payload
                    
                except Exception as e:
                    self._logger.error(f"[JobStreet] Failed to fetch batch starting at {i}: {e}")
                    continue
        else:
            self._logger.info("[JobStreet] Phase 2 SKIPPED: Job IDs saved to checkpoint only")
    
    async def _fetch_job_ids(self) -> List[str]:
        """Fetch job IDs from listing API (saves incrementally per page)."""
        job_ids = []
        page = 1
        total_pages = None
        
        # Create checkpoint file early for incremental saves
        checkpoint_path = self._get_checkpoint_path("job_ids.txt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("", encoding="utf-8")  # Start fresh
        
        while True:
            self._logger.info(f"[JobStreet] > Fetching Page: {page}{f' / {total_pages}' if total_pages else ' / x'}")
            
            params = {
                "siteKey": "SG-Main",
                "page": page,
                "daterange": DATE_RANGE,
                "pageSize": "32",
                "sortmode": "ListedDate",
                "where": "Singapore",
            }
            
            url = f"{LISTING_API_URL}?{urlencode(params)}"
            
            try:
                await asyncio.sleep(LISTING_PAGE_DELAY)
                
                payload = await self._fetch_url(url)
                data = json.loads(payload)
                
                # Update total_pages from API response
                current_total_jobs = data.get("totalCount", 0)
                current_total_pages = (current_total_jobs + 31) // 32
                if total_pages is None:
                    total_pages = current_total_pages
                    self._logger.info(f"[JobStreet] Total pages available: {total_pages}")
                elif current_total_pages != total_pages:
                    self._logger.info(f"[JobStreet] Total pages changed from {total_pages} to {current_total_pages}")
                    total_pages = current_total_pages
                
                jobs = data.get("data", [])
                if not jobs:
                    self._logger.info(f"[JobStreet] No jobs found on page {page}, ending")
                    break
                
                page_job_ids = []
                for job in jobs:
                    job_id = job.get("id")
                    if job_id:
                        job_ids.append(str(job_id))
                        page_job_ids.append(str(job_id))
                
                # Save this page's job IDs immediately (incremental safety)
                if page_job_ids:
                    with open(checkpoint_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(page_job_ids) + "\n")
                    self._logger.debug(f"[JobStreet] Checkpoint: Saved {len(page_job_ids)} IDs from page {page}")
                
                # Check if we've reached the last page
                if page >= total_pages:
                    self._logger.info(f"[JobStreet] Reached last page ({total_pages})")
                    break
                    
                # For debugging purpose, limit to first 2 pages
                if DEBUGGING and page >= 2:
                    self._logger.info("[JobStreet] Reached debugging page limit (2), ending")
                    break

                page += 1
                
            except Exception as e:
                self._logger.error(f"[JobStreet] Failed to fetch listing page {page}: {e}")
                break
        
        return job_ids
    
    def _build_batch_query(self, job_ids: List[str]) -> str:
        """Build a batch GraphQL query for multiple job IDs."""
        # Build parameter definitions: $jobId1: ID!, $jobId2: ID!, ...
        param_defs = []
        param_defs.append("$zone: Zone!")
        param_defs.append("$locale: Locale!")
        param_defs.append("$languageCode: LanguageCodeIso!")
        param_defs.append("$countryCode: CountryCodeIso2!")
        param_defs.append("$timezone: Timezone!")
        
        for idx in range(1, len(job_ids) + 1):
            param_defs.append(f"$jobId{idx}: ID!")
        
        # Build individual job queries: jobDetails1: jobDetails(id: $jobId1) { ...job }
        job_queries = []
        for idx in range(1, len(job_ids) + 1):
            job_queries.append(f"""
    jobDetails{idx}: jobDetails(id: $jobId{idx}) {{
        ...job
    }}""")
        
        # Combine into full query
        query = f"""
query BatchJobDetails({', '.join(param_defs)}) {{
{''.join(job_queries)}
}}

{GRAPHQL_FRAGMENT}
"""
        return query
    
    async def _fetch_graphql_batch(self, job_ids: List[str]) -> str:
        """Fetch job details via GraphQL for a batch of IDs (up to 40)."""
        # Build dynamic batch query
        query = self._build_batch_query(job_ids)
        
        # Build variables
        variables = {
            "zone": "asia-7",
            "locale": "en-SG",
            "languageCode": "en",
            "countryCode": "SG",
            "timezone": "Asia/Singapore"
        }
        
        # Add job IDs as jobId1, jobId2, ...
        for idx, job_id in enumerate(job_ids, 1):
            variables[f"jobId{idx}"] = job_id
        
        payload = {
            "operationName": "BatchJobDetails",
            "query": query,
            "variables": variables
        }
        
        self._logger.debug(f"[JobStreet] GraphQL batch request for {len(job_ids)} jobs")
        
        # Direct aiohttp POST (with retry logic)
        async def _graphql_request() -> str:
            await asyncio.sleep(random.uniform(1.0, 2.5))
            
            async with self.session.post(
                GRAPHQL_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._logger.error(f"[JobStreet] GraphQL error ({response.status}): {error_text[:300]}")
                response.raise_for_status()
                return await response.text()
        
        return await retry_async_call(
            _graphql_request,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=2.0),
            retry_on=(aiohttp.ClientError, asyncio.TimeoutError),
            on_retry=lambda attempt, exc: self._logger.warning(
                f"[JobStreet] GraphQL batch retry {attempt}: {exc}"
            ),
        )

    def parse(self, payload: str) -> Iterable[RawJob]:
        """Parse GraphQL batch response into RawJob objects."""
        try:
            data = json.loads(payload)
            
            # Log GraphQL errors but continue processing successful jobs
            if "errors" in data:
                for idx, error in enumerate(data['errors'], 1):
                    error_msg = error.get('message', 'Unknown')
                    error_path = error.get('path', [])
                    error_extensions = error.get('extensions', {})
                    
                    # Extract job_id from error path if available
                    job_id = "unknown"
                    if error_path and len(error_path) > 0:
                        # Path typically looks like: ["jobDetails3", "job", "field"]
                        job_details_key = error_path[0] if isinstance(error_path[0], str) and error_path[0].startswith("jobDetails") else None
                        if job_details_key and job_details_key in data.get("data", {}):
                            job_data = data["data"][job_details_key]
                            if job_data and isinstance(job_data, dict):
                                # Try to extract ID from the job data
                                if "job" in job_data and isinstance(job_data["job"], dict):
                                    job_id = job_data["job"].get("id", "unknown")
                                elif "id" in job_data:
                                    job_id = job_data["id"]
                    
                    self._logger.error(f"[JobStreet] GraphQL Error {idx} (job_id: {job_id}): {error_msg}")
                    if error_path:
                        self._logger.error(f"[JobStreet]   Path: {error_path}")
                    if error_extensions:
                        self._logger.error(f"[JobStreet]   Extensions: {error_extensions}")
                # Don't return - continue to process any successful jobs in the batch
            
            response_data = data.get("data", {})
            
            if not response_data:
                self._logger.warning("[JobStreet] No data in GraphQL response")
                return
            
            # Iterate through jobDetails1, jobDetails2, ... in the response
            parsed_count = 0
            null_count = 0
            expired_count = 0
            error_count = 0
            
            for key in sorted(response_data.keys()):
                if not key.startswith("jobDetails"):
                    continue
                
                try:
                    job_full = response_data[key]
                    if not job_full:
                        null_count += 1
                        continue
                    
                    job = job_full.get("job")
                    if not job:
                        null_count += 1
                        continue
                    
                    if job.get("isExpired"):
                        expired_count += 1
                        continue
                    
                    # Wrap individual job parsing in try-except to prevent one bad job from breaking the batch
                    try:
                        yield self._parse_single_job(job, job_full)
                        parsed_count += 1
                    except (KeyError, AttributeError, TypeError) as parse_err:
                        error_count += 1
                        job_id = job.get("id", "unknown")
                        self._logger.warning(
                            f"[JobStreet] Failed to parse job {job_id} in {key}: "
                            f"{type(parse_err).__name__}: {parse_err}"
                        )
                        # Log first occurrence with more details for debugging
                        if error_count == 1:
                            self._logger.debug(f"[JobStreet] Job data structure: {job.keys()}")
                        continue
                        
                except Exception as item_err:
                    error_count += 1
                    self._logger.warning(
                        f"[JobStreet] Error processing {key}: {item_err}"
                    )
                    continue
            
            self._logger.debug(
                f"[JobStreet] Batch results: {parsed_count} parsed, {expired_count} expired, "
                f"{null_count} null, {error_count} errors"
            )
                
        except json.JSONDecodeError as e:
            self._logger.error(f"[JobStreet] Failed to decode JSON: {e}")
        except Exception as e:
            self._logger.error(f"[JobStreet] Error parsing payload: {e}", exc_info=True)

    def _parse_single_job(self, job: Dict[str, Any], job_full: Dict[str, Any]) -> RawJob:
        """Extract fields from a single GraphQL job object.
        
        Uses safe navigation with .get() to handle missing keys gracefully.
        """
        
        job_id = job.get("id", "")
        
        # Safe extraction with defaults for all fields
        advertiser = job.get("advertiser") or {}
        location_obj = job.get("location") or {}
        listed_at = job.get("listedAt") or {}
        salary_obj = job.get("salary")
        work_types_raw = job.get("workTypes")
        classifications_raw = job.get("classifications")
        
        # Ensure work_types and classifications are lists
        work_types = work_types_raw if isinstance(work_types_raw, list) else []
        classifications = classifications_raw if isinstance(classifications_raw, list) else []
        
        # Build normalized payload with extracted fields
        payload = {
            "title": job.get("title", ""),
            "company": advertiser.get("name", ""),
            "location": location_obj.get("label", ""),
            "description": job.get("content", "") or job.get("abstract", ""),
            "date_posted": listed_at.get("dateTimeUtc", ""),
            "url": job.get("shareLink", "") or f"https://www.jobstreet.com.sg/job/{job_id}",
            "salary_text": salary_obj.get("label") if salary_obj else None,
            # Additional fields from GraphQL response
            "work_type": work_types[0].get("label") if (work_types and isinstance(work_types[0], dict)) else None,
            "is_verified": advertiser.get("isVerified"),
            "classification": [c.get("label", "") for c in classifications if isinstance(c, dict)],
            "raw": job_full  # Store complete raw response for reference
        }
        
        return RawJob(
            job_id=job_id,
            source="JobStreet",
            scrape_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            payload=payload
        )
    
    def _is_rate_limited(self, payload: str) -> bool:
        """Check if GraphQL response contains rate limit error."""
        try:
            data = json.loads(payload)
            if "errors" in data:
                for error in data.get("errors", []):
                    error_code = error.get("extensions", {}).get("code", "")
                    error_msg = error.get("message", "").lower()
                    if error_code == "RATE_LIMITED" or "rate limit" in error_msg or "too many requests" in error_msg:
                        return True
            return False
        except (json.JSONDecodeError, Exception):
            return False
    
    def _save_job_ids_checkpoint(self, job_ids: List[str]) -> None:
        """Save job IDs to checkpoint file."""
        checkpoint_path = self._get_checkpoint_path("job_ids.txt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("\n".join(job_ids) + "\n" if job_ids else "", encoding="utf-8")
        self._logger.info(f"[JobStreet] Saved {len(job_ids)} job IDs to {checkpoint_path.name}")
    
    def _load_job_ids_checkpoint(self) -> List[str]:
        """Load job IDs from checkpoint file using base class method."""
        return self._load_checkpoint("job_ids.txt")
