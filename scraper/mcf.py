"""MyCareersFuture scraper implementation.

Two-phase strategy:
1. Use Selenium to collect job UUIDs from search pages
2. Fetch details via API for each UUID

Reliability features:
- Chrome restart every 10 pages to prevent memory leaks
- Page load timeouts (30s) and script timeouts (30s)
- Automatic retry on page load failures
- Incremental UUID checkpointing per page
- Graceful error handling (skip failed pages, continue scraping)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Dict, Iterable, List

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from dotenv import load_dotenv

from scraper.base import BaseScraper
from utils.schemas import RawJob

# Load environment variables (critical for Docker)
load_dotenv()

# Configuration
MAX_PAGES = 100  # Maximum number of search pages to scrape
SELENIUM_PAGE_DELAY = 2.0  # Seconds to wait for dynamic content
SELENIUM_BETWEEN_PAGES = 2.0  # Seconds between page loads
API_CALL_DELAY = 0.5  # Seconds between API detail calls
PAGE_LOAD_TIMEOUT = 30  # Maximum seconds to wait for page load
SCRIPT_TIMEOUT = 30  # Maximum seconds for script execution
ELEMENT_WAIT_TIMEOUT = 15  # Maximum seconds to wait for elements
CHROME_RESTART_INTERVAL = 10  # Restart Chrome every N pages to prevent memory leaks

# Phase control (for testing/debugging)
RUN_PHASE_1 = True  # Collect UUIDs via Selenium
RUN_PHASE_2 = True  # Fetch details via API
DEBUGGING = False  # If True, limits to first 2 pages of listings

# MCF URLs
SEARCH_URL = "https://www.mycareersfuture.gov.sg/search?sortBy=new_posting_date&page={page}"
API_DETAIL_URL = "https://api.mycareersfuture.gov.sg/v2/jobs/{uuid}"


class MCFScraper(BaseScraper):
    """Scraper for MyCareersFuture using Selenium + API strategy."""

    async def fetch(self) -> AsyncIterator[str]:
        """Fetch job UUIDs via Selenium then details via API."""
        if not SELENIUM_AVAILABLE:
            self._logger.error("[MCF] Selenium not available. Install: pip install selenium")
            return
        
        uuids: List[str] = []
        
        # Phase 1: Collect UUIDs from search pages using Selenium
        if RUN_PHASE_1:
            self._logger.info("[MCF] \n")
            self._logger.info("[MCF] Phase 1: Collecting job UUIDs via Selenium")
            uuids = await self._fetch_job_uuids()
            self._logger.info(f"[MCF] Phase 1 Complete: Collected {len(uuids)} job UUIDs")
            # Note: UUIDs saved incrementally per page to checkpoint file
        else:
            self._logger.info("[MCF] Phase 1 SKIPPED: Loading UUIDs from checkpoint")
            uuids = self._load_uuids_checkpoint()
            if uuids:
                self._logger.info(f"[MCF] Loaded {len(uuids)} UUIDs from checkpoint")
        
        # Validate we have UUIDs before Phase 2
        if not uuids:
            self._logger.warning("[MCF] No UUIDs available - cannot proceed to Phase 2")
            return
        
        # Phase 2: Fetch details for each UUID via API
        if RUN_PHASE_2:
            self._logger.info("[MCF] \n")
            self._logger.info(f"[MCF] Phase 2: Fetching {len(uuids)} job details via API")
            
            for i, uuid in enumerate(uuids, 1):
                if i % 10 == 0:
                    self._logger.info(f"[MCF] Progress: {i} / {len(uuids)} jobs")
                
                try:
                    await asyncio.sleep(API_CALL_DELAY)
                    
                    url = API_DETAIL_URL.format(uuid=uuid)
                    payload = await self._fetch_url(url)
                    yield payload
                except Exception as e:
                    self._logger.error(f"[MCF] Failed to fetch job {uuid}: {e}")
                    continue
        else:
            self._logger.info("[MCF] Phase 2 SKIPPED: UUIDs saved to checkpoint only")
    
    async def _fetch_job_uuids(self) -> List[str]:
        """Use Selenium to collect job UUIDs from search pages."""
        uuids = []

        # Create checkpoint file early for incremental saves
        checkpoint_path = self._get_checkpoint_path("job_uuids.txt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("", encoding="utf-8")  # Start fresh
        
        # Get user agent from environment
        user_agent = os.getenv("SCRAPER_USER_AGENTS", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        # if "," in user_agent:
        #     user_agent = user_agent.split(",")[0].strip()  # Use first one for Selenium
        
        # Setup headless Chrome options (reusable for restarts)
        def create_chrome_options():
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--single-process")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument(f"user-agent={user_agent}")
            # Memory optimization
            chrome_options.add_argument("--disable-cache")
            chrome_options.add_argument("--aggressive-cache-discard")
            chrome_options.add_argument("--disable-application-cache")
            return chrome_options
        
        # Initialize Chrome driver
        driver = webdriver.Chrome(options=create_chrome_options())
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.set_script_timeout(SCRIPT_TIMEOUT)
        
        try:
            page = 0
            
            while page < MAX_PAGES:
                # Restart Chrome periodically to prevent memory leaks and timeouts
                if page > 0 and page % CHROME_RESTART_INTERVAL == 0:
                    self._logger.info(f"[MCF] Restarting Chrome (page {page}) to clear memory")
                    try:
                        driver.quit()
                    except Exception as e:
                        self._logger.warning(f"[MCF] Error quitting driver: {e}")
                    
                    time.sleep(2)  # Brief pause before restart
                    driver = webdriver.Chrome(options=create_chrome_options())
                    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
                    driver.set_script_timeout(SCRIPT_TIMEOUT)
                    self._logger.info(f"[MCF] Chrome restarted successfully")
                
                url = SEARCH_URL.format(page=page)
                self._logger.info(f"[MCF] > Loading page {page + 1} / {MAX_PAGES}")
                
                try:
                    driver.get(url)
                    job_cards_container = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-list']"))
                    )
                    time.sleep(SELENIUM_PAGE_DELAY)
                except Exception as e:
                    self._logger.error(f"[MCF] Timeout or error loading page {page + 1}: {e}")
                    # Try one more time before giving up
                    self._logger.info(f"[MCF] Retrying page {page + 1} after error")
                    time.sleep(5)
                    try:
                        driver.get(url)
                        job_cards_container = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-list']"))
                        )
                        time.sleep(SELENIUM_PAGE_DELAY)
                    except Exception as retry_error:
                        self._logger.error(f"[MCF] Retry failed for page {page + 1}: {retry_error}") 
                        # Continue to next page instead of breaking entire scrape
                        page += 1
                        continue
                
                # Extract job card elements - primary selector based on actual HTML structure
                try:
                    # Job cards have data-testid="white-job-card-{index}"
                    job_cards = job_cards_container.find_elements(By.CSS_SELECTOR, "div[data-testid^='white-job-card-']")
                    
                    if not job_cards:
                        self._logger.error(f"[MCF] No job cards found on page {page + 1}, ending scrape")
                        break
                    
                    self._logger.info(f"[MCF] Found {len(job_cards)} job cards")
                    
                    page_uuids = []
                    for card in job_cards:
                        # Extract UUID from job card link
                        uuid = None
                        href = None
                        
                        try:
                            # Primary method: Find the link with data-testid="job-card-link"
                            try:
                                job_link = card.find_element(By.CSS_SELECTOR, "a[data-testid='job-card-link']")
                                href = job_link.get_attribute("href")
                            except Exception:
                                pass
                            
                            # Fallback 1: Try any link with /job/ in href
                            if not href:
                                try:
                                    all_links = card.find_elements(By.TAG_NAME, "a")
                                    for link in all_links:
                                        link_href = link.get_attribute("href")
                                        if link_href and "/job/" in link_href:
                                            href = link_href
                                            self._logger.debug(f"[MCF] Used fallback link selector")
                                            break
                                except Exception:
                                    pass
                            
                            # Fallback 2: Check card's id attribute (e.g., "job-card-0" contains UUID in some cases)
                            if not href:
                                try:
                                    card_id = card.get_attribute("id")
                                    # Sometimes the bookmark checkbox has the UUID as its id
                                    checkbox = card.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                                    checkbox_id = checkbox.get_attribute("id")
                                    if checkbox_id and len(checkbox_id) == 32 and checkbox_id.isalnum():
                                        uuid = checkbox_id
                                        self._logger.debug(f"[MCF] Used fallback checkbox id: {uuid}")
                                except Exception:
                                    pass
                            
                            # Extract UUID from URL if we have an href
                            if href and "/job/" in href and not uuid:
                                # Remove query parameters first
                                url_without_query = href.split("?")[0]
                                
                                # Get the last path segment and extract UUID (last part after splitting by -)
                                path_segments = url_without_query.rstrip("/").split("/")
                                last_segment = path_segments[-1]  # e.g., "production-engineer-intern-keystone-cable-000a9afade92b79bd6827ba6fa694148"
                                
                                # UUID is the last part after the final hyphen
                                uuid = last_segment.split("-")[-1]
                            
                            # Validate and store UUID
                            if uuid and len(uuid) == 32 and uuid.isalnum() and uuid not in uuids:
                                uuids.append(uuid)
                                page_uuids.append(uuid)
                                # self._logger.debug(f"[MCF] Extracted UUID: {uuid}" + (f" from {href}" if href else " from card attributes"))
                            elif uuid:
                                self._logger.warning(f"[MCF] Invalid or duplicate UUID: {uuid}")
                            else:
                                self._logger.warning(f"[MCF] Could not extract UUID from card")
                                
                        except Exception as e:
                            self._logger.info(f"[MCF] Failed to extract UUID from card: {e}")
                            continue
                    
                    # Save this page's UUIDs immediately (incremental safety)
                    if page_uuids:
                        with open(checkpoint_path, "a", encoding="utf-8") as f:
                            f.write("\n".join(page_uuids) + "\n")
                        self._logger.debug(f"[MCF] Checkpoint: Saved {len(page_uuids)} UUIDs from page {page + 1}")

                    # For debugging purpose, limit to first 2 pages
                    if DEBUGGING and (page + 1) >= 2:
                        self._logger.info("[MCF] Reached debugging page limit (2), ending")
                        break
                    
                    page += 1
                    time.sleep(SELENIUM_BETWEEN_PAGES)
                    
                except Exception as e:
                    self._logger.error(f"[MCF] Error extracting UUIDs from page {page}: {e}")
                    # Don't break - save progress and continue to next page
                    page += 1
                    continue
        
        except Exception as e:
            self._logger.error(f"[MCF] Fatal error in Phase 1: {e}", exc_info=True)
        finally:
            # Always save checkpoint (even if empty) and quit driver
            try:
                if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                    self._logger.info(f"[MCF] Saving empty checkpoint (collected {len(uuids)} UUIDs)")
                    checkpoint_path.write_text("\n".join(uuids) + "\n" if uuids else "", encoding="utf-8")
            except Exception as e:
                self._logger.error(f"[MCF] Failed to save final checkpoint: {e}")
            
            try:
                driver.quit()
            except Exception as e:
                self._logger.warning(f"[MCF] Error quitting driver in finally block: {e}")
        
        return uuids

    def parse(self, payload: str) -> Iterable[RawJob]:
        """Parse single job API response into RawJob object."""
        try:
            # API returns single job object (not a list)
            item = json.loads(payload)
            yield self._parse_single_job(item)
                
        except json.JSONDecodeError as e:
            self._logger.error(f"[MCF] Failed to decode JSON: {e}")
        except Exception as e:
            self._logger.error(f"[MCF] Error parsing payload: {e}", exc_info=True)

    def _parse_single_job(self, item: Dict[str, Any]) -> RawJob:
        """Extract fields from a single job dictionary."""
        from datetime import datetime, timezone
        
        # Extract UUID
        uuid = item.get("uuid", "")
        job_url = item.get("jobDetailsUrl", "")
        
        # Salary extraction
        salary_min = item.get("salary", {}).get("minimum")
        salary_max = item.get("salary", {}).get("maximum")
        salary_type = item.get("salary", {}).get("type", {}).get("salaryType")
        salary_text = f"{salary_min} - {salary_max} {salary_type}" if salary_min and salary_max else None
        
        # Build payload with extracted fields
        payload = {
            "title": item.get("title", ""),
            "company": item.get("postedCompany", {}).get("name", ""),
            "location": "Singapore",  # MCF is SG only
            "description": item.get("description", ""),
            "date_posted": item.get("metadata", {}).get("updatedAt", ""),
            "url": job_url,
            "salary_text": salary_text,
            "raw": item  # Store complete raw response
        }

        return RawJob(
            job_id=uuid,
            source="MCF",
            scrape_timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload
        )
    
    def _load_uuids_checkpoint(self) -> List[str]:
        """Load UUIDs from checkpoint file using base class method."""
        return self._load_checkpoint("job_uuids.txt")

