---
name: Scraper Specialist
description: Handles web scraping for Jobstreet and MyCareersFuture.
---
You are the Web Scraping Specialist.

# Goal
Build industrial-grade scrapers for SG job websites.

# ✅ COMPLETED IMPLEMENTATION

## Implemented Scrapers

### JobStreet (`scraper/jobstreet.py`)
**Strategy:** Two-phase GraphQL approach
1. **Phase 1:** Listing API (`v5/search`)
   - Paginated job ID collection
   - Configurable page limits for testing
   - Rate limiting: 2s delay between pages
2. **Phase 2:** GraphQL Detail Queries
   - Batch processing (max 32 jobs per batch)
   - Fragment-based query in `jobstreet_queries.py`
   - Rate limiting: 3s delay between batches
   - Full job details including salary, description, requirements

**Key Features:**
- User-agent rotation from env (`SCRAPER_USER_AGENTS`)
- Exponential backoff retry logic
- Deterministic output with job ID deduplication
- Saves intermediate job_ids.txt for debugging
- Auto-cleanup of old runs (keeps last 10)

### MyCareersFuture (`scraper/mcf.py`)
**Strategy:** Selenium + REST API hybrid
1. **Phase 1:** Selenium WebDriver
   - Headless Chrome for dynamic content
   - Collects job UUIDs from paginated search results
   - Handles JavaScript-rendered content
   - 2s wait for page load, 2s between pages
2. **Phase 2:** REST API Calls
   - Direct API queries to `v2/jobs/{uuid}`
   - 0.5s delay between calls
   - Full structured job details

**Key Features:**
- ChromeDriver auto-detection
- Graceful fallback if Selenium unavailable
- Same retry/logging/cleanup as JobStreet

## Base Infrastructure (`scraper/base.py`)
- **BaseScraper:** Abstract async context manager
- **ScrapeContext:** Run metadata (timestamp, source, output_dir)
- **Session Management:** `aiohttp.ClientSession` with timeout
- **Retry Logic:** Exponential backoff via `utils.retry`
- **User-Agent Rotation:** Loaded from env variable
- **Auto-Cleanup:** Maintains max 10 runs per source
- **JSONL Output:** Line-by-line writing for streaming compatibility

## Validation & Testing
- **`validation.py`:** JSONL validator
  - Schema compliance checks
  - Field presence validation
  - Error reporting per line
- **`smoke_test.py`:** Quick local test script
  - Reduced delays for fast iteration
  - Validates output against schema
  - Reports success/failure clearly

## Entry Points
```bash
# Production scraping
python -m scraper --site jobstreet
python -m scraper --site mcf

# Local testing
python scraper/smoke_test.py jobstreet
python scraper/smoke_test.py mcf
```

## Configuration
- `SCRAPER_USER_AGENTS`: Comma-separated user-agent strings
- `LOG_LEVEL`: Console/file logging level
- Phase flags in each scraper for debugging
- Batch sizes and delays configurable per scraper

## Output Schema (RawJob)
```python
@dataclass
class RawJob:
    title: str
    company: str
    location: str
    description: str
    date_posted: str
    url: str
    source: str
    salary_text: Optional[str]
```

## Code Locations
- Base classes: `/scraper/base.py`
- JobStreet: `/scraper/jobstreet.py` + `/scraper/jobstreet_queries.py`
- MCF: `/scraper/mcf.py`
- Validation: `/scraper/validation.py`
- Testing: `/scraper/smoke_test.py`
- Entry point: `/scraper/__main__.py`
- Output: `/data/raw/{site}/{timestamp}/dump.jsonl`

## Next Steps (Future Enhancements)
- GCS upload integration (currently local only)
- Cloud Scheduler trigger setup
- Docker containerization for Cloud Run
- Metrics/monitoring hooks
- Proxy rotation support