"""Scraper entry point.

Designed to run as a single-purpose container in Cloud Run.
Each scraper runs in its own container for fault isolation and independent scaling.

Usage:
    python -m scraper --site jobstreet
    python -m scraper --site mcf
"""

import argparse
import asyncio
import sys
from datetime import datetime

from scraper.base import ScrapeContext
from scraper.jobstreet import JobStreetScraper
from scraper.mcf import MCFScraper
from utils.logging import configure_logging

# Configure logging using centralized utility
logger = configure_logging(service_name="scraper")


async def run_scraper(site: str):
    """Run a specific scraper."""
    # Use timestamp for unique output directory per run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    context = ScrapeContext(run_timestamp=run_timestamp, source=site)
    
    if site.lower() == "jobstreet":
        scraper_cls = JobStreetScraper
    elif site.lower() == "mcf" or site.lower() == "mycareersfuture":
        scraper_cls = MCFScraper
    else:
        raise ValueError(f"Unknown site: {site}")

    logger.info(f"[ScraperMain] Initializing {site} scraper at {run_timestamp}")
    
    async with scraper_cls(context=context) as scraper:
        await scraper.run()


async def main():
    parser = argparse.ArgumentParser(
        description="Run a single job scraper. Each site should run in its own container."
    )
    parser.add_argument(
        "--site",
        type=str,
        required=True,
        choices=["jobstreet", "mcf"],
        help="Site to scrape (jobstreet or mcf)",
    )
    
    args = parser.parse_args()
    
    try:
        await run_scraper(args.site)
    except Exception as e:
        logger.error(f"[ScraperMain] Scraper {args.site} failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[ScraperMain] Scraper interrupted by user.")
