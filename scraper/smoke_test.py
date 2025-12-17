"""Smoke test for scrapers.

Quick validation script to test scraper functionality locally.

Usage:
    python -m scraper.smoke_test jobstreet
    python -m scraper.smoke_test mcf
"""

import asyncio
import sys
from pathlib import Path

from scraper.base import ScrapeContext
from scraper.jobstreet import JobStreetScraper
from scraper.mcf import MCFScraper
from scraper.validation import validate_jsonl_file
from utils.logging import configure_logging

logger = configure_logging(service_name="smoke_test")


async def smoke_test(site: str):
    """Run a minimal scrape test."""
    logger.info(f"=== Smoke Test: {site} ===")
    
    # Create test context
    context = ScrapeContext(
        run_timestamp="smoke_test",
        source=site,
        output_dir="data/raw"
    )
    
    # Select scraper
    if site == "jobstreet":
        scraper_cls = JobStreetScraper
        # Override config for quick test
        import scraper.jobstreet as js_module
        original_delay = js_module.LISTING_PAGE_DELAY
        js_module.LISTING_PAGE_DELAY = 0.5  # Faster for testing
    elif site == "mcf":
        scraper_cls = MCFScraper
        # Override config for quick test
        import scraper.mcf as mcf_module
        original_pages = mcf_module.MAX_PAGES
        mcf_module.MAX_PAGES = 2  # Only 2 pages for smoke test
    else:
        logger.error(f"Unknown site: {site}")
        return False
    
    try:
        logger.info("Starting scraper...")
        async with scraper_cls(context=context) as scraper:
            await scraper.run()
        
        # Validate output
        output_path = Path(context.output_dir) / site / "smoke_test" / "dump.jsonl"
        logger.info(f"Validating output: {output_path}")
        
        is_valid, messages = validate_jsonl_file(output_path)
        
        for msg in messages:
            if is_valid:
                logger.info(msg)
            else:
                logger.error(msg)
        
        if is_valid:
            logger.info("✓ Smoke test PASSED")
            return True
        else:
            logger.error("✗ Smoke test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Smoke test failed with exception: {e}", exc_info=True)
        return False
    finally:
        # Restore original config
        if site == "jobstreet":
            js_module.LISTING_PAGE_DELAY = original_delay
        elif site == "mcf":
            mcf_module.MAX_PAGES = original_pages


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["jobstreet", "mcf"]:
        print("Usage: python -m scraper.smoke_test [jobstreet|mcf]")
        sys.exit(1)
    
    site = sys.argv[1]
    success = asyncio.run(smoke_test(site))
    sys.exit(0 if success else 1)
