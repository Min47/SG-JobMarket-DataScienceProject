"""
Cloud Function entrypoint for daily embedding generation.

Triggered by Cloud Scheduler (HTTP POST request).
Processes jobs from yesterday by default (buffer time for scrapers).

Environment Variables:
    GCP_PROJECT_ID: Google Cloud project ID
    BQ_DATASET_ID: BigQuery dataset ID (default: sg_job_market)
    PROCESS_TODAY: Set to 'true' to process today's jobs instead of yesterday
"""

import functions_framework
from datetime import datetime, timedelta, timezone
import logging
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_embeddings import generate_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROCESS_TODAY = os.getenv("PROCESS_TODAY", "false").lower() == "true"


@functions_framework.http
def generate_daily_embeddings(request):
    """
    Cloud Function HTTP handler for daily embedding generation.
    
    Triggered by: Cloud Scheduler (POST request)
    Schedule: Daily at 3:00 AM SGT (after scrapers complete)
    
    Query Parameters:
        process_today: 'true' to process today's jobs instead of yesterday
    
    Returns:
        JSON with processing statistics
    """
    try:
        logger.info("="*70)
        logger.info("Cloud Function: Daily Embedding Generation")
        logger.info("="*70)
        
        # Determine target date (default: yesterday)
        utc = timezone.utc
        process_today = request.args.get('process_today', 'false').lower() == 'true'
        process_today = process_today or PROCESS_TODAY  # ENV var overrides default
        
        if process_today:
            target_date = datetime.now(utc)
            logger.info("⚡ Manual override: Processing TODAY's jobs")
        else:
            target_date = datetime.now(utc) - timedelta(days=1)
            logger.info("📅 Default mode: Processing YESTERDAY's jobs (buffer time)")
        
        logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')} (UTC)")
        logger.info("="*70)
        
        # Call the main generate_embeddings function with date filter
        result = generate_embeddings(
            limit=None,  # No limit, process all jobs for target date
            batch_size=32,
            only_new=True,  # Only jobs without embeddings
            dry_run=False,
            target_date=target_date
        )
        
        logger.info("="*70)
        logger.info(f"✅ Complete: {result}")
        logger.info("="*70)
        
        return json.dumps(result), 200
        
    except Exception as e:
        logger.error(f"❌ Error during embedding generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }), 500
