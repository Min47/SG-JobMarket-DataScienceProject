"""Debug script to investigate payload JSON structure issues.

This script:
1. Queries raw_jobs from BigQuery
2. Inspects payload JSON structure
3. Tests transformation logic
4. Identifies what's missing
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from google.cloud import bigquery

from utils.config import Settings
from utils.logging import configure_logging
from etl.transform import transform_raw_to_cleaned

configure_logging(service_name="debug_payload")
logger = logging.getLogger(__name__)


def inspect_raw_payloads(limit: int = 10, source: str = "jobstreet") -> None:
    """Query and inspect raw_jobs payloads."""
    settings = Settings.load()
    client = bigquery.Client(project=settings.gcp_project_id)
    
    query = f"""
    SELECT 
        job_id,
        source,
        payload,
        scrape_timestamp
    FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.raw_jobs`
    WHERE lower(source) = lower('{source}')
    ORDER BY scrape_timestamp DESC
    LIMIT {limit}
    """
    
    logger.info(f"Querying {limit} raw jobs from BigQuery...")
    result = client.query(query).result()
    
    jobs = []
    for row in result:
        job_data = {
            "job_id": row["job_id"],
            "source": row["source"],
            "payload": row["payload"]
        }
        jobs.append(job_data)
    
    logger.info(f"Fetched {len(jobs)} jobs. Inspecting payloads...")
    
    for i, job in enumerate(jobs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Job {i+1}: {job['source']}:{job['job_id']}")
        logger.info(f"{'='*80}")
        
        payload = job['payload']
        
        # Print top-level keys
        logger.info(f"Top-level keys: {list(payload.keys())}")
        
        # For JobStreet, check for 'job' key
        if 'job' in payload:
            job_obj = payload['job']
            logger.info(f"  'job' keys: {list(job_obj.keys())}")
            
            # Check critical fields
            logger.info(f"  job.id: {job_obj.get('id', 'MISSING')}")
            logger.info(f"  job.title: {job_obj.get('title', 'MISSING')}")
            logger.info(f"  job.advertiser: {job_obj.get('advertiser', 'MISSING')}")
            
        else:
            logger.info(f"  ⚠️ 'job' key NOT FOUND in payload!")
            logger.info(f"  Available keys: {list(payload.keys())}")
        
        # Check for companyProfile
        if 'companyProfile' in payload:
            logger.info(f"  'companyProfile' exists: {list(payload['companyProfile'].keys())}")
        else:
            logger.info(f"  ⚠️ 'companyProfile' key NOT FOUND")
        
        # Print full payload for first 2 jobs (pretty print)
        if i < 2:
            logger.info(f"\n📄 Full payload (pretty):")
            logger.info(json.dumps(payload, indent=2, default=str)[:2000] + "...")


def test_transformation(job_id: str = None, source: str = "jobstreet") -> None:
    """Test transformation on a specific job."""
    settings = Settings.load()
    client = bigquery.Client(project=settings.gcp_project_id)
    
    if job_id:
        where_clause = f"AND job_id = '{job_id}'"
    else:
        where_clause = ""
    
    query = f"""
    SELECT 
        job_id,
        source,
        scrape_timestamp,
        payload
    FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.raw_jobs`
    WHERE lower(source) = lower('{source}')
        {where_clause}
    ORDER BY scrape_timestamp DESC
    LIMIT 100
    """
    
    logger.info("Testing transformation...")
    result = client.query(query).result()
    
    success_count = 0
    failed_count = 0
    
    for row in result:
        raw_job = {
            "job_id": row["job_id"],
            "source": row["source"],
            "scrape_timestamp": row["scrape_timestamp"],
            "payload": row["payload"]
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Transforming: {raw_job['source']}:{raw_job['job_id']}")
        logger.info(f"{'='*80}")
        
        try:
            cleaned = transform_raw_to_cleaned(raw_job)
            
            if cleaned:
                success_count += 1
                logger.info(f"✅ TRANSFORMATION SUCCESS")
                logger.info(f"\n📋 JOB DETAILS:")
                logger.info(f"  Job ID:         {cleaned.job_id}")
                logger.info(f"  Title:          {cleaned.job_title}")
                logger.info(f"  Company:        {cleaned.company_name}")
                logger.info(f"  Location:       {cleaned.job_location}")
                logger.info(f"  Classification: {cleaned.job_classification}")
                logger.info(f"  Work Type:      {cleaned.job_work_type}")
                
                logger.info(f"\n💰 SALARY INFORMATION:")
                logger.info(f"  Raw Min:        ${cleaned.job_salary_min_sgd_raw}" if cleaned.job_salary_min_sgd_raw else f"  Raw Min:        None")
                logger.info(f"  Raw Max:        ${cleaned.job_salary_max_sgd_raw}" if cleaned.job_salary_max_sgd_raw else f"  Raw Max:        None")
                logger.info(f"  Period:         {cleaned.job_salary_type}" if cleaned.job_salary_type else f"  Period:         None")
                logger.info(f"  Monthly Min:    ${cleaned.job_salary_min_sgd_monthly:.2f}" if cleaned.job_salary_min_sgd_monthly else f"  Monthly Min:    None")
                logger.info(f"  Monthly Max:    ${cleaned.job_salary_max_sgd_monthly:.2f}" if cleaned.job_salary_max_sgd_monthly else f"  Monthly Max:    None")
                logger.info(f"  Currency:       {cleaned.job_currency}")
                
                logger.info(f"\n🏢 COMPANY INFORMATION:")
                logger.info(f"  ID:             {cleaned.company_id if cleaned.company_id else 'N/A'}")
                url_display = cleaned.company_url[:60] + "..." if len(cleaned.company_url) > 60 else cleaned.company_url
                logger.info(f"  URL:            {url_display if url_display else 'N/A'}")
                logger.info(f"  Industry:       {cleaned.company_industry if cleaned.company_industry else 'N/A'}")
                logger.info(f"  Size:           {cleaned.company_size if cleaned.company_size else 'N/A'}")
                
                logger.info(f"\n📅 TIMESTAMPS:")
                logger.info(f"  Posted:         {cleaned.job_posted_timestamp}")
                logger.info(f"  Scraped:        {cleaned.scrape_timestamp}")
                logger.info(f"  BQ Ingestion:   {cleaned.bq_timestamp}")
                
                # Show description preview
                desc_preview = cleaned.job_description[:150].replace('\n', ' ') if cleaned.job_description else 'N/A'
                logger.info(f"\n📝 DESCRIPTION PREVIEW:")
                logger.info(f"  {desc_preview}...")
                
                # Show URL
                logger.info(f"\n🔗 JOB URL:")
                logger.info(f"  {cleaned.job_url}")
            else:
                failed_count += 1
                logger.warning(f"❌ TRANSFORMATION FAILED")
                logger.warning(f"  Returned None")
        
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ TRANSFORMATION EXCEPTION: {e}", exc_info=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Transformation Test Results:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    if success_count + failed_count > 0:
        logger.info(f"  Success Rate: {success_count / (success_count + failed_count) * 100:.1f}%")
    else:
        logger.warning(f"  No jobs found to test!")
    logger.info(f"{'='*80}")


def main():
    """Main debug workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug payload JSON issues")
    parser.add_argument("--inspect", action="store_true", help="Inspect payload structure")
    parser.add_argument("--test", action="store_true", help="Test transformation")
    parser.add_argument("--job-id", type=str, help="Specific job_id to test")
    parser.add_argument("--limit", type=int, default=10, help="Number of jobs to inspect")
    parser.add_argument("--source", type=str, default="jobstreet", help="Source to test (jobstreet or mcf)")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_raw_payloads(limit=args.limit, source=args.source)
    
    if args.test:
        test_transformation(job_id=args.job_id, source=args.source)
    
    if not args.inspect and not args.test:
        # Default: do both
        inspect_raw_payloads(limit=100, source=args.source)
        test_transformation(source=args.source)


if __name__ == "__main__":
    main()
