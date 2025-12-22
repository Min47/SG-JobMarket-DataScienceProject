"""Transform RawJob → CleanedJob (Main ETL Transformation Logic).

This module contains the core transformation logic that:
1. Extracts fields from raw JSON payloads
2. Applies text cleaning and normalization
3. Parses salary information
4. Builds CleanedJob records ready for BigQuery

**⭐ REVIEW THIS FILE CAREFULLY ⭐**: This is the main data transformation logic.

The transform_raw_to_cleaned() function is called by Stage 2 of the Cloud Function.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from etl.salary_parser import parse_salary_text
from etl.text_cleaning import (
    clean_description,
    normalize_company_name,
    normalize_location,
)
from utils.schemas import CleanedJob, RawJob

logger = logging.getLogger(__name__)


# =============================================================================
# Field Extraction Helpers (Source-Specific)
# =============================================================================

def extract_jobstreet_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fields from JobStreet payload JSON.
    
    JobStreet structure (GraphQL response):
        {
            "job": {
                "id": "89284778",
                "title": "Program Manager, Global Building Infrastructure Protection Services",
                "content": "<p>Job description HTML</p>",
                "shareLink": "https://sg.jobstreet.com/job/89284778?tracking=SHR-WEB-SharedJob-asia-7",
                "listedAt": {"dateTimeUtc": "2025-12-22T12:57:42.384Z"},
                "salary": {"label": "$5000 - $7000 per month"},
                "workTypes": {"label": "Full Time"},
                "advertiser": {"name": "ABC Corp Pte Ltd"},
                "tracking": {
                    "locationInfo": {"location": "Central"},
                    "classificationInfo": {
                        "classification": "Information Technology",
                        "subClassification": "Data Science"
                    }
                }
            },
            "companyProfile": {
                "id": "168550267780768",
                "overview": {
                    "description": {"paragraphs": ["Para 1", "Para 2"]},
                    "industry": "Information & Communication Technology",
                    "size": {"description": "More than 10,000 employees"}
                }
            },
            "companySearchUrl": "https://..."
        }
    
    Args:
        payload: Raw JobStreet JSON payload
        
    Returns:
        Dictionary of extracted fields
    """
    job = payload.get('job', {})
    company_profile = payload.get('companyProfile', {})
    overview = company_profile.get('overview', {})
    
    # Job classification (combine classification + subClassification)
    tracking = job.get('tracking', {})
    classification_info = tracking.get('classificationInfo', {})
    classification = classification_info.get('classification', '')
    sub_classification = classification_info.get('subClassification', '')
    
    if classification and sub_classification:
        full_classification = f"{classification} - {sub_classification}"
    else:
        full_classification = classification or sub_classification or ''
    
    # Company description (join paragraphs)
    desc_obj = overview.get('description', {})
    company_desc_paragraphs = desc_obj.get('paragraphs', []) if isinstance(desc_obj, dict) else []
    company_description = '\n\n'.join(company_desc_paragraphs) if company_desc_paragraphs else ''
    
    return {
        'job_id': str(job.get('id', '')),
        'job_url': job.get('shareLink', ''),
        'job_title': job.get('title', ''),
        'job_description': job.get('content', ''),  # Raw HTML
        'job_location': tracking.get('locationInfo', {}).get('location', ''),
        'job_classification': full_classification,
        'job_work_type': job.get('workTypes', {}).get('label', ''),
        'job_salary_text': job.get('salary', {}).get('label', ''), # Raw salary text
        'job_posted_timestamp': job.get('listedAt', {}).get('dateTimeUtc'),
        
        'company_id': str(company_profile.get('id', '')),
        'company_url': payload.get('companySearchUrl', ''),
        'company_name': job.get('advertiser', {}).get('name', ''),
        'company_description': company_description,
        'company_industry': overview.get('industry', ''),
        'company_size': overview.get('size', {}).get('description', ''),
    }


def extract_mcf_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fields from MyCareersFuture payload JSON.
    
    Handles two formats:
    1. Wrapped format from scraper (with 'raw' key containing full API response)
    2. Direct API response format (backwards compatibility)
    
    Wrapped structure:
        {
            "company": "ANEMO MARKETING SOLUTIONS",
            "date_posted": "2025-12-21T17:03:20.000Z",
            "description": "<p>Job description...</p>",
            "location": "Singapore",
            "title": "Job Title",
            "raw": {
                "uuid": "0020f55051bf3550cc65e1c86c246401",
                "title": "[IMMEDIATE START!] Junior Business Associate- 🌟office hours🌟",
                "description": "<p>Job description plain text</p>",
                "metadata": {
                    "jobDetailsUrl": "hhttps://www.mycareersfuture.gov.sg/job/advertising/immediate-start-junior-business-associate-%F0%9F%8C%9Foffice-hours%F0%9F%8C%9F-anemo-marketing-solutions-0020f55051bf3550cc65e1c86c246401",
                    "updatedAt": "2025-12-21T17:03:20.000Z"
                },
                "address": {
                    "districts": [{"location": "Central"}]
                },
                "categories": [{"category": "Information Technology"}, ...],
                "employmentTypes": [{"employmentType": "Full Time"}, ...],
                "salary": {
                    "minimum": 4000,
                    "maximum": 6000,
                    "type": {"salaryType": "Monthly"}
                },
                "postedCompany": {
                    "uen": "53505888L",
                    "name": "ANEMO MARKETING SOLUTIONS",
                    "description": "<p>Company description plain text</p>",
                    "employeeCount": 50
                    "_links": {"self": {"href": "https://..."}}
                }
            }
        }
    
    Args:
        payload: Raw MCF JSON payload (either wrapped or direct)
        
    Returns:
        Dictionary of extracted fields
    """
    # Check if this is the wrapped format from scraper (has 'raw' key)
    if 'raw' in payload:
        # Extract from nested 'raw' object (contains full MCF API response)
        raw_data = payload.get('raw', {})
        
        # Extract all data from raw_data (not top-level convenience fields)
        # Top-level fields are too generic (e.g., location="Singapore" instead of district)
        job_id = raw_data.get('uuid', '')
        job_title = raw_data.get('title', '')
        job_description = raw_data.get('description', '')
        date_posted = raw_data.get('metadata', {}).get('updatedAt', '')
        
        # Extract detailed fields from nested structures
        metadata = raw_data.get('metadata', {})
        address = raw_data.get('address', {})
        districts = address.get('districts', [])
        categories = raw_data.get('categories', [])
        employment_types = raw_data.get('employmentTypes', [])
        salary = raw_data.get('salary', {})
        posted_company = raw_data.get('postedCompany', {})
        company_links = posted_company.get('_links', {}).get('self', {})
        
        # Get specific location from districts (not generic "Singapore")
        job_location = districts[0].get('location', '') if districts and len(districts) > 0 else ''
        company_name = posted_company.get('name', '')
            
    else:
        # Direct API response format (backwards compatibility)
        metadata = payload.get('metadata', {})
        address = payload.get('address', {})
        districts = address.get('districts', [])
        categories = payload.get('categories', [])
        employment_types = payload.get('employmentTypes', [])
        salary = payload.get('salary', {})
        posted_company = payload.get('postedCompany', {})
        company_links = posted_company.get('_links', {}).get('self', {})
        
        job_id = payload.get('uuid', '')
        job_title = payload.get('title', '')
        job_description = payload.get('description', '')
        job_location = districts[0].get('location', '') if districts else ''
        company_name = posted_company.get('name', '')
        date_posted = metadata.get('updatedAt', '')
    
    # Build salary text from min/max (works for both formats)
    salary_min = salary.get('minimum') if isinstance(salary, dict) else None
    salary_max = salary.get('maximum') if isinstance(salary, dict) else None
    salary_type = salary.get('type', {}).get('salaryType', 'Monthly') if isinstance(salary, dict) else 'Monthly'
    
    if salary_min and salary_max:
        salary_text = f"${salary_min} - ${salary_max} {salary_type}"
    elif salary_min:
        salary_text = f"${salary_min} {salary_type}"
    else:
        salary_text = ''
    
    return {
        'job_id': job_id,
        'job_url': metadata.get('jobDetailsUrl', '') if isinstance(metadata, dict) else '',
        'job_title': job_title,
        'job_description': job_description,
        'job_location': job_location,
        'job_classification': categories[0].get('category', '') if categories and len(categories) > 0 else '',
        'job_work_type': employment_types[0].get('employmentType', '') if employment_types and len(employment_types) > 0 else '',
        'job_salary_text': salary_text,
        'job_posted_timestamp': date_posted,
        
        'company_id': posted_company.get('uen', '') if isinstance(posted_company, dict) else '',
        'company_url': company_links.get('href', '') if isinstance(company_links, dict) else '',
        'company_name': company_name,
        'company_description': posted_company.get('description', '') if isinstance(posted_company, dict) else '',
        'company_industry': '',  # Not available in MCF
        'company_size': str(posted_company.get('employeeCount', '')) if isinstance(posted_company, dict) and posted_company.get('employeeCount') else '',
    }


# =============================================================================
# Main Transformation Function
# =============================================================================

def transform_raw_to_cleaned(raw_job: RawJob) -> Optional[CleanedJob]:
    """Transform a RawJob into a CleanedJob.
    
    This is the core ETL transformation function that:
    1. Extracts fields from source-specific JSON payload
    2. Applies text cleaning (HTML removal, unicode, whitespace)
    3. Parses and normalizes salary
    4. Normalizes company names and locations
    5. Validates required fields
    6. Builds CleanedJob dataclass
    
    Args:
        raw_job: RawJob from raw_jobs table
        
    Returns:
        CleanedJob ready for BigQuery, or None if transformation fails
        
    Raises:
        No exceptions raised - logs errors and returns None
    """
    try:
        # Step 1: Extract fields based on source
        if raw_job.source.lower() == 'jobstreet':
            fields = extract_jobstreet_fields(raw_job.payload)
        elif raw_job.source.lower() == 'mcf':
            fields = extract_mcf_fields(raw_job.payload)
        else:
            logger.warning(f"Unknown source: {raw_job.source}, skipping")
            return None
        
        # Step 2: Validate required fields
        if not fields.get('job_id') or not fields.get('job_title'):
            logger.warning(f"Missing required fields (job_id or title) for {raw_job.source}:{raw_job.job_id}")
            return None
        
        # Step 3: Clean text fields
        job_title = fields.get('job_title', '').strip()

        job_description = clean_description(fields.get('job_description', ''))
        
        company_name_raw = fields.get('company_name', '')
        company_name = normalize_company_name(company_name_raw) if company_name_raw else ''
        
        company_description = clean_description(fields.get('company_description', ''))
        
        location_raw = fields.get('job_location', '')
        location = normalize_location(location_raw) if location_raw else ''
        
        # Step 4: Parse salary
        salary_text = fields.get('job_salary_text', '')
        salary_range = parse_salary_text(salary_text)
        
        # Step 5: Parse timestamps
        posted_timestamp_str = fields.get('job_posted_timestamp')
        if posted_timestamp_str:
            try:
                # Handle ISO 8601 format
                if isinstance(posted_timestamp_str, str):
                    # Remove timezone suffix if present (Z or +00:00)
                    posted_timestamp_str = posted_timestamp_str.replace('Z', '+00:00')
                    job_posted_timestamp = datetime.fromisoformat(posted_timestamp_str)
                else:
                    job_posted_timestamp = posted_timestamp_str
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not parse posted timestamp '{posted_timestamp_str}': {e}")
                job_posted_timestamp = raw_job.scrape_timestamp  # Fallback
        else:
            job_posted_timestamp = raw_job.scrape_timestamp  # Fallback
        
        # Step 6: Build CleanedJob
        cleaned_job = CleanedJob(
            # Metadata
            source=raw_job.source,
            scrape_timestamp=raw_job.scrape_timestamp,
            bq_timestamp=datetime.now(timezone.utc),
            
            # Job fields
            job_id=fields.get('job_id', ''),
            job_url=fields.get('job_url', ''),
            job_title=job_title,
            job_description=job_description,
            job_location=location,
            job_classification=fields.get('job_classification', ''),
            job_work_type=fields.get('job_work_type', ''),
            
            # Salary (raw)
            job_salary_min_sgd_raw=None,  # Will be parsed from salary_text in Stage 2
            job_salary_max_sgd_raw=None,
            job_salary_type=fields.get('job_salary_text', ''),  # Store original text for now
            
            # Salary (monthly converted)
            job_salary_min_sgd_monthly=salary_range.min_monthly_sgd,
            job_salary_max_sgd_monthly=salary_range.max_monthly_sgd,
            job_currency=salary_range.currency,
            
            job_posted_timestamp=job_posted_timestamp,
            
            # Company fields
            company_id=fields.get('company_id', ''),
            company_url=fields.get('company_url', ''),
            company_name=company_name,
            company_description=company_description,
            company_industry=fields.get('company_industry', ''),
            company_size=fields.get('company_size', ''),
        )
        
        return cleaned_job
    
    except Exception as e:
        logger.error(f"Failed to transform job {raw_job.source}:{raw_job.job_id}: {e}", exc_info=True)
        return None
