"""Validation utilities for scrapers.

Helper functions to validate scraper output and JSONL files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from utils.schemas import RawJob


def validate_jsonl_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate a JSONL file produced by scrapers.
    
    Returns:
        (is_valid, errors): Tuple of validation status and list of error messages
    """
    errors = []
    
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    
    if file_path.stat().st_size == 0:
        return False, ["File is empty"]
    
    line_count = 0
    valid_jobs = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                
                if not line.strip():
                    errors.append(f"Line {line_num}: Empty line")
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Validate RawJob schema (payload-based)
                    required_fields = ["job_id", "source", "scrape_timestamp", "payload"]
                    missing = [f for f in required_fields if f not in data]
                    
                    if missing:
                        errors.append(f"Line {line_num}: Missing RawJob fields: {missing}")
                        continue
                    
                    # Validate payload has essential job data
                    payload = data.get("payload", {})
                    if not isinstance(payload, dict):
                        errors.append(f"Line {line_num}: Payload must be a dictionary")
                        continue
                    
                    payload_required = ["title", "company", "url"]
                    missing_payload = [f for f in payload_required if not payload.get(f)]
                    
                    if missing_payload:
                        errors.append(f"Line {line_num}: Missing payload fields: {missing_payload}")
                    else:
                        valid_jobs += 1
                        
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    
    except Exception as e:
        return False, [f"Failed to read file: {e}"]
    
    # Summary
    if line_count == 0:
        return False, ["No lines found in file"]
    
    if valid_jobs == 0:
        return False, errors + ["No valid jobs found"]
    
    if errors:
        return False, errors[:10]  # Return first 10 errors
    
    return True, [f"✓ Valid: {valid_jobs}/{line_count} jobs"]


def validate_raw_job(job: RawJob) -> Tuple[bool, str]:
    """Validate a single RawJob instance.
    
    Returns:
        (is_valid, message): Validation status and message
    """
    if not job.job_id or len(job.job_id.strip()) < 2:
        return False, "Job ID missing or too short"
    
    if job.source not in ["JobStreet", "MyCareersFuture"]:
        return False, f"Invalid source: {job.source}"
    
    if not job.scrape_timestamp:
        return False, "Scrape timestamp missing"
    
    if not isinstance(job.payload, dict):
        return False, "Payload must be a dictionary"
    
    # Validate essential payload fields
    title = job.payload.get("title", "")
    company = job.payload.get("company", "")
    url = job.payload.get("url", "")
    
    if not title or len(title.strip()) < 2:
        return False, "Title in payload too short or empty"
    
    if not company or len(company.strip()) < 2:
        return False, "Company in payload too short or empty"
    
    if not url or not url.startswith("http"):
        return False, "Invalid or missing URL in payload"
    
    return True, "Valid"
