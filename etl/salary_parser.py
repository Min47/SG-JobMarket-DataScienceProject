"""Salary parsing utilities for extracting and normalizing salary information.

This module parses salary strings from job postings and normalizes them to:
- Extract min/max salary values
- Identify salary period (hourly, daily, monthly, yearly)
- Convert all salaries to monthly SGD equivalent

**REVIEW THIS FILE**: Salary extraction and normalization logic.

Handles patterns like:
- "$3000 - $5000 per month"
- "3k-5k monthly"
- "$3000 to $5000"
- "$5000/month"
- "$60000 per year"
- "$20/hour"
- "Competitive" / "Negotiable"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class SalaryRange:
    """Normalized salary range with both raw and monthly values."""

    min_raw_sgd: Optional[float]  # Raw value from posting
    max_raw_sgd: Optional[float]  # Raw value from posting
    salary_period: str  # hourly, daily, monthly, yearly
    min_monthly_sgd: Optional[float]  # Converted to monthly
    max_monthly_sgd: Optional[float]  # Converted to monthly
    currency: str = "SGD"


# =============================================================================
# Conversion Rates to Monthly
# =============================================================================

HOURLY_TO_MONTHLY = 160  # 40 hrs/week × 4 weeks
DAILY_TO_MONTHLY = 22    # Working days per month
YEARLY_TO_MONTHLY = 1 / 12


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text.
    
    Handles:
    - Comma separators: 3,000 → 3000
    - K suffix: 3k → 3000
    - Decimal: 3.5k → 3500
    
    Args:
        text: Text containing numbers
        
    Returns:
        List of extracted numbers
        
    Examples:
        >>> extract_numbers("$3,000 to $5,000")
        [3000.0, 5000.0]
        >>> extract_numbers("3k-5k")
        [3000.0, 5000.0]
    """
    if not text:
        return []
    
    numbers = []
    
    # Pattern: optional dollar sign, number with optional commas and decimals, optional 'k'
    # Examples: $3,000, 3000, 3.5k, $5k
    # Example: $3,500.75k → 3500750.0
    pattern = r'\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?' 
    # Regex explanation:
    # \$? - optional dollar sign
    # \s* - optional whitespace
    # (\d+(?:,\d{3})*(?:\.\d+)?) - number with optional commas and decimal
    # \s* - optional whitespace
    # k? - optional 'k' suffix
    
    for match in re.finditer(pattern, text.lower()):
        num_str = match.group(1).replace(',', '')  # Remove commas
        num = float(num_str)
        
        # Check if 'k' suffix follows
        if 'k' in text[match.start():match.end()].lower():
            num *= 1000
        
        numbers.append(num)
    
    return numbers


def identify_period(text: str) -> str:
    """Identify salary period from text.
    
    Returns one of: 'hourly', 'daily', 'monthly', 'yearly', 'unknown'
    
    Args:
        text: Salary text to analyze
        
    Returns:
        Period identifier
        
    Examples:
        >>> identify_period("$3000 per month")
        'monthly'
        >>> identify_period("$20/hour")
        'hourly'
    """
    if not text:
        return 'unknown'
    
    text_lower = text.lower()
    
    # Check for period indicators
    if any(word in text_lower for word in ['hour', 'hourly', '/hr', 'per hour']):
        return 'hourly'
    
    if any(word in text_lower for word in ['day', 'daily', '/day', 'per day']):
        return 'daily'
    
    if any(word in text_lower for word in ['year', 'annual', 'yearly', '/year', 'per year', 'per annum', 'p.a.', 'pa']):
        return 'yearly'
    
    if any(word in text_lower for word in ['month', 'monthly', '/month', 'per month', '/mth', 'mth']):
        return 'monthly'
    
    # Default to monthly if no period specified (most common)
    return 'monthly'


def convert_to_monthly(value: float, period: str) -> float:
    """Convert salary to monthly equivalent.
    
    Args:
        value: Salary value
        period: Period identifier (hourly, daily, monthly, yearly)
        
    Returns:
        Monthly equivalent
        
    Examples:
        >>> convert_to_monthly(3000, 'monthly')
        3000.0
        >>> convert_to_monthly(60000, 'yearly')
        5000.0
        >>> convert_to_monthly(20, 'hourly')
        3200.0
    """
    if period == 'hourly':
        return value * HOURLY_TO_MONTHLY
    elif period == 'daily':
        return value * DAILY_TO_MONTHLY
    elif period == 'yearly':
        return value * YEARLY_TO_MONTHLY
    else:  # monthly or unknown
        return value


def parse_salary_range(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """Parse salary range from text with enhanced edge case handling.
    
    Handles special patterns:
    - "up to $5000" → (None, 5000) - cap only
    - "from $3000" → (3000, None) - floor only
    - "$3000 - $5000" → (3000, 5000) - range
    - "$5000" → (5000, 5000) - single value
    
    Args:
        text: Salary text to parse
        
    Returns:
        Tuple of (min_value, max_value, period)
        
    Examples:
        >>> parse_salary_range("$3000 - $5000 per month")
        (3000.0, 5000.0, 'monthly')
        >>> parse_salary_range("up to $5000 monthly")
        (None, 5000.0, 'monthly')
        >>> parse_salary_range("from $3000")
        (3000.0, None, 'monthly')
        >>> parse_salary_range("$60000 per year")
        (60000.0, 60000.0, 'yearly')
        >>> parse_salary_range("Competitive")
        (None, None, 'unknown')
    """
    if not text or not text.strip():
        return (None, None, 'unknown')
    
    text_lower = text.lower()
    
    # Check for non-numeric salary indicators
    non_numeric_keywords = ['competitive', 'negotiable', 'attractive', 'commensurate', 'to be discussed', 'tbd']
    if any(keyword in text_lower for keyword in non_numeric_keywords):
        return (None, None, 'unknown')
    
    # Extract numbers
    numbers = extract_numbers(text)
    
    if not numbers:
        return (None, None, 'unknown')
    
    # Identify period
    period = identify_period(text)
    
    # Check for "up to" pattern (cap only)
    if re.search(r'\bup to\b', text_lower):
        # "up to $5000" → min=None, max=5000
        max_value = numbers[-1]  # Last number is the cap
        return (None, max_value, period)
    
    # Check for "from" pattern (floor only)
    if re.search(r'\bfrom\b', text_lower) and len(numbers) == 1:
        # "from $3000" → min=3000, max=None
        min_value = numbers[0]
        return (min_value, None, period)
    
    # If only one number (no special keywords), use it as both min and max
    if len(numbers) == 1:
        return (numbers[0], numbers[0], period)
    
    # If multiple numbers, use first as min, last as max
    # (handles "3000 to 5000" and "3000-5000")
    return (numbers[0], numbers[-1], period)


def parse_salary_text(salary_text: str) -> SalaryRange:
    """Parse a free-form salary string into a normalized monthly range.
    
    This is the main entry point for salary parsing.
    
    Args:
        salary_text: Raw salary string from job posting
        
    Returns:
        SalaryRange with both raw and monthly SGD values
        
    Examples:
        >>> parse_salary_text("$3000 - $5000 per month")
        SalaryRange(min_raw_sgd=3000.0, max_raw_sgd=5000.0, salary_period='monthly', 
                    min_monthly_sgd=3000.0, max_monthly_sgd=5000.0, currency='SGD')
        >>> parse_salary_text("$60000 per year")
        SalaryRange(min_raw_sgd=60000.0, max_raw_sgd=60000.0, salary_period='yearly',
                    min_monthly_sgd=5000.0, max_monthly_sgd=5000.0, currency='SGD')
        >>> parse_salary_text("Competitive")
        SalaryRange(min_raw_sgd=None, max_raw_sgd=None, salary_period='',
                    min_monthly_sgd=None, max_monthly_sgd=None, currency='SGD')
    """
    min_raw, max_raw, period = parse_salary_range(salary_text)
    
    if min_raw is None or max_raw is None:
        return SalaryRange(
            min_raw_sgd=None,
            max_raw_sgd=None,
            salary_period='',
            min_monthly_sgd=None,
            max_monthly_sgd=None
        )
    
    # Convert to monthly
    min_monthly = convert_to_monthly(min_raw, period)
    max_monthly = convert_to_monthly(max_raw, period)
    
    return SalaryRange(
        min_raw_sgd=min_raw,
        max_raw_sgd=max_raw,
        salary_period=period,
        min_monthly_sgd=min_monthly,
        max_monthly_sgd=max_monthly
    )

