"""Text cleaning and normalization utilities for ETL pipeline.

This module provides functions for:
- HTML tag removal from job descriptions
- Whitespace normalization
- Unicode cleaning
- Company name standardization
- Location normalization
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional
from langdetect import detect, LangDetectException


def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities from text.
    
    Handles:
    - HTML tags: <p>, <div>, <br>, etc.
    - HTML entities: &nbsp;, &amp;, etc.
    - Script/style blocks: Remove entirely
    
    Args:
        text: HTML text to clean
        
    Returns:
        Plain text with HTML removed
        
    Examples:
        >>> clean_html("<p>Hello <b>World</b></p>")
        'Hello World'
        >>> clean_html("Price: &pound;3000")
        'Price: £3000'
    """
    if not text:
        return ""
    
    # Remove script and style blocks entirely (including content)
    # Example: <script>...</script>, <style>...</style>
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace common block tags with newlines (to preserve paragraph structure)
    # Example: <p>, <div>, <br>, <li>, <tr>, <h1>-<h6>
    text = re.sub(r'</?(p|div|br|li|tr|h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    
    # Remove all other HTML tags
    # Example: <b>, <i>, <span>, <a>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (&nbsp; → space, &amp; → &, etc.)
    text = html.unescape(text)
    
    return text

def clean_unicode(text: str) -> str:
    """Clean and normalize unicode characters.
    
    Handles:
    - Normalize to NFC form (canonical composition)
    - Remove control characters (except newline, tab)
    - Replace non-breaking spaces with regular spaces
    - Remove zero-width characters
    
    Args:
        text: Text to clean
        
    Returns:
        Text with normalized unicode
        
    Examples:
        >>> clean_unicode("Hello\u00a0World\u200b!")
        'Hello World!'
        >>> clean_unicode("Control\u0007Char")
        'ControlChar'
    """
    if not text:
        return ""
    
    # Normalize unicode to NFC (canonical composition)
    # E.g., é (U+00E9) instead of e + ́ (U+0065 U+0301)
    text = unicodedata.normalize('NFC', text)
    
    # Replace non-breaking spaces with regular spaces
    # U+00A0, U+202F
    text = text.replace('\u00a0', ' ').replace('\u202f', ' ')
    
    # Remove zero-width characters
    # U+200B (zero-width space), U+200C (zero-width non-joiner), U+200D (zero-width joiner)
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    # Remove other control characters (except newline \n and tab \t)
    # Keep characters with Unicode category starting with 'C' only if they are \n or \t
    # E.g., remove U+0000 to U+001F except \n (U+000A) and \t (U+0009)
    # E.g., remove U+007F (DEL)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    return text

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Handles:
    - Multiple spaces → single space
    - Multiple newlines → single newline
    - Trim leading/trailing whitespace
    - Remove \r, \t, and other control characters
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized whitespace
        
    Examples:
        >>> normalize_whitespace("Hello   World\\n\\n\\nTest")
        'Hello World\\nTest'
    """
    if not text:
        return ""
    
    # Replace tabs and carriage returns with spaces
    text = text.replace('\t', ' ').replace('\r', '')
    
    # Collapse multiple spaces into one
    # E.g., "Hello    World" → "Hello World"
    text = re.sub(r' +', ' ', text)
    
    # Collapse multiple newlines into one
    # E.g., "\n\n\n" → "\n"
    text = re.sub(r'\n\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_company_name(name: str) -> str:
    """Standardize company names for better matching.
    
    Handles:
    - Strip leading/trailing whitespace
    - Normalize case to title case
    - Remove common suffixes: Pte Ltd, Private Limited, Inc, Corp
    - Collapse multiple spaces
    
    Args:
        name: Company name to normalize
        
    Returns:
        Normalized company name
        
    Examples:
        >>> normalize_company_name("ACME CORP PTE LTD")
        'Acme Corp'
        >>> normalize_company_name("  Tech   Start-up  Private Limited  ")
        'Tech Start-Up'
    """
    if not name:
        return ""
    
    # Strip and normalize whitespace
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    
    # Remove common suffixes (case-insensitive)
    # Order matters: try longer patterns first
    suffixes = [
        r'\bPrivate Limited\b',
        r'\bPte\.?\s*Ltd\.?\b',
        r'\bLimited\b',
        r'\bLtd\.?\b',
        r'\bIncorporated\b',
        r'\bInc\.?\b',
        r'\bCorporation\b',
        r'\bCorp\.?\b',
        r'\bLLC\b',
        r'\bL\.L\.C\.\b',
    ]
    
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Strip again after suffix removal
    name = name.strip()
    
    # Normalize case to title case
    # But preserve acronyms (all caps words)
    # E.g., "IBM" stays "IBM", "dBS" becomes "Dbs"
    words = name.split()
    normalized_words = []
    for word in words:
        if word.isupper() and len(word) > 1:
            # Keep acronyms as-is (e.g., IBM, DBS)
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())
    
    return ' '.join(normalized_words)


def normalize_location(location: str) -> str:
    """Standardize location strings.
    
    Handles:
    - Strip whitespace
    - Normalize to title case
    - Map common variations to standard names
    
    Args:
        location: Location string to normalize
        
    Returns:
        Normalized location
        
    Examples:
        >>> normalize_location("central region")
        'Central'
        >>> normalize_location("WEST COAST")
        'West'
    """
    if not location:
        return ""
    
    # Strip and normalize whitespace
    location = location.strip().lower()
    
    # Map common variations to standard Singapore regions
    # Singapore has 5 regions: Central, East, North, North-East, West
    location_map = {
        'central': 'Central',
        'central region': 'Central',
        'central area': 'Central',
        'cbd': 'Central',
        'city': 'Central',
        'downtown': 'Central',
        
        'east': 'East',
        'east region': 'East',
        'eastern': 'East',
        
        'west': 'West',
        'west region': 'West',
        'western': 'West',
        
        'north': 'North',
        'north region': 'North',
        'northern': 'North',
        
        'north-east': 'North-East',
        'northeast': 'North-East',
        'north east': 'North-East',
        'north-eastern': 'North-East',
        
        'singapore': 'Singapore',
        'sg': 'Singapore',
    }
    
    # Check if location matches a known mapping
    normalized = location_map.get(location)
    if normalized:
        return normalized
    
    # If no match, return title case
    return location.title()


def clean_description(text: str) -> str:
    """Full cleaning pipeline for job descriptions.
    
    Combines all cleaning steps:
    1. Remove HTML tags
    2. Clean unicode
    3. Normalize whitespace
    
    Args:
        text: Raw job description
        
    Returns:
        Cleaned description
    """
    text = clean_html(text)
    text = clean_unicode(text)
    text = normalize_whitespace(text)
    return text


# =============================================================================
# Optional: Language Detection (requires langdetect)
# =============================================================================

def detect_language(text: str) -> str:
    """Detect language of text.
    
    - Returns ISO 639-1 code (en, zh, ms, ta, etc.)
    
    Args:
        text: Text to analyze
        
    Returns:
        ISO 639-1 language code or 'unknown'
        
    Examples:
        >>> detect_language("This is a job description")
        'en'
        >>> detect_language("这是一个职位描述")
        'zh'
    """

    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'
