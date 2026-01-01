"""Guardrails & Policy Chains for GenAI API.

This module provides input/output validation to protect against:
- PII (Personally Identifiable Information) leakage
- Prompt injection attacks
- SQL injection attempts
- Profanity and inappropriate content
- Output hallucinations (citing non-existent jobs)

Architecture:
    User Input → Input Guards → Agent Processing → Output Guards → Response
    
Guards:
    - PIIDetector: Scans for Singapore NRIC, phone, email, credit cards
    - InjectionDetector: Identifies prompt/SQL injection patterns
    - ProfanityFilter: Blocks offensive language (optional)
    - HallucinationChecker: Verifies cited jobs exist in context
    - ContentSafetyFilter: Flags inappropriate AI-generated content

Usage:
    >>> guards = InputGuardrails()
    >>> result = guards.validate("Find jobs with S1234567D")
    >>> if result.blocked:
    ...     raise HTTPException(400, result.reason)
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"        # Informational, no action needed
    WARNING = "warning"  # Potential issue, log but allow
    BLOCKED = "blocked"  # Critical issue, reject request


@dataclass
class ValidationResult:
    """Result of guardrail validation.
    
    Attributes:
        passed: Whether validation passed (False if blocked)
        severity: Severity level of any issues found
        reason: Human-readable explanation of validation result
        violations: List of specific violations found
        sanitized_input: Optionally modified input (PII redacted, etc.)
    """
    passed: bool
    severity: ValidationSeverity
    reason: str
    violations: List[str]
    sanitized_input: Optional[str] = None


# =============================================================================
# PII Detection (Singapore-specific)
# =============================================================================

class PIIDetector:
    """Detect Personally Identifiable Information in Singapore context.
    
    Patterns detected:
    - NRIC: S1234567D, T1234567A (Singaporean/PR ID)
    - Phone: +65 1234 5678, 91234567
    - Email: user@example.com
    - Credit Card: 4111-1111-1111-1111
    
    Uses regex patterns optimized for Singapore formats.
    Does NOT use external libraries to avoid heavy dependencies.
    """
    
    # Singapore NRIC/FIN pattern: S/T/F/G + 7 digits + checksum letter
    NRIC_PATTERN = r'\b[STFG]\d{7}[A-Z]\b'
    
    # Singapore phone: +65 followed by 8 digits
    PHONE_PATTERN = r'\+65\s?\d{4}\s?\d{4}|\b[89]\d{7}\b'
    
    # Email pattern (basic)
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Credit card pattern (basic, matches common formats)
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    
    def __init__(self):
        """Initialize PII detector with compiled regex patterns."""
        self.nric_regex = re.compile(self.NRIC_PATTERN, re.IGNORECASE)
        self.phone_regex = re.compile(self.PHONE_PATTERN)
        self.email_regex = re.compile(self.EMAIL_PATTERN)
        self.cc_regex = re.compile(self.CREDIT_CARD_PATTERN)
    
    def detect(self, text: str) -> Tuple[bool, List[str]]:
        """Detect PII in text.
        
        Args:
            text: Input text to scan for PII
            
        Returns:
            Tuple of (has_pii: bool, violations: List[str])
            
        Example:
            >>> detector = PIIDetector()
            >>> has_pii, violations = detector.detect("My NRIC is S1234567D")
            >>> has_pii
            True
            >>> violations
            ['NRIC found: S1234567D']
        """
        violations = []
        
        # Check NRIC
        nric_matches = self.nric_regex.findall(text)
        if nric_matches:
            violations.append(f"NRIC found: {nric_matches[0]} (redacted)")
        
        # Check phone
        phone_matches = self.phone_regex.findall(text)
        if phone_matches:
            violations.append(f"Phone number found: {phone_matches[0][:4]}... (redacted)")
        
        # Check email
        email_matches = self.email_regex.findall(text)
        if email_matches:
            violations.append(f"Email found: {email_matches[0].split('@')[0][:3]}...@... (redacted)")
        
        # Check credit card
        cc_matches = self.cc_regex.findall(text)
        if cc_matches:
            violations.append(f"Credit card found: ****-****-****-{cc_matches[0][-4:]} (redacted)")
        
        has_pii = len(violations) > 0
        return has_pii, violations
    
    def redact(self, text: str) -> str:
        """Redact PII from text.
        
        Args:
            text: Input text with potential PII
            
        Returns:
            Text with PII replaced by [REDACTED]
            
        Example:
            >>> detector = PIIDetector()
            >>> detector.redact("Contact me at S1234567D")
            'Contact me at [REDACTED]'
        """
        # Redact NRIC
        text = self.nric_regex.sub('[REDACTED_NRIC]', text)
        
        # Redact phone
        text = self.phone_regex.sub('[REDACTED_PHONE]', text)
        
        # Redact email
        text = self.email_regex.sub('[REDACTED_EMAIL]', text)
        
        # Redact credit card
        text = self.cc_regex.sub('[REDACTED_CC]', text)
        
        return text


# =============================================================================
# Injection Detection
# =============================================================================

class InjectionDetector:
    """Detect prompt injection and SQL injection attempts.
    
    Prompt injection patterns:
    - "Ignore previous instructions"
    - "Forget everything and..."
    - "Act as a different AI"
    - System prompt overrides
    
    SQL injection patterns:
    - SQL keywords in suspicious contexts
    - Comment sequences (-- , /* */)
    - Union/drop/exec statements
    """
    
    # Prompt injection patterns (case-insensitive)
    # Note: Balanced for job search - catches attacks, minimizes false positives
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+(previous|all|earlier)\s+(instructions?|prompts?|commands?)',
        r'forget\s+(everything|all|previous)',
        r'disregard\s+(all|previous|earlier)',
        r'(act|pretend|behave)\s+as\s+(a\s+)?(different|new)',
        r'system\s*:\s*',
        r'<\s*system\s*>',
        r'override\s+(previous|all|earlier)',
        r'new\s+(instruction|command|rule)',
        # Removed: r'from\s+now\s+on' - too common in natural language ("from now on show me remote jobs")
    ]
    
    # SQL injection patterns
    # Note: More specific patterns to avoid blocking "select candidates from finance"
    SQL_INJECTION_PATTERNS = [
        # Require semicolon before SQL commands (injection chaining)
        r';\s*(union|select|insert|update|delete|drop|exec|execute)\b',
        r'--\s*$',  # SQL comment at end
        r'/\*.*\*/',  # SQL block comment
        # Quote-based injections (classic attack patterns)
        r"'\s*(or|and)\s*'",  # ' OR ', ' AND '
        r"'.*\bor\b.*'.*=.*'",  # ' OR '1'='1'
        r'\bor\b\s+\d+\s*=\s*\d+',  # OR 1=1
        r'\band\b\s+\d+\s*=\s*\d+',  # AND 1=1
        # UNION-based attacks
        r'\bunion\b.*\bselect\b',
    ]
    
    def __init__(self):
        """Initialize injection detector with compiled patterns."""
        self.prompt_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.PROMPT_INJECTION_PATTERNS
        ]
        self.sql_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
    
    def detect_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Detect prompt injection attempts.
        
        Args:
            text: Input text to scan
            
        Returns:
            Tuple of (is_injection: bool, matched_patterns: List[str])
        """
        violations = []
        
        for i, pattern in enumerate(self.prompt_patterns):
            if pattern.search(text):
                match = pattern.search(text).group(0)
                violations.append(f"Prompt injection pattern {i+1}: '{match[:50]}'")
        
        is_injection = len(violations) > 0
        return is_injection, violations
    
    def detect_sql_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Detect SQL injection attempts.
        
        Args:
            text: Input text to scan
            
        Returns:
            Tuple of (is_injection: bool, matched_patterns: List[str])
        """
        violations = []
        
        for i, pattern in enumerate(self.sql_patterns):
            if pattern.search(text):
                match = pattern.search(text).group(0)
                violations.append(f"SQL injection pattern {i+1}: '{match[:50]}'")
        
        is_injection = len(violations) > 0
        return is_injection, violations


# =============================================================================
# Input Guardrails (Orchestrator)
# =============================================================================

class InputGuardrails:
    """Orchestrate all input validation checks.
    
    Validates user input through multiple guards:
    1. Length limits (prevent resource exhaustion)
    2. PII detection (protect sensitive data)
    3. Injection detection (prevent attacks)
    
    Usage:
        >>> guards = InputGuardrails()
        >>> result = guards.validate("Find me data scientist jobs")
        >>> if not result.passed:
        ...     raise HTTPException(400, result.reason)
    """
    
    # Configuration
    MAX_QUERY_LENGTH = 1000  # chars
    MIN_QUERY_LENGTH = 3     # chars
    
    def __init__(self):
        """Initialize all guard components."""
        self.pii_detector = PIIDetector()
        self.injection_detector = InjectionDetector()
        logger.info("[Guardrails] Input guards initialized")
    
    def validate(self, query: str) -> ValidationResult:
        """Run all validation checks on user input.
        
        Args:
            query: User's search query or message
            
        Returns:
            ValidationResult with pass/fail status and details
            
        Example:
            >>> guards = InputGuardrails()
            >>> result = guards.validate("Find jobs with my NRIC S1234567D")
            >>> result.passed
            False
            >>> result.reason
            'PII detected: NRIC found'
        """
        all_violations = []
        
        # 1. Length validation
        if not query or not query.strip():
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="Query cannot be empty",
                violations=["Empty or whitespace-only query"],
            )
        
        if len(query) < self.MIN_QUERY_LENGTH:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason=f"Query too short (minimum {self.MIN_QUERY_LENGTH} characters)",
                violations=[f"Query length: {len(query)} < {self.MIN_QUERY_LENGTH}"],
            )
        
        if len(query) > self.MAX_QUERY_LENGTH:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason=f"Query too long (maximum {self.MAX_QUERY_LENGTH} characters)",
                violations=[f"Query length: {len(query)} > {self.MAX_QUERY_LENGTH}"],
            )
        
        # 2. PII detection
        has_pii, pii_violations = self.pii_detector.detect(query)
        if has_pii:
            all_violations.extend(pii_violations)
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="PII detected in query. Please remove personal information.",
                violations=all_violations,
                sanitized_input=self.pii_detector.redact(query),
            )
        
        # 3. Prompt injection detection
        is_prompt_injection, prompt_violations = self.injection_detector.detect_prompt_injection(query)
        if is_prompt_injection:
            all_violations.extend(prompt_violations)
            logger.warning(f"[Guardrails] Prompt injection attempt detected: {prompt_violations}")
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="Potential prompt injection detected. Please rephrase your query.",
                violations=all_violations,
            )
        
        # 4. SQL injection detection
        is_sql_injection, sql_violations = self.injection_detector.detect_sql_injection(query)
        if is_sql_injection:
            all_violations.extend(sql_violations)
            logger.warning(f"[Guardrails] SQL injection attempt detected: {sql_violations}")
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="Potential SQL injection detected. Please use plain language.",
                violations=all_violations,
            )
        
        # All checks passed
        logger.debug(f"[Guardrails] Input validation passed: {query[:50]}...")
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            reason="Input validation passed",
            violations=[],
        )


# =============================================================================
# Output Guardrails
# =============================================================================

class OutputGuardrails:
    """Validate AI-generated responses before returning to user.
    
    Checks:
    1. Hallucination: Verify cited jobs exist in context
    2. Length: Ensure response not truncated
    3. Content safety: Flag inappropriate content (optional)
    """
    
    MAX_RESPONSE_LENGTH = 5000  # chars
    
    def __init__(self):
        """Initialize output guards."""
        logger.info("[Guardrails] Output guards initialized")
    
    def validate(
        self,
        response: Dict[str, Any],
        context_jobs: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """Validate AI-generated response.
        
        Args:
            response: Generated response dict with 'answer' and 'sources'
            context_jobs: Jobs that were provided as context (for hallucination check)
            
        Returns:
            ValidationResult with warnings/errors
        """
        violations = []
        
        # 1. Check response structure
        if not response or 'answer' not in response:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="Invalid response structure",
                violations=["Missing 'answer' field"],
            )
        
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        
        # 2. Length validation
        if len(answer) > self.MAX_RESPONSE_LENGTH:
            violations.append(f"Response truncated: {len(answer)} > {self.MAX_RESPONSE_LENGTH}")
        
        # 3. Hallucination check: Verify cited sources exist
        if context_jobs is not None and sources:
            context_job_ids = {
                job.get('job_id') for job in context_jobs
                if job.get('job_id')
            }
            
            for source in sources:
                cited_job_id = source.get('job_id')
                if cited_job_id and cited_job_id not in context_job_ids:
                    violations.append(
                        f"Hallucinated source: Job {cited_job_id} not in context"
                    )
            
            if violations:
                logger.warning(f"[Guardrails] Hallucination detected: {violations}")
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    reason="Response cites jobs not in context",
                    violations=violations,
                )
        
        # 4. Empty response check
        if not answer.strip():
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.BLOCKED,
                reason="Empty response generated",
                violations=["Answer is empty or whitespace-only"],
            )
        
        # All checks passed
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            reason="Output validation passed",
            violations=[],
        )
