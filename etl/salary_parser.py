"""Salary parsing utilities (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class SalaryRange:
    """Normalized salary range."""

    min_monthly_sgd: Optional[float]
    max_monthly_sgd: Optional[float]
    currency: str = "SGD"


def parse_salary_text(salary_text: str) -> SalaryRange:
    """Parse a free-form salary string into a normalized range (placeholder)."""
    _ = salary_text
    return SalaryRange(min_monthly_sgd=None, max_monthly_sgd=None)

