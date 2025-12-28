"""Shared validation schemas for tool adapters.

Pydantic models used across multiple tools for consistent validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class JobFilters(BaseModel):
    """Common filters for job search tools.
    
    All fields are optional, allowing flexible filtering.
    """
    location: Optional[str] = Field(
        default=None,
        description="Filter by job location (e.g., 'Singapore', 'Central', 'Jurong')",
        examples=["Singapore", "Central", "Woodlands"]
    )
    min_salary: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum monthly salary in SGD",
        examples=[3000, 5000, 8000]
    )
    max_salary: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum monthly salary in SGD",
        examples=[8000, 12000, 20000]
    )
    work_type: Optional[str] = Field(
        default=None,
        description="Employment type (e.g., 'Full Time', 'Part Time', 'Contract')",
        examples=["Full Time", "Part Time", "Contract", "Temporary"]
    )
    classification: Optional[str] = Field(
        default=None,
        description="Job category (e.g., 'Information & Communication Technology')",
        examples=[
            "Information & Communication Technology",
            "Banking & Financial Services",
            "Healthcare & Medical"
        ]
    )


class ToolResponse(BaseModel):
    """Standard response format for all tools.
    
    Ensures consistent error handling and metadata tracking.
    """
    success: bool = Field(description="Whether tool execution succeeded")
    data: Optional[dict] = Field(default=None, description="Tool result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: dict = Field(default_factory=dict, description="Execution metadata")
