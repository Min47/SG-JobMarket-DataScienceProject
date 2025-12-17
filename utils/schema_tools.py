"""Schema management utilities.

Tools for inspecting and validating schema consistency.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Type

from utils.schemas import RawJob, CleanedJob, _dataclass_to_bq_schema


def print_schema_summary():
    """Print current schema definitions for inspection."""
    print("=" * 60)
    print("SCHEMA SUMMARY (Single Source of Truth)")
    print("=" * 60)
    
    for schema_name, schema_class in [("RawJob", RawJob), ("CleanedJob", CleanedJob)]:
        print(f"\n{schema_name}:")
        print("-" * 40)
        
        for field in fields(schema_class):
            mode = "NULLABLE" if "Optional" in str(field.type) or field.default else "REQUIRED"
            print(f"  {field.name:30} {str(field.type):20} [{mode}]")
        
        print(f"\n  BigQuery Schema Fields: {len(_dataclass_to_bq_schema(schema_class))}")
    
    print("\n" + "=" * 60)
    print("To modify: Edit utils/schemas.py dataclasses only")
    print("=" * 60)


def compare_schemas():
    """Compare Python dataclass fields with BigQuery schema output."""
    print("\n🔍 Schema Consistency Check\n")
    
    for schema_name, schema_class in [("RawJob", RawJob), ("CleanedJob", CleanedJob)]:
        dc_fields = [f.name for f in fields(schema_class)]
        bq_fields = [f.name for f in _dataclass_to_bq_schema(schema_class)]
        
        if dc_fields == bq_fields:
            print(f"✅ {schema_name}: Dataclass ↔ BigQuery schemas aligned ({len(dc_fields)} fields)")
        else:
            print(f"❌ {schema_name}: MISMATCH!")
            print(f"   Dataclass: {dc_fields}")
            print(f"   BigQuery:  {bq_fields}")


if __name__ == "__main__":
    print_schema_summary()
    compare_schemas()
