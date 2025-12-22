"""BigQuery schema migration utilities.

This module provides safe, production-ready tools for:
- Adding columns to existing tables
- Renaming columns
- Updating column descriptions
- Backfilling column data

**Usage:**
    from utils.bq_migrations import add_column, backfill_column
    from utils.config import Settings
    
    settings = Settings.load()
    add_column(
        dataset_id="sg_job_market",
        table_id="cleaned_jobs",
        column_name="job_salary_text_raw",
        column_type="STRING",
        description="Original salary text before parsing",
        settings=settings
    )

All operations are:
- Idempotent (safe to run multiple times)
- Logged with detailed context
- Non-destructive (never delete data)
"""

from __future__ import annotations

import logging
from typing import Optional, List, Callable, Any

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from utils.bq import bq_client
from utils.config import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Column Addition
# =============================================================================

def add_column(
    dataset_id: str,
    table_id: str,
    column_name: str,
    column_type: str,
    description: Optional[str] = None,
    mode: str = "NULLABLE",
    settings: Optional[Settings] = None,
) -> bigquery.Table:
    """Add a new column to an existing BigQuery table.
    
    This operation is idempotent - safe to run multiple times.
    If the column already exists, it will be skipped with a warning.
    
    Args:
        dataset_id: Dataset ID (e.g., "sg_job_market")
        table_id: Table ID (e.g., "cleaned_jobs")
        column_name: Name of new column (e.g., "job_salary_text_raw")
        column_type: BigQuery type: STRING, INTEGER, FLOAT, BOOLEAN, TIMESTAMP, JSON
        description: Optional column description
        mode: NULLABLE (default) or REQUIRED
        settings: Configuration settings
        
    Returns:
        Updated table object
        
    Example:
        >>> add_column(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     column_name="job_salary_text_raw",
        ...     column_type="STRING",
        ...     description="Original salary text before parsing"
        ... )
    """
    if settings is None:
        settings = Settings.load()
    
    client = bq_client(settings)
    table_ref = f"{settings.gcp_project_id}.{dataset_id}.{table_id}"
    
    logger.info(f"[Migration] Adding column '{column_name}' to {table_ref}")
    
    try:
        # Get current table
        table = client.get_table(table_ref)
        
        # Check if column already exists
        existing_fields = {field.name for field in table.schema}
        if column_name in existing_fields:
            logger.warning(f"[Migration] Column '{column_name}' already exists in {table_id}, skipping")
            return table
        
        # Add new column to schema
        new_field = bigquery.SchemaField(
            name=column_name,
            field_type=column_type,
            mode=mode,
            description=description
        )
        
        new_schema = list(table.schema) + [new_field]
        table.schema = new_schema
        
        # Update table
        table = client.update_table(table, ["schema"])
        
        logger.info(f"[Migration] ✓ Column '{column_name}' ({column_type}) added to {table_id}")
        return table
    
    except NotFound:
        logger.error(f"[Migration] ✗ Table not found: {table_ref}")
        raise
    except Exception as e:
        logger.error(f"[Migration] ✗ Failed to add column: {e}", exc_info=True)
        raise


# =============================================================================
# Column Renaming
# =============================================================================

def rename_column(
    dataset_id: str,
    table_id: str,
    old_name: str,
    new_name: str,
    settings: Optional[Settings] = None,
) -> bigquery.Table:
    """Rename a column in an existing BigQuery table.
    
    **Note:** BigQuery doesn't support direct column renaming.
    This creates a copy with the new name, then you must manually drop the old column.
    
    Args:
        dataset_id: Dataset ID
        table_id: Table ID
        old_name: Current column name
        new_name: New column name
        settings: Configuration settings
        
    Returns:
        Updated table object
        
    Example:
        >>> rename_column(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     old_name="job_salary_type",
        ...     new_name="job_salary_text_raw"
        ... )
    """
    if settings is None:
        settings = Settings.load()
    
    client = bq_client(settings)
    table_ref = f"{settings.gcp_project_id}.{dataset_id}.{table_id}"
    
    logger.info(f"[Migration] Renaming column '{old_name}' → '{new_name}' in {table_ref}")
    
    try:
        # Get current table
        table = client.get_table(table_ref)
        
        # Find the column to rename
        old_field = None
        for field in table.schema:
            if field.name == old_name:
                old_field = field
                break
        
        if not old_field:
            logger.error(f"[Migration] ✗ Column '{old_name}' not found in {table_id}")
            raise ValueError(f"Column '{old_name}' does not exist")
        
        # Check if new name already exists
        existing_fields = {field.name for field in table.schema}
        if new_name in existing_fields:
            logger.warning(f"[Migration] Column '{new_name}' already exists in {table_id}, skipping")
            return table
        
        # Create new column with same type and mode
        add_column(
            dataset_id=dataset_id,
            table_id=table_id,
            column_name=new_name,
            column_type=old_field.field_type,
            description=old_field.description,
            mode=old_field.mode,
            settings=settings
        )
        
        # Copy data from old column to new column
        query = f"""
        UPDATE `{table_ref}`
        SET {new_name} = {old_name}
        WHERE {new_name} IS NULL
        """
        
        logger.info(f"[Migration] Copying data from '{old_name}' to '{new_name}'...")
        query_job = client.query(query)
        query_job.result()  # Wait for completion
        
        logger.info(f"[Migration] ✓ Column renamed: '{old_name}' → '{new_name}'")
        logger.warning(f"[Migration] ⚠️  Manual action required: Drop old column '{old_name}' after verifying data")
        
        return client.get_table(table_ref)
    
    except Exception as e:
        logger.error(f"[Migration] ✗ Failed to rename column: {e}", exc_info=True)
        raise


# =============================================================================
# Column Description Updates
# =============================================================================

def update_column_description(
    dataset_id: str,
    table_id: str,
    column_name: str,
    description: str,
    settings: Optional[Settings] = None,
) -> bigquery.Table:
    """Update the description of an existing column.
    
    Args:
        dataset_id: Dataset ID
        table_id: Table ID
        column_name: Column name to update
        description: New description text
        settings: Configuration settings
        
    Returns:
        Updated table object
        
    Example:
        >>> update_column_description(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     column_name="job_salary_min_sgd_monthly",
        ...     description="Minimum monthly salary in SGD (normalized from various periods)"
        ... )
    """
    if settings is None:
        settings = Settings.load()
    
    client = bq_client(settings)
    table_ref = f"{settings.gcp_project_id}.{dataset_id}.{table_id}"
    
    logger.info(f"[Migration] Updating description for '{column_name}' in {table_ref}")
    
    try:
        # Get current table
        table = client.get_table(table_ref)
        
        # Find and update the field
        new_schema = []
        found = False
        
        for field in table.schema:
            if field.name == column_name:
                # Create new field with updated description
                new_field = bigquery.SchemaField(
                    name=field.name,
                    field_type=field.field_type,
                    mode=field.mode,
                    description=description,
                    fields=field.fields  # Preserve nested fields if any
                )
                new_schema.append(new_field)
                found = True
            else:
                new_schema.append(field)
        
        if not found:
            logger.error(f"[Migration] ✗ Column '{column_name}' not found in {table_id}")
            raise ValueError(f"Column '{column_name}' does not exist")
        
        # Update table schema
        table.schema = new_schema
        table = client.update_table(table, ["schema"])
        
        logger.info(f"[Migration] ✓ Description updated for '{column_name}'")
        return table
    
    except Exception as e:
        logger.error(f"[Migration] ✗ Failed to update description: {e}", exc_info=True)
        raise


# =============================================================================
# Data Backfilling
# =============================================================================

def backfill_column(
    dataset_id: str,
    table_id: str,
    column_name: str,
    sql_expression: str,
    where_clause: Optional[str] = None,
    dry_run: bool = False,
    settings: Optional[Settings] = None,
) -> dict:
    """Backfill a column with data using a SQL expression.
    
    Use this after adding a new column to populate it with computed values.
    
    Args:
        dataset_id: Dataset ID
        table_id: Table ID
        column_name: Column to backfill
        sql_expression: SQL expression to compute value (can reference other columns)
        where_clause: Optional WHERE clause to limit which rows to update
        dry_run: If True, only count affected rows without updating
        settings: Configuration settings
        
    Returns:
        Dictionary with statistics: {"affected_rows": int, "dry_run": bool}
        
    Examples:
        >>> # Copy from existing column
        >>> backfill_column(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     column_name="job_salary_text_raw",
        ...     sql_expression="job_salary_type",
        ...     where_clause="job_salary_text_raw IS NULL"
        ... )
        
        >>> # Compute from multiple columns
        >>> backfill_column(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     column_name="job_full_title",
        ...     sql_expression="CONCAT(job_title, ' at ', company_name)",
        ...     where_clause="job_full_title IS NULL"
        ... )
    """
    if settings is None:
        settings = Settings.load()
    
    client = bq_client(settings)
    table_ref = f"{settings.gcp_project_id}.{dataset_id}.{table_id}"
    
    where_part = f"WHERE {where_clause}" if where_clause else ""
    
    if dry_run:
        # Count affected rows
        count_query = f"""
        SELECT COUNT(*) as count
        FROM `{table_ref}`
        {where_part}
        """
        logger.info(f"[Migration] Dry run: counting affected rows for '{column_name}'...")
        query_job = client.query(count_query)
        result = list(query_job.result())[0]
        count = result.count
        
        logger.info(f"[Migration] ✓ Dry run: {count} rows would be updated")
        return {"affected_rows": count, "dry_run": True}
    
    # Execute backfill
    update_query = f"""
    UPDATE `{table_ref}`
    SET {column_name} = {sql_expression}
    {where_part}
    """
    
    logger.info(f"[Migration] Backfilling '{column_name}' with expression: {sql_expression}")
    logger.info(f"[Migration] Query: {update_query}")
    
    try:
        query_job = client.query(update_query)
        result = query_job.result()  # Wait for completion
        
        affected_rows = query_job.num_dml_affected_rows
        logger.info(f"[Migration] ✓ Backfilled '{column_name}': {affected_rows} rows updated")
        
        return {"affected_rows": affected_rows, "dry_run": False}
    
    except Exception as e:
        logger.error(f"[Migration] ✗ Failed to backfill column: {e}", exc_info=True)
        raise


# =============================================================================
# Batch Column Operations
# =============================================================================

def add_multiple_columns(
    dataset_id: str,
    table_id: str,
    columns: List[dict],
    settings: Optional[Settings] = None,
) -> bigquery.Table:
    """Add multiple columns in a single operation.
    
    More efficient than calling add_column() multiple times.
    
    Args:
        dataset_id: Dataset ID
        table_id: Table ID
        columns: List of column specs, each with keys:
            - name: Column name
            - type: BigQuery type
            - description: Optional description
            - mode: Optional mode (default: NULLABLE)
        settings: Configuration settings
        
    Returns:
        Updated table object
        
    Example:
        >>> add_multiple_columns(
        ...     dataset_id="sg_job_market",
        ...     table_id="cleaned_jobs",
        ...     columns=[
        ...         {"name": "job_salary_text_raw", "type": "STRING", "description": "Original salary text"},
        ...         {"name": "job_benefits", "type": "STRING", "description": "Job benefits"},
        ...         {"name": "job_remote_allowed", "type": "BOOLEAN", "description": "Remote work allowed"}
        ...     ]
        ... )
    """
    if settings is None:
        settings = Settings.load()
    
    client = bq_client(settings)
    table_ref = f"{settings.gcp_project_id}.{dataset_id}.{table_id}"
    
    logger.info(f"[Migration] Adding {len(columns)} columns to {table_ref}")
    
    try:
        # Get current table
        table = client.get_table(table_ref)
        existing_fields = {field.name for field in table.schema}
        
        # Build list of new fields
        new_fields = []
        for col_spec in columns:
            col_name = col_spec["name"]
            
            if col_name in existing_fields:
                logger.warning(f"[Migration] Column '{col_name}' already exists, skipping")
                continue
            
            new_field = bigquery.SchemaField(
                name=col_name,
                field_type=col_spec["type"],
                mode=col_spec.get("mode", "NULLABLE"),
                description=col_spec.get("description")
            )
            new_fields.append(new_field)
        
        if not new_fields:
            logger.info("[Migration] No new columns to add")
            return table
        
        # Update schema
        new_schema = list(table.schema) + new_fields
        table.schema = new_schema
        table = client.update_table(table, ["schema"])
        
        logger.info(f"[Migration] ✓ Added {len(new_fields)} columns to {table_id}")
        for field in new_fields:
            logger.info(f"[Migration]   - {field.name} ({field.field_type})")
        
        return table
    
    except Exception as e:
        logger.error(f"[Migration] ✗ Failed to add columns: {e}", exc_info=True)
        raise


# =============================================================================
# CLI Support
# =============================================================================

def _cli_main():
    """Command-line interface for migration utilities."""
    import sys
    from utils.logging import configure_logging
    
    if len(sys.argv) < 2:
        print("Usage: python -m utils.bq_migrations <command> [args...]")
        print("\nCommands:")
        print("  add-column <dataset> <table> <column> <type> [description]")
        print("  update-desc <dataset> <table> <column> <description>")
        print("  backfill <dataset> <table> <column> <expression> [where_clause]")
        print("\nExamples:")
        print("  python -m utils.bq_migrations add-column sg_job_market cleaned_jobs job_salary_text_raw STRING 'Original salary text'")
        print("  python -m utils.bq_migrations backfill sg_job_market cleaned_jobs job_salary_text_raw job_salary_type")
        sys.exit(1)
    
    configure_logging(service_name="bq_migrations")
    settings = Settings.load()
    command = sys.argv[1]
    
    if command == "add-column":
        if len(sys.argv) < 6:
            print("Error: add-column requires: <dataset> <table> <column> <type> [description]")
            sys.exit(1)
        
        add_column(
            dataset_id=sys.argv[2],
            table_id=sys.argv[3],
            column_name=sys.argv[4],
            column_type=sys.argv[5],
            description=sys.argv[6] if len(sys.argv) > 6 else None,
            settings=settings
        )
    
    elif command == "update-desc":
        if len(sys.argv) < 6:
            print("Error: update-desc requires: <dataset> <table> <column> <description>")
            sys.exit(1)
        
        update_column_description(
            dataset_id=sys.argv[2],
            table_id=sys.argv[3],
            column_name=sys.argv[4],
            description=sys.argv[5],
            settings=settings
        )
    
    elif command == "backfill":
        if len(sys.argv) < 6:
            print("Error: backfill requires: <dataset> <table> <column> <expression> [where_clause]")
            sys.exit(1)
        
        backfill_column(
            dataset_id=sys.argv[2],
            table_id=sys.argv[3],
            column_name=sys.argv[4],
            sql_expression=sys.argv[5],
            where_clause=sys.argv[6] if len(sys.argv) > 6 else None,
            settings=settings
        )
    
    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
