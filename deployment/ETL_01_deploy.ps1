#!/usr/bin/env pwsh
# PowerShell script to prepare Cloud Function deployment package
# This creates a clean deployment directory with only necessary files

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cloud Function Deployment Preparation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Remove existing .deploy_temp if it exists
if (Test-Path ".deploy_temp") {
    Write-Host "[1/5] Cleaning up old deployment directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".deploy_temp"
    Write-Host "      ✓ Removed .deploy_temp/" -ForegroundColor Green
} else {
    Write-Host "[1/5] No cleanup needed (first deployment)" -ForegroundColor Gray
}

# Step 2: Create deployment directory
Write-Host "[2/5] Creating deployment directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path ".deploy_temp" | Out-Null
Write-Host "      ✓ Created .deploy_temp/" -ForegroundColor Green

# Step 3: Copy main.py
Write-Host "[3/5] Copying main.py..." -ForegroundColor Yellow
if (Test-Path "etl\main.py") {
    Copy-Item "etl\main.py" ".deploy_temp\main.py"
    Write-Host "      ✓ Copied etl/main.py → .deploy_temp/main.py" -ForegroundColor Green
} else {
    Write-Host "      ✗ ERROR: etl/main.py not found!" -ForegroundColor Red
    exit 1
}

# Step 4: Copy directories
Write-Host "[4/5] Copying directories..." -ForegroundColor Yellow

$directories = @("etl", "utils")
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Copy-Item -Recurse -Force $dir ".deploy_temp\$dir"
        Write-Host "      ✓ Copied $dir/ → .deploy_temp/$dir/" -ForegroundColor Green
    } else {
        Write-Host "      ✗ ERROR: $dir/ not found!" -ForegroundColor Red
        exit 1
    }
}

# Step 5: Copy requirements.txt
Write-Host "[5/5] Copying requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" ".deploy_temp\requirements.txt"
    Write-Host "      ✓ Copied requirements.txt → .deploy_temp/requirements.txt" -ForegroundColor Green
} else {
    Write-Host "      ✗ ERROR: requirements.txt not found!" -ForegroundColor Red
    exit 1
}

# Success summary
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "✓ Deployment package ready in .deploy_temp/" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run the deployment command:" -ForegroundColor White
Write-Host "     gcloud functions deploy etl-gcs-to-bigquery --gen2 --runtime=python313 --region=asia-southeast1 --source=.deploy_temp --entry-point=etl_gcs_to_bigquery --trigger-event-filters=`"type=google.cloud.storage.object.v1.finalized`" --trigger-event-filters=`"bucket=sg-job-market-data`" --memory=512MB --timeout=540s --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com --set-env-vars=`"GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market,GCP_REGION=asia-southeast1,GCS_BUCKET=sg-job-market-data`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. After deployment, cleanup:" -ForegroundColor White
Write-Host "     Remove-Item -Recurse -Force .deploy_temp" -ForegroundColor Gray
Write-Host ""
