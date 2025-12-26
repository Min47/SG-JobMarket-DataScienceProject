#!/usr/bin/env pwsh
# =============================================================================
# Cloud Scheduler Job Creation for Embeddings Generator
# Creates scheduler to trigger Cloud Run Job (matching scraper pattern)
# =============================================================================
#
# Runs daily at 7:00 AM SGT (after scrapers and ETL complete)
# Triggers: Cloud Run Job cloudjob-embeddings-generator
#
# Schedule Sequence:
#   9:00 PM SGT → JobStreet scraper
#   9:00 AM SGT → MCF scraper
#   ↓ ETL triggers automatically
#   7:00 AM SGT → Embeddings generation (THIS SCHEDULER)

param(
    [string]$ProjectId = "sg-job-market",
    [string]$Region = "asia-southeast1",
    [string]$SchedulerName = "scheduler-embeddings-daily",
    [string]$Schedule = "0 19 * * *",  # 3:00 AM SGT (19:00 UTC previous day)
    [string]$JobName = "cloudjob-embeddings-generator"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Creating Cloud Scheduler for Embeddings" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Step 1: Verify Cloud Run Job exists
# =============================================================================

Write-Host "[1/3] Verifying Cloud Run Job exists..." -ForegroundColor Yellow

$JobExists = gcloud run jobs describe $JobName `
    --region=$Region `
    --project=$ProjectId `
    --format="value(name)" 2>$null

if (-not $JobExists) {
    Write-Host "❌ Error: Cloud Run Job '$JobName' not found" -ForegroundColor Red
    Write-Host "   Deploy the job first:" -ForegroundColor Yellow
    Write-Host "   .\deployment\NLP_01_Deploy_Embeddings_CloudRun.ps1" -ForegroundColor White
    exit 1
}

Write-Host "✅ Found Cloud Run Job: $JobName" -ForegroundColor Green
Write-Host ""

# =============================================================================
# Step 2: Build Cloud Run Job URI
# =============================================================================

Write-Host "[2/3] Building target URI..." -ForegroundColor Yellow

# Cloud Run Job URI format (matching scraper pattern)
$TARGET_URI = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/${JobName}:run"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Scheduler Name: $SchedulerName" -ForegroundColor Gray
Write-Host "  Schedule:       $Schedule" -ForegroundColor Gray
Write-Host "  Target:         $TARGET_URI" -ForegroundColor Gray
Write-Host ""

# =============================================================================
# Step 3: Create or Update Scheduler Job
# =============================================================================

Write-Host "[3/3] Creating/Updating Cloud Scheduler job..." -ForegroundColor Yellow

# Check if scheduler exists
$ExistingScheduler = gcloud scheduler jobs describe $SchedulerName `
    --location=$Region `
    --project=$ProjectId `

$SERVICE_ACCOUNT = "GCP-general-sa@$ProjectId.iam.gserviceaccount.com"

if ($ExistingScheduler) {
    Write-Host "⚠️  Scheduler exists. Updating..." -ForegroundColor Yellow
    
    gcloud scheduler jobs update http $SchedulerName `
        --location=$Region `
        --project=$ProjectId `
        --schedule="$Schedule" `
        --uri="$TARGET_URI" `
        --http-method=POST `
        --oauth-service-account-email=$SERVICE_ACCOUNT `
        --description="Daily embedding generation for jobs (runs after ETL at 3 AM SGT)"
} else {
    Write-Host "Creating new scheduler job..." -ForegroundColor Yellow
    
    gcloud scheduler jobs create http $SchedulerName `
        --location=$Region `
        --project=$ProjectId `
        --schedule="$Schedule" `
        --uri="$TARGET_URI" `
        --http-method=POST `
        --oauth-service-account-email=$SERVICE_ACCOUNT `
        --description="Daily embedding generation for jobs (runs after ETL at 3 AM SGT)"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Cloud Scheduler Created!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule Details:" -ForegroundColor Cyan
    Write-Host "  Daily at: 3:00 AM SGT (19:00 UTC previous day)" -ForegroundColor Gray
    Write-Host "  Target:   Cloud Run Job '$JobName'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Manual Trigger Commands:" -ForegroundColor Yellow
    Write-Host "  # Trigger scheduler manually" -ForegroundColor Gray
    Write-Host "  gcloud scheduler jobs run $SchedulerName --location=$Region" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Execute Cloud Run Job directly" -ForegroundColor Gray
    Write-Host "  gcloud run jobs execute $JobName --region=$Region" -ForegroundColor White
    Write-Host ""
    Write-Host "View Logs:" -ForegroundColor Yellow
    Write-Host "  # Job execution logs" -ForegroundColor Gray
    Write-Host "  gcloud run jobs logs read $JobName --region=$Region --limit=50" -ForegroundColor White
    Write-Host ""
    Write-Host "  # List executions" -ForegroundColor Gray
    Write-Host "  gcloud run jobs executions list --job=$JobName --region=$Region" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ Failed to Create Scheduler" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Verify service account exists and has roles:" -ForegroundColor Gray
    Write-Host "     roles/run.invoker, roles/cloudscheduler.admin" -ForegroundColor White
    Write-Host "  2. Check Cloud Run Job is deployed:" -ForegroundColor Gray
    Write-Host "     gcloud run jobs describe $JobName --region=$Region" -ForegroundColor White
}
