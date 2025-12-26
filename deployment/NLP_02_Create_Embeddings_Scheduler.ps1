# Create Cloud Scheduler job for daily embeddings generation
#
# Runs daily at 3:00 AM SGT (after scrapers complete)
# Processes jobs from YESTERDAY (gives buffer time for JobStreet scraper)

param(
    [string]$ProjectId = "sg-job-market",
    [string]$Region = "asia-southeast1",
    [string]$JobName = "scheduler-embeddings-daily-job",
    [string]$Schedule = "0 19 * * *",  # 3:00 AM SGT (19:00 UTC previous day)
    [string]$FunctionName = "generate-daily-embeddings"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Creating Cloud Scheduler Job" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the Cloud Function URL dynamically
Write-Host "Fetching Cloud Function URL..." -ForegroundColor Yellow
$FunctionUrl = gcloud functions describe $FunctionName `
    --region=$Region `
    --project=$ProjectId `
    --format="get(serviceConfig.uri)" 2>$null

if (-not $FunctionUrl) {
    Write-Host "❌ Error: Could not find Cloud Function '$FunctionName'" -ForegroundColor Red
    Write-Host "   Make sure the function is deployed first:" -ForegroundColor Yellow
    Write-Host "   .\deployment\NLP_01_Deploy_Embeddings_CFunc.ps1" -ForegroundColor White
    exit 1
}

Write-Host "✅ Found function URL: $FunctionUrl" -ForegroundColor Green
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Job Name: $JobName" -ForegroundColor Gray
Write-Host "  Schedule: $Schedule ($TimeZone)" -ForegroundColor Gray
Write-Host "  Target: $FunctionUrl" -ForegroundColor Gray
Write-Host ""

# Check if job already exists
Write-Host "Checking if job exists..." -ForegroundColor Yellow
$ExistingJob = gcloud scheduler jobs describe $JobName --location=$Region 2>$null

if ($ExistingJob) {
    Write-Host "⚠️  Job already exists. Updating..." -ForegroundColor Yellow
    
    gcloud scheduler jobs update http $JobName `
        --location=$Region `
        --schedule="$Schedule" `
        --uri="$FunctionUrl" `
        --http-method=POST `
        --message-body='{"process_today": "false"}' `
        --oidc-service-account-email="GCP-general-sa@$ProjectId.iam.gserviceaccount.com" `
        --description="Daily embedding generation for jobs from yesterday"
} else {
    Write-Host "Creating new scheduler job..." -ForegroundColor Yellow
    
    gcloud scheduler jobs create http $JobName `
        --location=$Region `
        --schedule="$Schedule" `
        --uri="$FunctionUrl" `
        --http-method=POST `
        --message-body='{"process_today": "false"}' `
        --oidc-service-account-email="GCP-general-sa@$ProjectId.iam.gserviceaccount.com" `
        --description="Daily embedding generation for jobs from yesterday"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Scheduler Job Created!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule: Every day at 3:00 AM SGT (19:00 UTC previous day)" -ForegroundColor Cyan
    Write-Host "Processes: Jobs from YESTERDAY (buffer time)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Manual trigger commands:" -ForegroundColor Yellow
    Write-Host "  # Trigger yesterday's jobs (default)" -ForegroundColor Gray
    Write-Host "  gcloud scheduler jobs run $JobName --location=$Region" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Trigger today's jobs (manual override)" -ForegroundColor Gray
    Write-Host "  curl -X POST '$FunctionUrl?process_today=true'" -ForegroundColor White
    Write-Host ""
    Write-Host "View logs:" -ForegroundColor Yellow
    Write-Host "  gcloud functions logs read $FunctionName --region=$Region --limit=50" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ Failed to Create Scheduler" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}
