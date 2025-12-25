# Deploy Embeddings Generation Cloud Function
# 
# This script packages the NLP module and deploys to Cloud Functions Gen 2
# Similar to ETL deployment, it creates a temporary directory with all dependencies

param(
    [string]$ProjectId = "sg-job-market",
    [string]$Region = "asia-southeast1",
    [string]$FunctionName = "generate-daily-embeddings",
    [int]$Memory = 2048,  # 2GB for sentence-transformers
    [int]$Timeout = 540    # 9 minutes max for Cloud Functions
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deploying Embeddings Cloud Function" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Create temporary deployment directory
$DeployTemp = ".deploy_temp_embeddings"
Write-Host "Creating deployment package..." -ForegroundColor Yellow

if (Test-Path $DeployTemp) {
    Remove-Item -Recurse -Force $DeployTemp
}
New-Item -ItemType Directory -Path $DeployTemp | Out-Null

# Copy NLP files
Write-Host "  Copying nlp/ module..." -ForegroundColor Gray
Copy-Item -Path "nlp/cloud_function_main.py" -Destination "$DeployTemp/main.py"  # Rename for Cloud Functions
Copy-Item -Path "nlp/embeddings.py" -Destination "$DeployTemp/"
Copy-Item -Path "nlp/generate_embeddings.py" -Destination "$DeployTemp/"

# Copy root requirements.txt
Write-Host "  Copying requirements.txt..." -ForegroundColor Gray
Copy-Item -Path "requirements.txt" -Destination "$DeployTemp/requirements.txt"

# Copy utils (needed by generate_embeddings.py)
Write-Host "  Copying utils/ dependencies..." -ForegroundColor Gray
New-Item -ItemType Directory -Path "$DeployTemp/utils" | Out-Null
Copy-Item -Path "utils/__init__.py" -Destination "$DeployTemp/utils/"
Copy-Item -Path "utils/logging.py" -Destination "$DeployTemp/utils/"
Copy-Item -Path "utils/config.py" -Destination "$DeployTemp/utils/"
Copy-Item -Path "utils/schemas.py" -Destination "$DeployTemp/utils/"

# Create .gcloudignore
Write-Host "  Creating .gcloudignore..." -ForegroundColor Gray
@"
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.venv/
venv/
.pytest_cache/
.mypy_cache/
*.log
"@ | Out-File -FilePath "$DeployTemp/.gcloudignore" -Encoding utf8

Write-Host "✅ Deployment package ready" -ForegroundColor Green
Write-Host ""

# Deploy to Cloud Functions
Write-Host "Deploying to Cloud Functions..." -ForegroundColor Yellow
Write-Host "  Project: $ProjectId" -ForegroundColor Gray
Write-Host "  Region: $Region" -ForegroundColor Gray
Write-Host "  Memory: $Memory MB" -ForegroundColor Gray
Write-Host "  Timeout: $Timeout seconds" -ForegroundColor Gray
Write-Host ""

gcloud functions deploy $FunctionName `
    --gen2 `
    --runtime=python313 `
    --region=$Region `
    --source=$DeployTemp `
    --entry-point=generate_daily_embeddings `
    --trigger-http `
    --allow-unauthenticated `
    --memory="$Memory`MB" `
    --timeout="$Timeout`s" `
    --max-instances=1 `
    --service-account=GCP-general-sa@$ProjectId.iam.gserviceaccount.com `
    --set-env-vars="GCP_PROJECT_ID=$ProjectId,BQ_DATASET_ID=sg_job_market"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Deployment Successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Cloud Function URL:" -ForegroundColor Cyan
    Write-Host "https://$Region-$ProjectId.cloudfunctions.net/$FunctionName" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Create Cloud Scheduler job (see nlp/README.md)" -ForegroundColor Gray
    Write-Host "  2. Test manually: gcloud scheduler jobs run embeddings-daily-job --location=$Region" -ForegroundColor Gray
    Write-Host "  3. Monitor logs: gcloud functions logs read $FunctionName --region=$Region --limit=50" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ Deployment Failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Check logs above for errors" -ForegroundColor Gray
}

# Cleanup
Write-Host ""
Write-Host "Cleaning up temporary files..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $DeployTemp
Write-Host "✅ Cleanup complete" -ForegroundColor Green
