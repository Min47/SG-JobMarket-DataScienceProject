#!/usr/bin/env pwsh
# =============================================================================
# Cloud Run JOB Deployment Script for Embeddings Generator
# Deploys Docker container as Cloud Run Job (matching scraper pattern)
# =============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cloud Run JOB Deployment: Embeddings Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Configuration
# =============================================================================

$PROJECT_ID = "sg-job-market"
$REGION = "asia-southeast1"
$JOB_NAME = "cloudjob-embeddings-generator"
$REPOSITORY = "embeddings-docker"
$IMAGE_NAME = "sg-job-embeddings"
$DOCKERFILE = "Dockerfile.embeddings"

$FULL_IMAGE_PATH = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/${IMAGE_NAME}:latest"

# Environment variables for Cloud Run
$ENV_VARS = @(
    "GCP_PROJECT_ID=$PROJECT_ID",
    "BQ_DATASET_ID=sg_job_market",
    "GCP_REGION=$REGION"
)

# Service account
$SERVICE_ACCOUNT = "GCP-general-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Cloud Run Job configuration
$CPU = "1"
$MEMORY = "2Gi"
$TIMEOUT = "3600s"  # 60 minutes
$MAX_RETRIES = "3"

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================

Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

# Check if Artifact Registry repository exists
$repoExists = gcloud artifacts repositories describe $REPOSITORY `
    --location=$REGION `
    --project=$PROJECT_ID `
    2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "      Creating Artifact Registry repository..." -ForegroundColor Gray
    gcloud artifacts repositories create $REPOSITORY `
        --repository-format=docker `
        --location=$REGION `
        --description="Docker images for embeddings generator" `
        --project=$PROJECT_ID
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✓ Created repository: $REPOSITORY" -ForegroundColor Green
    } else {
        Write-Host "      ✗ Failed to create repository" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "      ✓ Repository exists: $REPOSITORY" -ForegroundColor Green
}

# =============================================================================
# Step 2: Build Docker image
# =============================================================================

Write-Host "[2/6] Building Docker image..." -ForegroundColor Yellow
Write-Host "      Image: $FULL_IMAGE_PATH" -ForegroundColor Gray

docker build -t $FULL_IMAGE_PATH -f $DOCKERFILE .

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Docker image built successfully" -ForegroundColor Green

# =============================================================================
# Step 3: Push image to Artifact Registry
# =============================================================================

Write-Host "[3/6] Pushing image to Artifact Registry..." -ForegroundColor Yellow

# Configure Docker authentication
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

docker push $FULL_IMAGE_PATH

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Image pushed to Artifact Registry" -ForegroundColor Green

# =============================================================================
# Step 4: Deploy to Cloud Run Job
# =============================================================================

Write-Host "[4/6] Deploying to Cloud Run Job..." -ForegroundColor Yellow

# Check if job exists
$existingJob = gcloud run jobs describe $JOB_NAME `
    --region=$REGION `
    --project=$PROJECT_ID `
    --format="value(name)" 2>$null

if ($existingJob) {
    Write-Host "      Updating existing job: $JOB_NAME" -ForegroundColor Gray
    
    gcloud run jobs update $JOB_NAME `
        --image=$FULL_IMAGE_PATH `
        --region=$REGION `
        --project=$PROJECT_ID `
        --service-account=$SERVICE_ACCOUNT `
        --set-env-vars="$(($ENV_VARS -join ','))" `
        --memory=$MEMORY `
        --cpu=$CPU `
        --task-timeout=$TIMEOUT `
        --max-retries=$MAX_RETRIES `
        --execute-now=false `
        --quiet
} else {
    Write-Host "      Creating new job: $JOB_NAME" -ForegroundColor Gray
    
    gcloud run jobs create $JOB_NAME `
        --image=$FULL_IMAGE_PATH `
        --region=$REGION `
        --project=$PROJECT_ID `
        --service-account=$SERVICE_ACCOUNT `
        --set-env-vars="$(($ENV_VARS -join ','))" `
        --memory=$MEMORY `
        --cpu=$CPU `
        --task-timeout=$TIMEOUT `
        --max-retries=$MAX_RETRIES `
        --execute-now=false `
        --quiet
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Cloud Run Job deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Deployed to Cloud Run Job" -ForegroundColor Green

# =============================================================================
# Step 5: Verify deployment
# =============================================================================

Write-Host "[5/6] Verifying deployment..." -ForegroundColor Yellow

$JOB_READY = gcloud run jobs describe $JOB_NAME `
    --region=$REGION `
    --project=$PROJECT_ID `
    --format="value(status.conditions[0].type)" 2>$null

if ($JOB_READY) {
    Write-Host "      ✓ Job ready: $JOB_NAME" -ForegroundColor Green
} else {
    Write-Host "      ✗ Failed to verify job status" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Step 6: Deployment Summary
# =============================================================================

Write-Host "[6/6] Deployment Summary" -ForegroundColor Yellow
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "✓ Cloud Run Job Deployed Successfully!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Job Details:" -ForegroundColor Cyan
Write-Host "  Job Name:       $JOB_NAME" -ForegroundColor Gray
Write-Host "  Region:         $REGION" -ForegroundColor Gray
Write-Host "  Image:          $FULL_IMAGE_PATH" -ForegroundColor Gray
Write-Host "  Memory:         $MEMORY" -ForegroundColor Gray
Write-Host "  CPU:            $CPU" -ForegroundColor Gray
Write-Host "  Timeout:        $TIMEOUT" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Create scheduler job:" -ForegroundColor Gray
Write-Host "     .\deployment\NLP_02_Create_Embeddings_Scheduler.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Manual Commands:" -ForegroundColor Yellow
Write-Host "  # Run job manually" -ForegroundColor Gray
Write-Host "  gcloud run jobs execute $JOB_NAME --region=$REGION" -ForegroundColor White
Write-Host ""
Write-Host "  # View logs" -ForegroundColor Gray
Write-Host "  gcloud run jobs logs read $JOB_NAME --region=$REGION --limit=50" -ForegroundColor White
Write-Host ""
Write-Host "  # List executions" -ForegroundColor Gray
Write-Host "  gcloud run jobs executions list --job=$JOB_NAME --region=$REGION" -ForegroundColor White
Write-Host ""
Write-Host "Service Details:" -ForegroundColor Cyan
Write-Host "  Name:     $SERVICE_NAME" -ForegroundColor White
Write-Host "  URL:      $SERVICE_URL" -ForegroundColor White
Write-Host "  Region:   $REGION" -ForegroundColor White
Write-Host "  Image:    $FULL_IMAGE_PATH" -ForegroundColor White
Write-Host "  Timeout:  3600s (60 minutes)" -ForegroundColor White
Write-Host "  Memory:   2GB" -ForegroundColor White
Write-Host ""
Write-Host "Test the service:" -ForegroundColor Cyan
Write-Host "  curl -X POST $SERVICE_URL/generate \`" -ForegroundColor Gray
Write-Host "    -H 'Content-Type: application/json' \`" -ForegroundColor Gray
Write-Host "    -d '{\"limit\": 100, \"process_today\": false}'" -ForegroundColor Gray
Write-Host ""
Write-Host "Check logs:" -ForegroundColor Cyan
Write-Host "  gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50" -ForegroundColor Gray
Write-Host ""
Write-Host "Scheduler:" -ForegroundColor Cyan
Write-Host "  Job: $SCHEDULER_JOB" -ForegroundColor White
Write-Host "  Schedule: Daily at 2:00 AM SGT" -ForegroundColor White
Write-Host "  Run manually: gcloud scheduler jobs run $SCHEDULER_JOB --location=$REGION" -ForegroundColor Gray
Write-Host ""
