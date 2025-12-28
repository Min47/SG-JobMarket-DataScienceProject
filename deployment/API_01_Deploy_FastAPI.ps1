#!/usr/bin/env pwsh
# =============================================================================
# Cloud Run SERVICE Deployment Script for FastAPI GenAI Service
# Deploys Docker container as Cloud Run Service (HTTP endpoint)
# =============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Cloud Run SERVICE Deployment: FastAPI GenAI" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Configuration
# =============================================================================

$PROJECT_ID = "sg-job-market"
$REGION = "asia-southeast1"
$SERVICE_NAME = "genai-api"
$REPOSITORY = "genai-api-docker"
$IMAGE_NAME = "sg-job-api"
$CLOUDBUILD_CONFIG = "cloudbuild.api.yaml"

$FULL_IMAGE_PATH = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/${IMAGE_NAME}:latest"

$REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Environment variables for Cloud Run
$ENV_VARS = @(
    "GCP_PROJECT_ID=$PROJECT_ID",
    "BQ_DATASET_ID=sg_job_market",
    "GCP_REGION=$REGION",
    "PYTHONUNBUFFERED=1"
)

# Service account
$SERVICE_ACCOUNT = "GCP-general-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Cloud Run Service configuration
$CPU = "2"
$MEMORY = "4Gi"
$TIMEOUT = "300s"  # 5 minutes (enough for agent queries)
$MIN_INSTANCES = "0"  # Scale to zero when idle
$MAX_INSTANCES = "10"
$CONCURRENCY = "10"  # Max concurrent requests per instance

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================

Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

# Check if Artifact Registry repository exists
$null = gcloud artifacts repositories describe $REPOSITORY `
    --location=$REGION `
    --project=$PROJECT_ID `
    2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "      Creating Artifact Registry repository..." -ForegroundColor Gray
    gcloud artifacts repositories create $REPOSITORY `
        --repository-format=docker `
        --location=$REGION `
        --description="Docker images for FastAPI GenAI service" `
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
# Step 2: Build and push image via Cloud Build
# =============================================================================

Write-Host "[2/6] Building and pushing image via Cloud Build..." -ForegroundColor Yellow
Write-Host "      Context: $REPO_ROOT" -ForegroundColor Gray
Write-Host "      Config: $CLOUDBUILD_CONFIG" -ForegroundColor Gray
Write-Host "      Image: $FULL_IMAGE_PATH" -ForegroundColor Gray

Push-Location $REPO_ROOT
try {
    if (-not (Test-Path (Join-Path $REPO_ROOT "Dockerfile.api"))) {
        Write-Host "      ✗ Missing Dockerfile.api in repo root" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path (Join-Path $REPO_ROOT $CLOUDBUILD_CONFIG))) {
        Write-Host "      ✗ Missing $CLOUDBUILD_CONFIG in repo root" -ForegroundColor Red
        exit 1
    }

    gcloud builds submit . --config=$CLOUDBUILD_CONFIG --project=$PROJECT_ID
} finally {
    Pop-Location
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Cloud Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Image built and pushed successfully" -ForegroundColor Green

# =============================================================================
# Step 3: Deploy Cloud Run Service
# =============================================================================

Write-Host "[3/6] Deploying Cloud Run Service..." -ForegroundColor Yellow
Write-Host "      Service: $SERVICE_NAME" -ForegroundColor Gray
Write-Host "      Region: $REGION" -ForegroundColor Gray
Write-Host "      CPU: $CPU, Memory: $MEMORY" -ForegroundColor Gray
Write-Host "      Timeout: $TIMEOUT" -ForegroundColor Gray
Write-Host "      Min/Max Instances: $MIN_INSTANCES/$MAX_INSTANCES" -ForegroundColor Gray

# Build environment variables string
$ENV_VARS_STRING = ($ENV_VARS -join ',')

gcloud run services deploy $SERVICE_NAME `
    --image=$FULL_IMAGE_PATH `
    --platform=managed `
    --region=$REGION `
    --service-account=$SERVICE_ACCOUNT `
    --cpu=$CPU `
    --memory=$MEMORY `
    --timeout=$TIMEOUT `
    --min-instances=$MIN_INSTANCES `
    --max-instances=$MAX_INSTANCES `
    --concurrency=$CONCURRENCY `
    --set-env-vars=$ENV_VARS_STRING `
    --allow-unauthenticated `
    --port=8080 `
    --project=$PROJECT_ID

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Cloud Run deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Cloud Run Service deployed successfully" -ForegroundColor Green

# =============================================================================
# Step 4: Get service URL
# =============================================================================

Write-Host "[4/6] Retrieving service URL..." -ForegroundColor Yellow

$SERVICE_URL = gcloud run services describe $SERVICE_NAME `
    --platform=managed `
    --region=$REGION `
    --format='value(status.url)' `
    --project=$PROJECT_ID

if ($LASTEXITCODE -ne 0) {
    Write-Host "      ✗ Failed to retrieve service URL" -ForegroundColor Red
    exit 1
}

Write-Host "      ✓ Service URL: $SERVICE_URL" -ForegroundColor Green

# =============================================================================
# Step 5: Test health endpoint
# =============================================================================

Write-Host "[5/6] Testing health endpoint..." -ForegroundColor Yellow
Write-Host "      Waiting 10s for service to warm up..." -ForegroundColor Gray
Start-Sleep -Seconds 10

try {
    $response = Invoke-WebRequest -Uri "$SERVICE_URL/health" -Method Get -TimeoutSec 30
    $health = $response.Content | ConvertFrom-Json
    
    Write-Host "      ✓ Health check passed" -ForegroundColor Green
    Write-Host "        Status: $($health.status)" -ForegroundColor Gray
    Write-Host "        Version: $($health.version)" -ForegroundColor Gray
    
    if ($health.services.bigquery -eq "ok") {
        Write-Host "        BigQuery: ✓ OK" -ForegroundColor Green
    } else {
        Write-Host "        BigQuery: ✗ $($health.services.bigquery)" -ForegroundColor Yellow
    }
    
    if ($health.services.vertex_ai -eq "ok") {
        Write-Host "        Vertex AI: ✓ OK" -ForegroundColor Green
    } else {
        Write-Host "        Vertex AI: ✗ $($health.services.vertex_ai)" -ForegroundColor Yellow
    }
    
    if ($health.services.embeddings -eq "ok") {
        Write-Host "        Embeddings: ✓ OK" -ForegroundColor Green
    } else {
        Write-Host "        Embeddings: ✗ $($health.services.embeddings)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "      ✗ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# =============================================================================
# Step 6: Test vector search endpoint
# =============================================================================

Write-Host "[6/6] Testing vector search endpoint..." -ForegroundColor Yellow

try {
    $searchPayload = @{
        query = "software engineer"
        top_k = 3
    } | ConvertTo-Json

    $response = Invoke-WebRequest `
        -Uri "$SERVICE_URL/v1/search" `
        -Method Post `
        -ContentType "application/json" `
        -Body $searchPayload `
        -TimeoutSec 30
    
    $result = $response.Content | ConvertFrom-Json
    
    Write-Host "      ✓ Vector search working" -ForegroundColor Green
    Write-Host "        Found: $($result.count) jobs" -ForegroundColor Gray
    Write-Host "        Processing time: $($result.processing_time_ms)ms" -ForegroundColor Gray
    
    if ($result.jobs.Count -gt 0) {
        Write-Host "        Sample: $($result.jobs[0].job_title)" -ForegroundColor Gray
    }
} catch {
    Write-Host "      ✗ Vector search test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# =============================================================================
# Summary
# =============================================================================

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✓ DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Service Details:" -ForegroundColor White
Write-Host "   Service Name: $SERVICE_NAME" -ForegroundColor Gray
Write-Host "   Region: $REGION" -ForegroundColor Gray
Write-Host "   Image: $FULL_IMAGE_PATH" -ForegroundColor Gray
Write-Host "   Service URL: $SERVICE_URL" -ForegroundColor Gray
Write-Host ""
Write-Host "🔗 Available Endpoints:" -ForegroundColor White
Write-Host "   Root:        $SERVICE_URL/" -ForegroundColor Cyan
Write-Host "   Health:      $SERVICE_URL/health" -ForegroundColor Cyan
Write-Host "   Search:      $SERVICE_URL/v1/search" -ForegroundColor Cyan
Write-Host "   Chat:        $SERVICE_URL/v1/chat" -ForegroundColor Cyan
Write-Host "   Stats:       $SERVICE_URL/v1/stats" -ForegroundColor Cyan
Write-Host "   Job Details: $SERVICE_URL/v1/jobs/{id}" -ForegroundColor Cyan
Write-Host "   Similar:     $SERVICE_URL/v1/jobs/{id}/similar" -ForegroundColor Cyan
Write-Host "   Swagger UI:  $SERVICE_URL/docs" -ForegroundColor Cyan
Write-Host "   ReDoc:       $SERVICE_URL/redoc" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor White
Write-Host "   1. Test endpoints: curl $SERVICE_URL/health" -ForegroundColor Gray
Write-Host "   2. View Swagger docs: Open $SERVICE_URL/docs in browser" -ForegroundColor Gray
Write-Host "   3. Monitor logs: gcloud run services logs read $SERVICE_NAME --region=$REGION" -ForegroundColor Gray
Write-Host "   4. Update DNS: Point api.sg-job-market.com to $SERVICE_URL" -ForegroundColor Gray
Write-Host ""
