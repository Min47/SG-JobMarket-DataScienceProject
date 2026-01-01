# OPA/Gatekeeper & Trivy for SG Job Market Platform

**Context:** Evaluating if Open Policy Agent (OPA), Gatekeeper, and Trivy fit the scope of this job market intelligence project.

---

## Project Overview (Recap)

**What We're Building:**
- Job scraping and analytics platform (Singapore market)
- GCP Cloud Run services (scraper, ETL, embeddings, GenAI API)
- BigQuery data warehouse
- Streamlit dashboards
- FastAPI REST API with guardrails

**Current Infrastructure:**
- Dockerized microservices on Cloud Run
- Cloud Functions for ETL
- Cloud Scheduler for automation
- No Kubernetes (all serverless)

---

## 1. Open Policy Agent (OPA) & Gatekeeper

### What is OPA?

**Open Policy Agent** is a policy engine for cloud-native environments.

**Key Features:**
- Policy-as-code (Rego language)
- Decoupled authorization (policies separate from application)
- Works with Kubernetes, Envoy, Terraform, etc.

**Gatekeeper:**
- OPA specifically for Kubernetes
- Admission controller (validates resources before deployment)
- Enforces policies like "no privileged containers" or "all images must be scanned"

### Example OPA Policy (Rego)

```rego
# Deny containers without resource limits
package kubernetes.admission

deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.resources.limits
    msg := sprintf("Container %v must have resource limits", [container.name])
}
```

---

### Does OPA/Gatekeeper Fit Our Project?

#### ❌ **NO - Not a Good Fit**

**Reasons:**

1. **No Kubernetes**
   - Our project uses **Cloud Run** (serverless, not Kubernetes)
   - Gatekeeper is a Kubernetes-specific tool (admission controller)
   - OPA integrates best with Kubernetes API server
   - We don't have a k8s cluster to install Gatekeeper on

2. **Wrong Abstraction Level**
   - OPA is for **infrastructure policy** (pod security, namespace isolation)
   - Our guardrails are for **application policy** (PII, injection, hallucination)
   - Different layers of the stack

3. **Cloud Run Has Built-in Policies**
   - IAM roles (who can deploy)
   - Service account permissions (what services can access)
   - VPC connector policies (network isolation)
   - Managed by GCP, not OPA

4. **Overkill for Current Scale**
   - OPA adds complexity (learning Rego, deploying policy server)
   - Our "policies" are Python code in guardrails.py (simpler)
   - 10 req/min doesn't need enterprise policy engine

5. **Not Designed for GenAI Guardrails**
   - OPA validates YAML/JSON configs (Kubernetes manifests)
   - Our guardrails validate natural language queries (PII, injection)
   - Wrong tool for the job

---

### When Would OPA Make Sense?

**Consider OPA IF:**

1. **Migrating to Kubernetes**
   - Move from Cloud Run to GKE (Google Kubernetes Engine)
   - Need to enforce pod security standards (PSS)
   - Want centralized policy management across clusters

2. **Multi-Tenant Platform**
   - Platform has 100+ customers with different policies
   - Need to enforce "Customer A can only access Singapore data"
   - Require audit trail of policy decisions

3. **Complex Access Control**
   - Role-based access control (RBAC) not sufficient
   - Attribute-based access control (ABAC) needed
   - Policies like "data scientists can query but not export"

4. **Cross-Service Policy**
   - 10+ microservices with shared policies
   - Need consistent authorization across services
   - Centralized policy repo (GitOps for policies)

**Current Reality:** We have 4 services (scraper, ETL, embeddings, API), all serverless. IAM is sufficient.

---

## 2. Trivy (Security Scanner)

### What is Trivy?

**Trivy** is an open-source vulnerability scanner by Aqua Security.

**Scans:**
- Docker images (OS packages, app dependencies)
- Infrastructure as Code (Terraform, CloudFormation)
- Kubernetes manifests
- Git repos (secrets, misconfigurations)

**Detects:**
- CVEs (Common Vulnerabilities and Exposures)
- Misconfigurations (CIS benchmarks)
- Exposed secrets (API keys, passwords)
- Malware signatures

### Example Trivy Scan

```bash
trivy image genai-api:latest

# Output:
# genai-api:latest (debian 12.4)
# ==================================
# Total: 5 (UNKNOWN: 0, LOW: 2, MEDIUM: 2, HIGH: 1, CRITICAL: 0)
#
# ┌─────────────────┬────────────────┬──────────┬───────────────────┬───────────────┬──────────────────────────────────┐
# │     Library     │ Vulnerability  │ Severity │ Installed Version │ Fixed Version │             Title                │
# ├─────────────────┼────────────────┼──────────┼───────────────────┼───────────────┼──────────────────────────────────┤
# │ openssl         │ CVE-2024-12345 │ HIGH     │ 3.0.11-1          │ 3.0.13-1      │ OpenSSL buffer overflow          │
# └─────────────────┴────────────────┴──────────┴───────────────────┴───────────────┴──────────────────────────────────┘
```

---

### Does Trivy Fit Our Project?

#### ✅ **YES - Good Fit (Highly Recommended)**

**Reasons:**

1. **Scans Docker Images**
   - We have 4 Dockerfiles (scraper-jobstreet, scraper-mcf, embeddings, api)
   - All deployed to Cloud Run as containers
   - Trivy can scan for vulnerabilities before deployment

2. **Easy Integration with Cloud Build**
   - Add Trivy scan step to cloudbuild.yaml
   - Fail build if HIGH/CRITICAL CVEs found
   - Automated security checks on every deployment

3. **Python Dependency Scanning**
   - Our images have 50+ Python packages (requirements.txt)
   - Trivy scans pip packages for known vulnerabilities
   - Example: Flask 2.0.0 has CVE-2024-XXXX → Trivy alerts

4. **Infrastructure as Code Scanning**
   - We have Dockerfiles, Cloud Run YAML configs
   - Trivy can detect misconfigurations (root user, no health check)
   - Improve security posture proactively

5. **Zero-Cost Security**
   - Trivy is free and open source
   - Runs in Cloud Build (no extra infrastructure)
   - 2-3 minutes added to build time (acceptable)

6. **Industry Best Practice**
   - Most production systems scan images before deploy
   - Required for SOC 2, ISO 27001 compliance
   - Shows security maturity to stakeholders

---

### How to Add Trivy to Our Project

#### Step 1: Add Trivy Scan to Cloud Build

**File:** `cloudbuild.api.yaml` (or other cloudbuild files)

```yaml
steps:
  # Step 1: Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--file=Dockerfile.api'
      - '--tag=${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/${_IMAGE_NAME}:latest'
      - '.'
    timeout: 900s

  # Step 2: Scan image with Trivy
  - name: 'aquasec/trivy:latest'
    args:
      - 'image'
      - '--severity=HIGH,CRITICAL'  # Only fail on serious issues
      - '--exit-code=1'              # Exit 1 if vulnerabilities found (fails build)
      - '--no-progress'              # Cleaner logs
      - '--format=table'             # Human-readable output
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/${_IMAGE_NAME}:latest'
    timeout: 300s

  # Step 3: Push image (only if scan passed)
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPOSITORY}/${_IMAGE_NAME}:latest'
```

#### Step 2: Add Dockerfile Scan (Pre-Build)

```yaml
steps:
  # Step 0: Scan Dockerfile for misconfigurations
  - name: 'aquasec/trivy:latest'
    args:
      - 'config'
      - '--severity=MEDIUM,HIGH,CRITICAL'
      - '--exit-code=1'
      - 'Dockerfile.api'
    timeout: 60s
```

#### Step 3: Add Python Dependencies Scan (Optional)

```yaml
steps:
  # Scan Python dependencies before building image
  - name: 'aquasec/trivy:latest'
    args:
      - 'fs'
      - '--scanners=vuln'
      - '--severity=HIGH,CRITICAL'
      - '--exit-code=0'  # Don't fail build (just report)
      - 'requirements-api.txt'
```

---

### Trivy Pros for Our Project

#### ✅ Advantages

1. **Catches Vulnerabilities Early**
   - Detect CVEs before production deployment
   - Fix issues in dev, not after incidents

2. **No Extra Infrastructure**
   - Runs in Cloud Build (ephemeral container)
   - No agents to maintain
   - No servers to patch

3. **Fast Scans**
   - Image scan: 2-3 minutes
   - Dockerfile scan: 10-30 seconds
   - Negligible build time increase

4. **Actionable Reports**
   - Shows exact CVE ID and severity
   - Links to fix documentation
   - Suggests patched versions

5. **Compliance Friendly**
   - Generates SBOM (Software Bill of Materials)
   - Satisfies audit requirements
   - Proves due diligence

6. **Integrates with Cloud Security Command Center**
   - Trivy can push findings to GCP CSCC
   - Centralized security dashboard
   - Alerts via Cloud Monitoring

---

### Trivy Cons

#### ⚠️ Considerations

1. **False Positives**
   - Some CVEs are not exploitable in our context
   - May need to suppress low-risk findings
   - Requires manual review initially

2. **Build Time Increase**
   - +2-3 minutes per build
   - Acceptable for production (12 min → 15 min)
   - Can disable for dev builds

3. **Dependency Updates Needed**
   - Will flag outdated packages (good but time-consuming)
   - May force updates before ready
   - Example: pandas 2.0 → 2.3 (breaking changes)

4. **Not a Silver Bullet**
   - Scans known CVEs only (not zero-days)
   - Doesn't catch logic bugs in our code
   - Need other security measures too (guardrails, pen testing)

---

## Comparison Summary

| Tool | Purpose | Fit for Our Project | Recommendation |
|------|---------|---------------------|----------------|
| **OPA** | Infrastructure policy engine | ❌ Poor (no Kubernetes) | **Don't use** |
| **Gatekeeper** | Kubernetes admission control | ❌ Poor (no Kubernetes) | **Don't use** |
| **Trivy** | Vulnerability scanner | ✅ Excellent | **Use it!** |

---

## Final Recommendations

### ❌ Skip OPA/Gatekeeper

**Why:**
- Designed for Kubernetes (we use Cloud Run)
- Wrong layer (infrastructure vs application)
- Overkill for current scale (10 req/min, 4 services)
- Cloud Run IAM is sufficient for access control

**Alternative:**
- Keep application-level guardrails (guardrails.py)
- Use GCP IAM for service-to-service auth
- Use VPC Service Controls if need network isolation

---

### ✅ Add Trivy to Build Pipeline

**Why:**
- Scans Docker images for CVEs (50+ Python packages)
- Easy integration with Cloud Build (2-3 min scan)
- Industry best practice (security hygiene)
- Zero cost (open source, runs in Cloud Build)

**How:**
1. Add Trivy scan step to all cloudbuild.yaml files
2. Set `--exit-code=1` to fail builds on HIGH/CRITICAL CVEs
3. Monitor scan reports in Cloud Build logs
4. Update dependencies monthly based on findings

**Impact:**
- Build time: +2-3 minutes (15 min total)
- Deployment safety: Catches vulnerable deps before production
- Compliance: Satisfies security audit requirements

---

## Implementation Priority

### Phase 1: Add Trivy (High Priority)

**Task 4.6.2: Container Security Scanning**
- Add Trivy to cloudbuild.api.yaml (genai API)
- Add Trivy to cloudbuild.embeddings.yaml
- Add Trivy to cloudbuild.jobstreet.yaml, cloudbuild.mcf.yaml
- Document suppression process (if needed)

**Estimated Time:** 2-3 hours (add, test, document)

### Phase 2: Skip OPA/Gatekeeper (Not Needed)

**Reason:** Wrong tool for serverless Cloud Run architecture

**Alternative:** If we later migrate to Kubernetes (GKE), reconsider OPA/Gatekeeper

---

## When to Reconsider OPA/Gatekeeper

**Consider IF:**

1. **Migrate to Kubernetes**
   - Move from Cloud Run to GKE
   - Need pod security policies
   - Want GitOps for policy management

2. **Platform-as-a-Service**
   - Turn job market API into multi-tenant SaaS
   - 100+ customers with different policies
   - Need fine-grained ABAC

3. **Regulatory Requirements**
   - Need audit trail of all authorization decisions
   - Require policy versioning and rollback
   - Compliance mandates centralized policy engine

**Current Reality:** None of these apply. Cloud Run IAM + guardrails.py is sufficient.

---

## Example Trivy Integration (Quick Start)

### 1. Update cloudbuild.api.yaml

```yaml
steps:
  # Build image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.api', '-t', 'genai-api:latest', '.']

  # Scan for vulnerabilities
  - name: 'aquasec/trivy:latest'
    args:
      - 'image'
      - '--severity=HIGH,CRITICAL'
      - '--exit-code=1'
      - 'genai-api:latest'

  # Push if scan passed
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'genai-api:latest']
```

### 2. Test Locally

```bash
# Build image
docker build -f Dockerfile.api -t genai-api:test .

# Scan with Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image genai-api:test
```

### 3. Review Output

```
genai-api:test (debian 12.4)
Total: 12 (HIGH: 8, CRITICAL: 4)

┌───────────────┬────────────────┬──────────┬───────────────┬───────────────┬─────────────────────┐
│   Library     │ Vulnerability  │ Severity │    Installed  │     Fixed     │       Title         │
├───────────────┼────────────────┼──────────┼───────────────┼───────────────┼─────────────────────┤
│ langchain     │ CVE-2024-56789 │ CRITICAL │ 1.0.0         │ 1.0.5         │ SQL Injection       │
│ uvicorn       │ CVE-2024-12345 │ HIGH     │ 0.27.0        │ 0.27.1        │ Header Injection    │
└───────────────┴────────────────┴──────────┴───────────────┴───────────────┴─────────────────────┘
```

### 4. Fix Vulnerabilities

```bash
# Update requirements-api.txt
langchain==1.0.5  # Was 1.0.0
uvicorn==0.27.1   # Was 0.27.0

# Rebuild and rescan
docker build -f Dockerfile.api -t genai-api:test .
trivy image genai-api:test
# Output: Total: 0 (HIGH: 0, CRITICAL: 0) ✅
```

---

## Cost-Benefit Analysis

### OPA/Gatekeeper

| Benefit | Cost | Verdict |
|---------|------|---------|
| Centralized policy | Learn Rego, deploy OPA server | ❌ Not worth it |
| Kubernetes admission control | Need Kubernetes (don't have) | ❌ Doesn't apply |
| Audit trail | Complexity overhead | ❌ IAM sufficient |

**Total:** ❌ Not recommended

### Trivy

| Benefit | Cost | Verdict |
|---------|------|---------|
| Catch CVEs before production | +2-3 min build time | ✅ Worth it |
| Free vulnerability database | Update deps monthly | ✅ Worth it |
| Compliance documentation | Initial setup (2 hours) | ✅ Worth it |

**Total:** ✅ Highly recommended

---

## Conclusion

### 1. ❌ Don't Use OPA/Gatekeeper
- Wrong tool for Cloud Run architecture
- Designed for Kubernetes (we don't have)
- Application guardrails (guardrails.py) + Cloud Run IAM is sufficient

### 2. ✅ Use Trivy
- Scans Docker images for vulnerabilities
- Easy Cloud Build integration (2-3 min overhead)
- Industry best practice for container security
- Zero cost, high value

### 3. Priority
- **High:** Add Trivy to build pipeline (Task 4.6.2)
- **Low:** OPA/Gatekeeper (defer until Kubernetes migration)

**Ready to add Trivy?** I can update all cloudbuild.yaml files now.
