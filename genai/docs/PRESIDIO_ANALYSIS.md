# Presidio-Analyzer vs Custom Regex for PII Detection

**Context:** Evaluating whether to replace current custom regex PII detection with Microsoft's presidio-analyzer library.

---

## Current Implementation (Custom Regex)

**File:** `genai/guardrails.py` (lines 58-149)

```python
class PIIDetector:
    NRIC_PATTERN = r'\b[STFG]\d{7}[A-Z]\b'
    PHONE_PATTERN = r'\+65\s?\d{4}\s?\d{4}|\b[89]\d{7}\b'
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
```

**Coverage:** 4 PII types (NRIC, phone, email, credit card)

---

## Presidio-Analyzer Overview

**Library:** `presidio-analyzer` by Microsoft (open source)  
**Purpose:** PII detection and anonymization framework  
**Dependencies:** spaCy NLP model (en_core_web_lg ~560MB), presidio-analyzer (~50MB)  
**Total Size:** ~610MB additional

### How It Works
- Uses NLP (Named Entity Recognition) + regex patterns
- Pre-trained models for 50+ PII entity types
- Context-aware detection (e.g., "John" near "address" → likely name)
- Supports custom recognizers (can add Singapore-specific patterns)

### Built-in Entity Types
- PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD
- US_SSN, US_PASSPORT, UK_NHS, etc.
- IP_ADDRESS, IBAN_CODE, CRYPTO, URL

### Singapore-Specific Gap
- ❌ No built-in NRIC/FIN recognizer
- ✅ Can add custom recognizer (same regex we use now)

---

## Comparison: Presidio vs Custom Regex

### 1. Detection Accuracy

| Entity | Custom Regex | Presidio | Winner |
|--------|--------------|----------|--------|
| **Singapore NRIC** | ✅ S1234567D | ❌ Not detected (needs custom) | Custom |
| **Singapore Phone** | ✅ +65 9123 4567 | ⚠️ Generic phone (false positives) | Custom |
| **Email** | ✅ Basic pattern | ✅ Advanced validation | Presidio |
| **Credit Card** | ✅ 16 digits | ✅ Luhn checksum validation | Presidio |
| **Names** | ❌ Not detected | ✅ John Smith | Presidio |
| **Addresses** | ❌ Not detected | ✅ 123 Main St | Presidio |

**Verdict:** Presidio wins on global entities, custom wins on Singapore-specific.

---

### 2. Performance

| Metric | Custom Regex | Presidio | Winner |
|--------|--------------|----------|--------|
| **Latency** | ~1-2ms per query | ~50-100ms per query | Custom (50x faster) |
| **Memory** | ~5MB (Python runtime) | ~600MB (spaCy model) | Custom (120x lighter) |
| **Cold Start** | ~50ms (regex compile) | ~3-5 seconds (model load) | Custom (100x faster) |
| **CPU Usage** | Minimal (regex engine) | High (NLP inference) | Custom |

**Verdict:** Custom regex is significantly faster and lighter.

---

### 3. Maintainability

| Aspect | Custom Regex | Presidio | Winner |
|--------|--------------|----------|--------|
| **Code Complexity** | 150 lines, no dependencies | 20 lines + library | Presidio |
| **Pattern Updates** | Manual regex editing | Library handles updates | Presidio |
| **Singapore Patterns** | Built-in | Need custom recognizers | Custom |
| **Testing** | Simple unit tests | Complex integration tests | Custom |
| **Debugging** | Easy (print regex matches) | Harder (black-box NLP) | Custom |

**Verdict:** Tie - Presidio easier for global entities, custom easier for local patterns.

---

### 4. False Positives

| Scenario | Custom Regex | Presidio | Winner |
|----------|--------------|----------|--------|
| **Job ID "S1234567"** | ❌ Detected as NRIC | ❌ Also detected | Tie |
| **Reference code "T1234567X"** | ❌ Detected as NRIC | ❌ Also detected | Tie |
| **"John Python"** | ✅ Not detected | ❌ Detected as name | Custom |
| **"Call 91234567"** | ⚠️ Detected as phone | ⚠️ Also detected | Tie |

**Observation:** Both have false positives on job-specific text. Context matters more than tool.

---

### 5. Cost & Deployment

| Factor | Custom Regex | Presidio | Winner |
|--------|--------------|----------|--------|
| **Dockerfile Size** | 1.8GB (current) | 2.4GB (+600MB) | Custom |
| **Cloud Run Memory** | 4GB (current) | 5-6GB (need +1-2GB) | Custom |
| **Cloud Run Cost** | ~$15-30/month | ~$25-40/month | Custom |
| **Build Time** | 12 min (current) | 15-18 min (+spaCy) | Custom |
| **Dependencies** | 0 extra | 2 extra (spaCy + presidio) | Custom |

**Verdict:** Custom regex is cheaper and simpler to deploy.

---

## Presidio Pros (Why USE it)

### ✅ Advantages

1. **Broader Coverage**
   - Detects 50+ entity types out of the box
   - Useful if expanding beyond Singapore (regional platform)
   - Catches names, addresses, medical IDs we currently miss

2. **Context-Aware Detection**
   - Uses NLP to understand context ("John" near "email" → likely person)
   - Reduces false positives in some cases
   - Better for unstructured text

3. **Industry Standard**
   - Used by Microsoft, AWS Macie uses similar approach
   - Well-tested by large community
   - Regular updates for new PII types

4. **Advanced Features**
   - PII anonymization (replace with [REDACTED])
   - Confidence scores (0.0-1.0) for each detection
   - Language support (English, Spanish, German, etc.)

5. **Regulatory Compliance**
   - Helps meet GDPR, CCPA, PDPA requirements
   - Pre-built recognizers for global regulations
   - Audit trail for PII detection

---

## Presidio Cons (Why NOT use it)

### ❌ Disadvantages

1. **No Singapore-Specific Recognizers**
   - NRIC/FIN not built-in (would still need custom regex)
   - Singapore phone format needs custom recognizer
   - Not optimized for local context

2. **Performance Overhead**
   - 50x slower than regex (50ms vs 1ms)
   - 120x more memory (600MB vs 5MB)
   - 3-5 second cold start (model loading)

3. **Deployment Complexity**
   - +600MB Docker image size (1.8GB → 2.4GB)
   - +1-2GB Cloud Run memory needed (4GB → 5-6GB)
   - +30% build time (12 min → 15-18 min)
   - spaCy model download during build (network dependency)

4. **False Positives in Job Context**
   - "Python Developer" → detects "Python" as potential name
   - "John Doe & Associates" (company name) → detects "John Doe" as person
   - Job IDs like "S1234567" still trigger NRIC pattern

5. **Cost Impact**
   - +$10-15/month Cloud Run costs (extra memory/CPU)
   - Not justified for current 10 req/min traffic
   - Only makes sense at higher scale (100+ req/min)

6. **Over-Engineering Risk**
   - Current use case: Block obvious PII from job search queries
   - Users rarely input PII in "Find data scientist jobs" queries
   - Heavy tool for a light problem

---

## Custom Regex Pros

### ✅ Advantages

1. **Optimized for Singapore**
   - NRIC/FIN pattern specifically designed
   - Phone number format matches Singapore style (+65)
   - No wasted detection on irrelevant global patterns

2. **Blazing Fast**
   - 1-2ms per query (50x faster than presidio)
   - No model loading overhead
   - Minimal CPU usage

3. **Lightweight**
   - 0 external dependencies
   - 5MB memory footprint
   - 150 lines of code (easy to audit)

4. **Production-Ready Now**
   - Already deployed and tested (10/10 tests passing)
   - No migration risk
   - Known false positive patterns (can tune)

5. **Transparent**
   - Easy to debug (print regex matches)
   - Clear what's detected and why
   - Simple to add new patterns (just add regex)

---

## Custom Regex Cons

### ❌ Disadvantages

1. **Limited Entity Coverage**
   - Only 4 PII types (NRIC, phone, email, credit card)
   - Doesn't catch names, addresses, medical IDs

2. **Context-Unaware**
   - "S1234567" (job ID) vs "S1234567D" (NRIC) both trigger
   - Can't distinguish based on surrounding words

3. **Manual Maintenance**
   - New PII patterns need manual regex updates
   - No automatic updates from community

4. **Regulatory Gaps**
   - May not meet full GDPR/CCPA requirements (needs more entity types)
   - Auditors may question "homegrown" solution

---

## Recommendation

### ✅ KEEP Custom Regex (Don't use presidio)

**Reasons:**

1. **Fit for Purpose**
   - Current use case: Block PII from job search queries
   - Users rarely input sensitive data in "Find data scientist jobs"
   - 4 PII types are sufficient (NRIC, phone, email, card)

2. **Cost-Benefit**
   - Custom regex: 1ms, 5MB, $0 extra cost
   - Presidio: 50ms, 600MB, $10-15/month extra
   - Not justified for 10 req/min traffic

3. **Singapore-First**
   - NRIC is the most critical PII type for Singapore
   - Presidio doesn't have it (would need same regex anyway)
   - No benefit over custom for local patterns

4. **Already Production-Validated**
   - 10/10 tests passing
   - Deployed and working
   - Migration risk not worth it

5. **Industry Precedent**
   - AWS Bedrock Guardrails: Regex-based
   - Azure Content Safety: Hybrid (regex + ML)
   - Google Vertex AI Safety: Pattern matching first
   - Regex is industry standard for input validation

---

## When to Reconsider Presidio

**Consider switching IF:**

1. **Expanding Regionally**
   - Platform expands to Malaysia, Indonesia, etc.
   - Need to detect local IDs from multiple countries
   - Presidio's extensibility becomes valuable

2. **Higher Traffic**
   - Scaling to 100+ req/min (50ms latency acceptable)
   - Cost of extra resources justified by usage
   - Performance overhead becomes negligible

3. **Stricter Compliance**
   - GDPR/CCPA audits require broader PII coverage
   - Need to detect names, addresses, medical IDs
   - Custom regex insufficient for regulatory approval

4. **PII in Job Descriptions**
   - Expanding scope to scan job descriptions (not just queries)
   - Need context-aware detection (names near contact info)
   - NLP-based approach more accurate

---

## Hybrid Approach (Optional)

**If you want best of both worlds:**

1. **Keep custom regex for queries** (fast, lightweight)
2. **Add presidio for job descriptions** (thorough, context-aware)
3. **Use presidio asynchronously** (background PII scan, not blocking)

```python
# Fast path (query validation)
input_guards = InputGuardrails()  # Custom regex
result = input_guards.validate(query)  # 1ms

# Slow path (job description scan)
async def scan_job_descriptions():
    analyzer = PresidioAnalyzer()  # 50ms per job
    for job in new_jobs:
        pii_found = analyzer.analyze(job.description)
        if pii_found:
            flag_for_review(job)
```

**Benefit:** Best of both - fast queries, thorough job scans.

---

## Decision Summary

| Criteria | Custom Regex | Presidio | Winner |
|----------|--------------|----------|--------|
| **Singapore Coverage** | ✅ Excellent | ❌ Needs custom | Custom |
| **Performance** | ✅ 1ms | ❌ 50ms | Custom |
| **Memory** | ✅ 5MB | ❌ 600MB | Custom |
| **Cost** | ✅ $0 extra | ❌ $10-15/month | Custom |
| **Deployment** | ✅ Simple | ❌ Complex | Custom |
| **Global Coverage** | ❌ Limited | ✅ Excellent | Presidio |
| **Maintenance** | ⚠️ Manual | ✅ Automated | Presidio |
| **Compliance** | ⚠️ Basic | ✅ Comprehensive | Presidio |

**Verdict:** **Keep custom regex** for current scope. Reconsider presidio at 10x scale or regional expansion.

---

## Implementation Note

If you still want to try presidio:

```bash
# Add to requirements-api.txt
presidio-analyzer==2.2.354
spacy==3.7.2

# Download spacy model (in Dockerfile)
RUN python -m spacy download en_core_web_lg

# Add custom Singapore recognizer
from presidio_analyzer import Pattern, PatternRecognizer

nric_recognizer = PatternRecognizer(
    supported_entity="SG_NRIC",
    patterns=[Pattern("NRIC", r'\b[STFG]\d{7}[A-Z]\b', 0.8)]
)
analyzer.registry.add_recognizer(nric_recognizer)
```

**Build Impact:** +600MB image, +3-5 min build time, +1GB memory requirement.
