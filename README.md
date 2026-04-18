# ForgeShield — AI Document Forgery Detection System

> **ThinkRoot x Vortex '26 | NIT Trichy | 36-Hour Online Hackathon**

---

## 1. Hackathon Context

| Field | Details |
|---|---|
| Event | ThinkRoot x Vortex '26 |
| Host | National Institute of Technology, Trichy |
| Format | 36-hour online hackathon |
| Track | AI / Security |

---

## 2. Problem Statement

Document forgery is a pervasive problem across high-stakes domains:

- **College admissions** — fake marksheets, forged bonafide certificates, tampered transcripts
- **Scholarship applications** — altered income certificates, fabricated government letters
- **Government & legal verification** — counterfeit ID cards, manipulated affidavits, cloned seals

Manual verification is slow, inconsistent, and scales poorly. Existing tools require expensive proprietary software or expert forensic analysts. Small institutions and verification offices have no accessible, affordable solution.

---

## 3. Solution Overview

ForgeShield provides a REST API that accepts any document (PDF or image), runs it through Claude's vision model using a forensic analysis prompt, and returns a structured, explainable report with:

```
User uploads document
        ↓
ThinkRoot Frontend (multipart/form-data POST)
        ↓
ForgeShield Backend (Express.js)
        ↓  PDF → per-page PNG conversion
        ↓  Image optimization (sharp)
Claude Vision API (claude-sonnet-4-20250514)
        ↓  Forensic analysis across 6 dimensions
Structured JSON Response
        ↓
ThinkRoot Frontend renders verdict + flags + heatmap
```

Every flag includes a **category**, **description**, **severity**, **confidence score**, and **spatial coordinates** so the frontend can overlay bounding boxes on the document.

---

## 4. Key Features

- 🔍 **Multi-dimensional forensic analysis** — 6 detection categories covering text, font, layout, image artifacts, seals, and metadata
- 📄 **PDF and image support** — accepts PDF (multi-page), JPG, and PNG; each PDF page is analyzed independently
- 🗺️ **Spatial region mapping** — every flag includes bounding box coordinates (x_percent, y_percent, width_percent, height_percent) for frontend overlay
- ⚡ **Explainable AI** — every flag has a human-readable description, category, severity, and confidence score
- 🌐 **RESTful API** — CORS-open, consumed by the ThinkRoot-deployed frontend or any HTTP client
- 🔒 **Input validation and rate limiting** — MIME + extension validation, 10 req/min/IP limit, 10 MB file cap

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                 │
│            uploads PDF / JPG / PNG document                 │
└──────────────────────────┬──────────────────────────────────┘
                           │  multipart/form-data
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            ThinkRoot Frontend (AI-generated UI)             │
│         Drag & drop upload → renders verdict + flags        │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /api/analyze
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           ForgeShield Backend  (Node.js / Express)          │
│                                                             │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────┐  │
│  │    Multer    │ → │  pdfToImage.js │ → │  sharp       │  │
│  │  file upload │   │  PDF → PNG[]   │   │  optimize    │  │
│  └──────────────┘   └────────────────┘   └──────┬───────┘  │
│                                                  │          │
│                                     base64 image(s)         │
│                                                  ▼          │
│                                    ┌─────────────────────┐  │
│                                    │  claudeAnalyzer.js  │  │
│                                    │  forensic prompt    │  │
│                                    └──────────┬──────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │  vision API call
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Anthropic Claude Vision API                    │
│         claude-sonnet-4-20250514 (multimodal)               │
└──────────────────────────┬──────────────────────────────────┘
                           │  structured JSON analysis
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 JSON Response to Frontend                   │
│      verdict / risk / flags / confidence / coordinates      │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Node.js ≥ 18 |
| Framework | Express.js 4 |
| AI Model | Anthropic Claude `claude-sonnet-4-20250514` (vision) |
| PDF Conversion (primary) | pdf2pic (requires GraphicsMagick + Ghostscript) |
| PDF Conversion (fallback) | pdfjs-dist (pure JS) |
| Image Optimization | sharp |
| File Upload | Multer |
| Rate Limiting | express-rate-limit |
| Environment | dotenv |

---

## 7. API Documentation

### GET /api/health

Returns server status.

**Response**
```json
{
  "status": "ok",
  "timestamp": "2026-04-16T10:00:00.000Z"
}
```

---

### POST /api/analyze

Analyze a document for forgery.

**Request**

| Field | Type | Description |
|---|---|---|
| `document` | File (multipart) | PDF, JPG, or PNG — max 10 MB |

**Response (success)**
```json
{
  "success": true,
  "filename": "marksheet.pdf",
  "total_pages": 2,
  "processing_time_ms": 3421,
  "results": [
    {
      "page": 1,
      "verdict": "FORGED",
      "overall_confidence": 91,
      "risk_level": "HIGH",
      "summary": "Multiple text tampering indicators detected...",
      "document_type_detected": "marksheet",
      "flags": [
        {
          "id": "flag_1",
          "category": "TEXT_TAMPERING",
          "description": "Grade field shows inconsistent character spacing",
          "severity": "HIGH",
          "confidence": 94,
          "region": {
            "description": "center-right grade column",
            "x_percent": 60,
            "y_percent": 45,
            "width_percent": 20,
            "height_percent": 5
          }
        }
      ],
      "recommendations": ["Cross-verify with issuing institution", "Check original ink signatures"]
    }
  ],
  "overall_verdict": "FORGED",
  "overall_risk": "HIGH"
}
```

**Response (error)**
```json
{
  "success": false,
  "error": "No file uploaded. Send a file in the \"document\" field.",
  "code": "NO_FILE"
}
```

**Error Codes**

| Code | HTTP Status | Meaning |
|---|---|---|
| `NO_FILE` | 400 | No file in the `document` field |
| `INVALID_FILE_TYPE` | 400 | MIME type not allowed |
| `MIME_EXTENSION_MISMATCH` | 400 | Extension doesn't match MIME type |
| `FILE_TOO_LARGE` | 400 | File exceeds 10 MB |
| `CONVERSION_FAILED` | 422 | PDF/image processing error |
| `RATE_LIMIT_EXCEEDED` | 429 | 10 req/min/IP limit hit |
| `NOT_FOUND` | 404 | Unknown route |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## 8. Setup Instructions

### Prerequisites

1. **Node.js ≥ 18** — [nodejs.org](https://nodejs.org)

2. **GraphicsMagick** (for pdf2pic PDF conversion)
   - macOS: `brew install graphicsmagick ghostscript`
   - Ubuntu/Debian: `sudo apt-get install graphicsmagick ghostscript`
   - Windows: Download from [graphicsmagick.org](http://www.graphicsmagick.org/) and add to PATH

   > If GraphicsMagick is not installed, the backend automatically falls back to `pdfjs-dist` for PDF conversion (no system deps required).

3. **Anthropic API Key** — [console.anthropic.com](https://console.anthropic.com)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd forgeshield-backend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=sk-ant-...

# Start in development mode
npm run dev

# Or start in production
npm start
```

### Verify

```bash
curl http://localhost:5000/api/health
# → {"status":"ok","timestamp":"..."}
```

### Test with a document

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "document=@/path/to/your/document.pdf"
```

---

## 9. Detection Categories

| Category | What ForgeShield Looks For |
|---|---|
| **TEXT_TAMPERING** | Inconsistent character spacing, misaligned baselines, copy-paste artifacts, color/opacity variations, contextually inconsistent content |
| **FONT_INCONSISTENCY** | Mixed font families or weights, rendering quality differences, anti-aliasing mismatches within the same document |
| **LAYOUT_ANOMALY** | Misaligned elements, uneven margins, inconsistent line spacing, composite sections from different sources |
| **IMAGE_ARTIFACT** | JPEG compression blocks, cloning artifacts, healing brush traces, unnatural edges, inconsistent noise, lighting mismatches |
| **SEAL_SIGNATURE** | Digitally inserted stamps (sharp edges, no ink spread), signature inconsistencies, missing security features |
| **METADATA** | Inconsistent dates, irregular serial numbers, modified letterheads, formatting that deviates from standard conventions |

---

## 10. Judging Criteria Alignment

| Judging Criterion | ForgeShield Feature |
|---|---|
| **Innovation / Novelty** | Multi-dimensional forensic prompt engineering with spatial bounding box output — not just "real or fake" but *where* and *why* |
| **Technical Complexity** | Dual PDF conversion pipeline (pdf2pic + pdfjs-dist fallback), vision model integration, structured output parsing with retry logic |
| **Practical Impact** | Directly addresses document fraud in college admissions, scholarships, and government verification — real-world deployment-ready |
| **Completeness** | Full REST API with validation, rate limiting, error handling, logging, multi-page PDF support |
| **Explainability / AI Use** | Every flag is categorized, described, scored, and located — Claude is used as a reasoning engine, not a black box |
| **Code Quality** | Modular structure, clear separation of concerns, fallback strategies, clean error codes |

---

## 11. Limitations & Future Scope

### Current Limitations

- Forensic analysis accuracy depends on image quality — low-resolution scans may miss subtle artifacts
- Rate limited to 10 req/min/IP — not suitable for bulk batch processing without modification
- PDF conversion requires GraphicsMagick for best quality (fallback is lower resolution)
- No persistent storage — documents are deleted after analysis

### Future Scope

- **Batch processing endpoint** — analyze multiple documents in parallel
- **Document history and audit trail** — store analysis results in a database
- **Fine-tuned model** — train on a labeled dataset of forged vs. authentic documents
- **OCR integration** — extract and cross-check text content against known templates
- **Webhook support** — async analysis for large PDFs with callback URL
- **Admin dashboard** — analytics on forgery patterns and document types flagged
- **Digital signature verification** — validate cryptographic signatures on PDFs

---

## 12. Team

| Name | Role |
|---|---|
| *(your name)* | Backend & AI Engineering |
| *(teammate)* | Frontend (ThinkRoot) |

---

## 13. License

MIT License — see [LICENSE](LICENSE) for details.
