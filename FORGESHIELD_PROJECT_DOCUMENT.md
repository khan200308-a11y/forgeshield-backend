# ForgeShield — AI-Powered Document Forgery Detection System
### ThinkRoot × Vortex '26 Hackathon Submission

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution](#3-solution)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Frontend — Built with ThinkRoot](#6-frontend--built-with-thinkroot)
7. [Backend — Node.js REST API](#7-backend--nodejs-rest-api)
8. [AI Analysis Engine](#8-ai-analysis-engine)
9. [Document Processing Pipeline](#9-document-processing-pipeline)
10. [API Reference](#10-api-reference)
11. [Security & Reliability Features](#11-security--reliability-features)
12. [Analysis Output Schema](#12-analysis-output-schema)
13. [How to Run](#13-how-to-run)
14. [Project Structure](#14-project-structure)

---

## 1. Project Overview

**ForgeShield** is an AI-powered document forgery detection platform built for the **ThinkRoot × Vortex '26 Hackathon**. It allows users to upload any document — a certificate, marksheet, ID card, admit card, bonafide letter, or official correspondence — and receive an instant forensic analysis that identifies signs of tampering, manipulation, or outright forgery.

The system uses **multimodal AI vision** to examine documents the same way a trained forensic document examiner would: looking at fonts, layout, image artifacts, seals, signatures, metadata inconsistencies, and more.

**Core Value Proposition:**
- Upload a document → Get a forensic verdict in seconds
- No human examiner needed for initial screening
- Detailed, explainable flags pointing to exactly what looks suspicious and where on the document

---

## 2. Problem Statement

Document fraud is a widespread and growing problem across education, employment, legal, and financial sectors:

- **Fake educational certificates** are used to fraudulently obtain jobs and admissions
- **Forged ID cards** enable identity theft and unauthorized access
- **Tampered official letters** are used to deceive institutions
- Manual verification is **slow, expensive, and inconsistent**
- Institutions lack scalable tools to screen documents before human review

There is no accessible, fast, AI-powered tool that gives non-experts a clear, structured forensic verdict on document authenticity.

---

## 3. Solution

ForgeShield solves this with a **three-layer pipeline**:

```
User Uploads Document
        ↓
Backend validates & preprocesses (PDF → Image conversion, optimization)
        ↓
AI Vision Model performs 6-dimension forensic analysis
        ↓
Structured JSON verdict returned to frontend
        ↓
User sees verdict, confidence score, risk level, and pinpointed flags
```

The result is a clear **AUTHENTIC / SUSPICIOUS / FORGED** verdict with:
- An overall confidence score (0–100)
- A risk level (LOW / MEDIUM / HIGH)
- Specific flags describing what was detected, where on the document, and how severe it is
- Actionable recommendations

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (ThinkRoot)                  │
│  - Document upload UI                                    │
│  - Drag & drop / file picker (PDF, JPG, PNG)            │
│  - Results dashboard with verdict, flags, risk badge     │
│  - Flag visualizer with region highlights on document   │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP POST multipart/form-data
                     │  /api/analyze
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (Node.js + Express)            │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Multer    │   │   Rate       │   │    CORS      │ │
│  │  (upload)   │   │  Limiter     │   │  Middleware  │ │
│  └──────┬──────┘   └──────────────┘   └──────────────┘ │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │           /api/analyze  (POST route)             │   │
│  │  1. Validate file type, size, MIME/ext match     │   │
│  │  2. PDF → convert each page to PNG via pdf2pic   │   │
│  │     (fallback: pdfjs-dist pure JS renderer)      │   │
│  │  3. Optimize images via sharp (max 2000px)       │   │
│  │  4. Send each page to AI Analyzer                │   │
│  │  5. Aggregate results, cleanup temp files        │   │
│  │  6. Return structured JSON response              │   │
│  └──────┬──────────────────────────────────────────┘   │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │          claudeAnalyzer.js (AI Engine)           │   │
│  │  - Sends base64 image to AI Vision API           │   │
│  │  - Enforces strict JSON schema response          │   │
│  │  - Auto-retry with correction prompt on failure  │   │
│  └──────┬──────────────────────────────────────────┘   │
└─────────┼───────────────────────────────────────────────┘
          │  HTTPS API call (base64 image)
          ▼
┌─────────────────────────┐
│   AI Vision Model API   │
│  (Gemini 2.0 Flash /    │
│   Claude Vision)        │
└─────────────────────────┘
```

---

## 5. Technology Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| **Node.js** | ≥ 18.0.0 | Runtime environment |
| **Express.js** | 4.19.2 | HTTP server & routing |
| **Multer** | 1.4.5 | Multipart file upload handling |
| **Sharp** | 0.33.4 | Image optimization & resizing |
| **pdf2pic** | 3.1.3 | PDF-to-PNG conversion (primary, uses Ghostscript) |
| **pdfjs-dist** | 4.4.168 | PDF-to-PNG conversion (fallback, pure JS) |
| **express-rate-limit** | 7.3.1 | Rate limiting (10 req/min/IP) |
| **cors** | 2.8.5 | Cross-Origin Resource Sharing |
| **dotenv** | 16.4.5 | Environment variable management |
| **nodemon** | 3.1.4 | Dev server auto-reload |

### AI / ML
| Technology | Purpose |
|---|---|
| **Gemini 2.0 Flash** (Google AI) | Primary multimodal vision model for forensic analysis |
| **@google/generative-ai SDK** | Official Google Generative AI Node.js client |

### Frontend
| Technology | Purpose |
|---|---|
| **ThinkRoot** | Frontend UI framework/component library (see Section 6) |

---

## 6. Frontend — Built with ThinkRoot

The ForgeShield frontend is built using **ThinkRoot**, providing the complete user interface for the document forgery detection system.

### Frontend Features

**Document Upload Interface**
- Drag-and-drop file upload area
- File picker supporting PDF, JPG, and PNG formats
- File size validation (up to 10 MB)
- Upload progress indicator

**Results Dashboard**
- Large, prominent verdict badge: `AUTHENTIC` / `SUSPICIOUS` / `FORGED`
- Overall confidence score displayed as a percentage
- Risk level indicator: `LOW` / `MEDIUM` / `HIGH`
- Document type detection label (certificate, marksheet, ID card, etc.)

**Forensic Flags Panel**
- List of all detected anomalies/flags
- Each flag shows:
  - Category (Text Tampering, Font Inconsistency, Layout Anomaly, Image Artifact, Seal/Signature, Metadata)
  - Severity (HIGH / MEDIUM / LOW)
  - Confidence score
  - Human-readable description of the anomaly
  - Location on the document (region coordinates as percentages)
- Visual highlights/overlays on the document image showing where each flag was detected

**Summary & Recommendations**
- 2–3 sentence AI-generated forensic summary
- Actionable recommendations for next steps

**Multi-page Document Support**
- For PDFs with multiple pages, results are shown per-page
- Overall verdict aggregated across all pages (worst-case wins)

### Frontend ↔ Backend Communication

The frontend sends a `multipart/form-data` POST request to:
```
POST http://localhost:5000/api/analyze
Content-Type: multipart/form-data

Field name: "document"   → the file
Field name: "language"   → (optional) document language hint, e.g. "Hindi"
```

And receives a structured JSON response (see Section 12).

---

## 7. Backend — Node.js REST API

### Entry Point: `server.js`

The Express server configures all middleware and boots on port `5000` (configurable via `PORT` env var).

**Middleware Stack (in order):**

1. **CORS** — Open to all origins (`*`), allowing `GET`, `POST`, `OPTIONS` with `Content-Type` and `Authorization` headers. Required for frontend-backend communication across different localhost ports.

2. **Body Parser** — JSON and URL-encoded bodies up to 10 MB.

3. **Request Logger** — Logs every incoming request with ISO timestamp, method, and path to console.

4. **Rate Limiter** — `express-rate-limit` enforces **10 requests per IP per minute** on `/api/analyze`. Returns a `429` response with code `RATE_LIMIT_EXCEEDED` if exceeded. Prevents abuse and AI API cost runaway.

5. **Multer File Upload** — Handles `multipart/form-data` uploads:
   - Accepted MIME types: `application/pdf`, `image/jpeg`, `image/jpg`, `image/png`
   - Accepted extensions: `.pdf`, `.jpg`, `.jpeg`, `.png`
   - Max file size: **10 MB**
   - Files saved to `uploads/` directory with sanitized, timestamped filenames
   - Filenames sanitized: path separators and special characters replaced, truncated to 200 chars

**Routes:**
- `GET /api/health` — Health check endpoint, returns `{ status: "ok", timestamp }`. Used to verify the server is running.
- `POST /api/analyze` — Main document analysis endpoint (see Section 10).

**Global Error Handler** — Catches all unhandled errors:
- `LIMIT_FILE_SIZE` → 400 with user-friendly message
- `INVALID_FILE_TYPE` → 400 with type details
- Other errors → 500 (hides internal details in production)

---

### Route Handler: `routes/analyze.js`

The POST handler performs these steps sequentially:

**Step 1 — File Validation**
- Checks file was uploaded
- Validates MIME type against allowlist
- Validates file extension matches MIME type (prevents extension spoofing)
- Validates file size ≤ 10 MB

**Step 2 — Document Conversion**
- **PDF files:** Passed to `pdfToImages()` which converts every page to a base64 PNG string
- **Image files:** Read as buffer, passed through Sharp for optimization, encoded as base64

**Step 3 — Per-Page AI Analysis**
- Iterates through each page's base64 image
- PDF pages are re-optimized through Sharp before sending to AI
- Calls `analyzeDocument()` for each page with the base64 image, media type, and language hint
- Errors on individual pages are caught gracefully — the page gets a `SUSPICIOUS` result with the error message rather than crashing the whole request

**Step 4 — Aggregation**
- `aggregateVerdict()`: Returns `FORGED` if any page is FORGED, else `SUSPICIOUS` if any page is SUSPICIOUS, else `AUTHENTIC`
- `aggregateRisk()`: Returns `HIGH` if any page is HIGH, else `MEDIUM`, else `LOW`

**Step 5 — Cleanup & Response**
- Temp uploaded file is deleted from `uploads/` directory (`safeUnlink`)
- Structured JSON response returned with all per-page results plus aggregated verdict

---

## 8. AI Analysis Engine

### File: `utils/claudeAnalyzer.js`

The AI engine is the core intelligence of ForgeShield. It sends document images to a multimodal vision AI model and parses structured forensic analysis results.

### Six Forensic Dimensions

The AI model is instructed to analyze every document across exactly **six forensic dimensions**:

| # | Dimension | What It Looks For |
|---|---|---|
| 1 | **Text Tampering** | Inconsistent character spacing, misaligned baselines, copy-paste artifacts, color/opacity variations, contextually inconsistent content |
| 2 | **Font Inconsistencies** | Variations in font family, weight, size, rendering quality, or anti-aliasing that differ from the document's dominant style |
| 3 | **Layout Anomalies** | Misaligned elements, uneven margins, inconsistent line spacing, elements that break the visual grid, composited sections |
| 4 | **Image Editing Artifacts** | JPEG compression blocking, cloning artifacts, healing brush traces, unnatural edges, inconsistent noise patterns, shadow/lighting mismatches |
| 5 | **Seal & Signature Analysis** | Digitally inserted stamps/seals (sharp edges, missing ink spread), signature inconsistencies, missing security features |
| 6 | **Metadata & Context** | Inconsistent dates, serial numbers that don't follow expected patterns, modified letterheads, formatting deviations from standard conventions |

### Multi-Language Support

The analyzer accepts an optional `language` parameter. For non-English documents (e.g., Hindi, Telugu, Arabic), it adds a language-specific instruction to the system prompt, telling the model to apply forensic analysis criteria specific to that script's typography.

### Robust JSON Parsing with Auto-Retry

The AI model is instructed to return **raw JSON only** — no markdown, no explanation. However, models occasionally wrap responses in code fences. The analyzer handles this with a two-attempt strategy:

1. **First attempt:** Call the model, strip any markdown fences, parse JSON
2. **If parsing fails:** Start a chat session with the conversation history intact, send a correction prompt asking the model to return only raw JSON
3. **If retry also fails:** Return a safe fallback object with `verdict: SUSPICIOUS`, `overall_confidence: 0`, and `_parse_error: true` — the request never crashes

### Model Configuration

```
Model:          gemini-2.0-flash (fast, multimodal)
Max tokens:     2048 output tokens
System prompt:  Forensic document examiner persona with strict JSON schema instruction
```

---

## 9. Document Processing Pipeline

### PDF Processing: `utils/pdfToImage.js`

PDFs require conversion to images before the AI can analyze them visually. ForgeShield uses a **dual-method approach** for maximum compatibility:

**Primary Method: pdf2pic**
- Uses **Ghostscript** (system dependency) for high-fidelity rasterization
- Renders at **300 DPI** with output dimensions of **2480×3508px** (A4 at 300 DPI)
- Produces the highest quality images for AI analysis
- Requires Ghostscript and GraphicsMagick installed on the system

**Fallback Method: pdfjs-dist**
- Pure JavaScript PDF renderer — **no system dependencies**
- Uses the `canvas` npm package for Node.js canvas rendering
- Renders at **2× scale** (~150 DPI equivalent)
- Automatically used if pdf2pic/Ghostscript is unavailable

**Page Count Detection:**
- Uses `pdfjs-dist` to count pages before conversion
- Defaults to 1 page if detection fails
- Each page is converted and analyzed independently

### Image Optimization: Sharp

Before sending any image to the AI API, it is optimized with **Sharp**:
- Resized to maximum **2000×2000px** (preserving aspect ratio, no enlargement)
- Converted to **PNG** format with compression level 6
- This reduces API payload size while preserving forensically relevant detail

---

## 10. API Reference

### Health Check

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-18T10:30:00.000Z"
}
```

---

### Analyze Document

```
POST /api/analyze
Content-Type: multipart/form-data
```

**Request Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `document` | File | Yes | The document to analyze. PDF, JPG, or PNG. Max 10 MB. |
| `language` | String | No | Language hint for non-English documents (e.g., "Hindi", "Telugu"). Default: "English" |

**Success Response (200):**
```json
{
  "success": true,
  "filename": "certificate.pdf",
  "total_pages": 2,
  "processing_time_ms": 3421,
  "results": [ ... ],
  "overall_verdict": "SUSPICIOUS",
  "overall_risk": "MEDIUM"
}
```

**Error Responses:**

| Status | Code | Cause |
|---|---|---|
| 400 | `NO_FILE` | No file included in request |
| 400 | `INVALID_FILE_TYPE` | File type not supported |
| 400 | `MIME_EXTENSION_MISMATCH` | Extension doesn't match MIME type |
| 400 | `FILE_TOO_LARGE` | File exceeds 10 MB |
| 422 | `CONVERSION_FAILED` | PDF/image could not be processed |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests (>10/min) |
| 500 | `INTERNAL_ERROR` | Unexpected server error |

---

## 11. Security & Reliability Features

| Feature | Implementation | Purpose |
|---|---|---|
| **File type validation** | MIME type + extension checked independently | Prevents extension spoofing attacks |
| **File size limit** | 10 MB hard limit in Multer + route handler | Prevents DoS via large file uploads |
| **Filename sanitization** | Special characters stripped, length capped at 200 chars | Prevents path traversal attacks |
| **Temp file cleanup** | `safeUnlink` deletes uploaded files after processing | No files accumulate on the server |
| **Rate limiting** | 10 requests/IP/minute on analyze endpoint | Prevents API cost abuse |
| **CORS configuration** | Explicit methods and headers allowlist | Controlled cross-origin access |
| **Error isolation** | Per-page try/catch in analysis loop | One failed page doesn't crash the whole request |
| **AI retry logic** | Auto-retry with correction prompt on JSON parse failure | Handles model response variability |
| **Production error masking** | Internal error details hidden in `NODE_ENV=production` | No stack trace leakage |

---

## 12. Analysis Output Schema

Each page result follows this schema:

```json
{
  "page": 1,
  "verdict": "FORGED | SUSPICIOUS | AUTHENTIC",
  "overall_confidence": 87,
  "summary": "The document shows clear signs of text manipulation in the grade section...",
  "risk_level": "HIGH | MEDIUM | LOW",
  "document_type_detected": "marksheet",
  "flags": [
    {
      "id": "flag_1",
      "category": "TEXT_TAMPERING | FONT_INCONSISTENCY | LAYOUT_ANOMALY | IMAGE_ARTIFACT | SEAL_SIGNATURE | METADATA",
      "description": "Grades in the result table appear to have been overwritten with a different font rendering",
      "severity": "HIGH | MEDIUM | LOW",
      "confidence": 92,
      "region": {
        "description": "Center table, grade column",
        "x_percent": 60,
        "y_percent": 45,
        "width_percent": 20,
        "height_percent": 30
      }
    }
  ],
  "recommendations": [
    "Cross-verify grades with the issuing institution directly.",
    "Request original physical document for ink analysis."
  ]
}
```

**Top-level response also includes:**
- `overall_verdict` — Aggregated verdict across all pages (FORGED > SUSPICIOUS > AUTHENTIC)
- `overall_risk` — Aggregated risk across all pages (HIGH > MEDIUM > LOW)
- `total_pages` — Number of pages processed
- `processing_time_ms` — Total backend processing time in milliseconds

---

## 13. How to Run

### Prerequisites

- Node.js ≥ 18.0.0
- npm
- Ghostscript (optional, for PDF processing — fallback available without it)
- A valid AI API key (Gemini or Anthropic)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd forgeshield-backend

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API key:
# GEMINI_API_KEY=your_key_here
# PORT=5000

# 4. Start the server
npm run dev        # Development (auto-reload)
npm start          # Production
```

### Verify

```bash
curl http://localhost:5000/api/health
# → {"status":"ok","timestamp":"..."}
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key from AI Studio |
| `PORT` | No | Server port (default: 5000) |
| `NODE_ENV` | No | `development` or `production` |

---

## 14. Project Structure

```
forgeshield-backend/
├── server.js                  # Express app entry point, middleware, multer config
├── routes/
│   └── analyze.js             # POST /api/analyze handler, orchestration logic
├── utils/
│   ├── claudeAnalyzer.js      # AI vision model integration, forensic prompt, JSON parsing
│   └── pdfToImage.js          # PDF-to-PNG conversion (pdf2pic + pdfjs-dist fallback)
├── uploads/
│   └── .gitkeep               # Temp directory for uploaded files (auto-cleaned after processing)
├── .env                       # Environment variables (not committed)
├── .env.example               # Environment variable template
├── package.json               # Dependencies and scripts
└── FORGESHIELD_PROJECT_DOCUMENT.md  # This document
```

---

## Summary

ForgeShield is a **full-stack AI forensic document analysis system** that brings document fraud detection to anyone with a browser. Built in the ThinkRoot × Vortex '26 Hackathon, it demonstrates:

- **Practical AI application** — multimodal vision AI applied to a real-world fraud detection problem
- **Production-grade backend** — rate limiting, file validation, error isolation, temp file cleanup
- **Robust AI integration** — structured prompting, JSON enforcement, auto-retry, multi-language support
- **Multi-format support** — PDFs (multi-page), JPG, PNG with dual-method PDF conversion
- **Explainable results** — not just a verdict, but specific flags with locations and confidence scores

**The core insight:** Document forgery can often be detected visually by an expert — ForgeShield makes that expertise available at scale, instantly, through AI vision.

---

*ForgeShield — Built at ThinkRoot × Vortex '26 Hackathon*
