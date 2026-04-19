# ForgeShield — Document Forgery Detection System

Dual-engine document forgery detector combining:
- **Claude Vision AI** — semantic, layout, and contextual analysis
- **Python ML Detector** — pixel-level ELA, ViT font forensics, LayoutLMv3, Noiseprint++, LightGBM fusion

---

## Project Structure

```
final_project/
├── backend/                  Node.js Express API (port 5000)
│   ├── server.js
│   ├── routes/
│   │   └── analyze.js        POST /api/analyze — merges both engines
│   └── utils/
│       ├── claudeAnalyzer.js Claude Vision integration
│       ├── pdfToImage.js     PDF → PNG conversion
│       └── pythonDetector.js HTTP client for Python API
│
├── detector/                 Python FastAPI service (port 8000)
│   ├── detector_api.py       REST wrapper (start this first)
│   ├── unified_detector.py   7-expert ML detector
│   ├── training_pipeline.py  LightGBM training pipeline
│   ├── Noiseprint/           Camera noise fingerprint tool
│   └── requirements.txt
│
├── data/                     Dataset
│   ├── original/
│   └── forged_dataset/
│
├── .env.example
├── start.bat                 Windows — start both services
└── start.sh                  Linux/Mac — start both services
```

---

## Setup

### 1. Environment variables

```bash
cp .env.example backend/.env
# Edit backend/.env and set ANTHROPIC_API_KEY
```

### 2. Node.js backend

```bash
cd backend
npm install
```

### 3. Python detector

```bash
cd detector
pip install -r requirements.txt
```

**System dependencies (Windows):**
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) — install to `C:\Program Files\Tesseract-OCR\`
- [GraphicsMagick](http://www.graphicsmagick.org/) + [Ghostscript](https://www.ghostscript.com/) — required for PDF conversion via pdf2pic (optional, has JS fallback)

---

## Running

### Windows (recommended)
```
double-click start.bat
```

### Manual (two terminals)

**Terminal 1 — Python detector:**
```bash
cd detector
python detector_api.py
```

**Terminal 2 — Node.js backend:**
```bash
cd backend
npm start
```

---

## API

### `POST /api/analyze`

Upload a document for forgery analysis.

**Request:** `multipart/form-data`
- `document` — PDF, JPG, or PNG (max 10 MB)
- `language` *(optional)* — document language hint (e.g. `Hindi`, `Arabic`)

**Response:**
```json
{
  "success": true,
  "filename": "certificate.pdf",
  "total_pages": 1,
  "processing_time_ms": 8200,
  "overall_verdict": "FORGED",
  "overall_risk": "HIGH",
  "results": [
    {
      "page": 1,
      "verdict": "FORGED",
      "risk_level": "HIGH",
      "overall_confidence": 92,
      "summary": "...",
      "document_type_detected": "certificate",
      "flags": [...],
      "recommendations": [...],
      "claude_analysis": {
        "verdict": "FORGED",
        "confidence": 92,
        "risk_level": "HIGH"
      },
      "python_analysis": {
        "verdict": "FORGED",
        "confidence": 0.81,
        "scores": {
          "ela": 0.72,
          "visual": 0.65,
          "layout": 0.58,
          "ocr": 0.43,
          "text_perp": 0.61,
          "font_gmm": 0.49,
          "noiseprint": 0.55
        },
        "explanations": {...},
        "visualization_base64": "<base64 PNG>"
      }
    }
  ]
}
```

The `python_analysis` field is `null` if the Python detector service is unavailable — the system degrades gracefully to Claude-only analysis.

### `GET /api/health`

```json
{
  "status": "ok",
  "timestamp": "2026-04-19T12:00:00.000Z",
  "python_detector": {
    "status": "ok",
    "detector_ready": true
  }
}
```

---

## Training the LightGBM model

```bash
cd detector
python training_pipeline.py --train --genuine_folder ../data/original
```

The trained model is saved as `lightgbm_model.txt` in the `detector/` directory and is automatically loaded on next startup.

---

## Verdict Merge Logic

| Claude       | Python     | Python Confidence | Final Verdict |
|--------------|------------|-------------------|---------------|
| FORGED       | any        | any               | FORGED        |
| AUTHENTIC    | FORGED     | > 60%             | FORGED        |
| AUTHENTIC    | FORGED     | ≤ 60%             | SUSPICIOUS    |
| SUSPICIOUS   | GENUINE    | any               | SUSPICIOUS    |
| AUTHENTIC    | GENUINE    | any               | AUTHENTIC     |
