# ForgeShield

ForgeShield is a document forgery detection system with:

- a built-in frontend served by the Node backend at `http://localhost:5000`
- an Express API that accepts PDF, JPG, and PNG uploads
- a Python detector service that runs OCR, image forensics, rule-based checks, and learned fusion
- optional Claude analysis when `ANTHROPIC_API_KEY` is configured
- graceful fallback to Python-only analysis when Claude is unavailable

## Current Architecture

```text
Browser UI
  -> Node.js backend (`backend/`, port 5000)
     -> PDF/image preprocessing
     -> Claude analysis (optional)
     -> Python detector (`detector/`, port 8000)
     -> merged page-level results
```

The frontend is already included in this repo at `backend/public/index.html`.

## Main Features

- Upload UI for PDF, JPG, and PNG
- Per-page verdicts and aggregated document verdict
- Python forensic dashboard with:
  - composite evidence map
  - anomaly mask
  - suspicious field regions
  - suspicious crops
- CPU-friendly training and inference defaults
- cross-validated LightGBM fusion model
- out-of-distribution warning for unfamiliar document styles
- optional offline local LLM semantic audit layer

## Project Structure

```text
forgeshield-backend/
+-- backend/
|   +-- public/                 Frontend served at `/`
|   +-- routes/analyze.js       Main analysis route
|   +-- utils/                  Claude, Python, and PDF helpers
|   +-- server.js               Express entrypoint
|   `-- README.md
+-- detector/
|   +-- detector_api.py         FastAPI wrapper
|   +-- unified_detector.py     Main detector pipeline
|   +-- training_pipeline.py    Fusion training
|   +-- requirements.txt
|   `-- Noiseprint/
+-- data/
|   +-- original/
|   `-- forged_dataset/
+-- .env.example
`-- README.md
```

## Setup

### 1. Backend environment

Copy `.env.example` to `backend/.env`.

Example:

```env
PORT=5000
NODE_ENV=development
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PYTHON_DETECTOR_URL=http://localhost:8000
PYTHON_DETECTOR_TIMEOUT_MS=120000
DETECTOR_PORT=8000
```

`ANTHROPIC_API_KEY` is optional if you want Python-only operation.

### 2. Install backend dependencies

```powershell
cd backend
npm install
```

### 3. Install detector dependencies

```powershell
cd detector
pip install -r requirements.txt
```

Recommended Windows dependency:

- Tesseract OCR at `C:\Program Files\Tesseract-OCR\tesseract.exe`

Optional:

- PaddleOCR for stronger OCR
- GraphicsMagick and Ghostscript for the primary PDF conversion path

## Run

Start the Python detector first:

```powershell
cd detector
python detector_api.py
```

Then start the backend:

```powershell
cd backend
npm start
```

Open:

- `http://localhost:5000` for the frontend
- `http://localhost:5000/api/health` for backend health
- `http://localhost:8000/health` for detector health

## Training

Train the fusion model from the detector folder:

```powershell
cd detector
python training_pipeline.py --train --genuine_folder ..\data\original --overwrite_dataset --num_forged_per_genuine 4 --num_genuine_aug_per_genuine 2
```

Notes:

- training imports external labeled data from `data/forged_dataset` by default
- the detector is CPU-friendly by default
- the trained model is saved as `detector/lightgbm_model.txt`
- metadata is saved as `detector/lightgbm_model_meta.json`

If the feature schema changes, remove old artifacts first:

```powershell
Remove-Item feature_cache.json, lightgbm_model.txt, lightgbm_model_meta.json, lightgbm_model_features.pkl -ErrorAction SilentlyContinue
```

## Optional Offline LLM Layer

ForgeShield can run an optional local semantic audit layer through an Ollama-compatible endpoint.

Example:

```powershell
$env:FORGESHIELD_ENABLE_OFFLINE_LLM='1'
$env:FORGESHIELD_OFFLINE_LLM_MODEL='qwen2.5:3b-instruct'
python detector_api.py
```

This layer is optional and mainly improves semantic consistency checks and explanations. It does not replace the image forensics pipeline.

## API Summary

### `POST /api/analyze`

Request:

- `multipart/form-data`
- field `document`
- optional field `language`

Response includes:

- overall verdict and risk
- per-page results
- analysis source labels such as `Claude + Python fusion`, `Python detector only`, or `Claude only`
- Python scores, reliability info, visualization, suspicious regions, and suspicious crops when available

### `GET /api/health`

Returns backend status plus:

- whether Claude is configured
- whether the Python detector is reachable and ready

## Runtime Behavior

- If Claude is unavailable and Python works, ForgeShield still returns a usable result.
- If Python is unavailable and Claude works, Claude-only analysis is returned.
- If neither engine is available, `/api/analyze` returns an error.
- On CPU, heavyweight experts such as LayoutLMv3, Font ViT, and Noiseprint are disabled by default unless explicitly enabled.

## Important Files

- [backend/server.js](backend/server.js)
- [backend/routes/analyze.js](backend/routes/analyze.js)
- [backend/public/index.html](backend/public/index.html)
- [detector/detector_api.py](detector/detector_api.py)
- [detector/unified_detector.py](detector/unified_detector.py)
- [detector/training_pipeline.py](detector/training_pipeline.py)
