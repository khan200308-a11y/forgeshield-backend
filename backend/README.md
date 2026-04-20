# Backend

This folder contains the ForgeShield Express backend and the built-in frontend.

## What It Does

- serves the UI from `public/index.html`
- accepts uploads on `POST /api/analyze`
- converts PDFs to page images
- runs Claude analysis when configured
- calls the Python detector
- merges results into a single page-level response

## Routes

### `GET /`

Serves the frontend UI.

### `GET /api/health`

Returns:

- backend status
- whether Claude is configured
- Python detector health and readiness

### `POST /api/analyze`

Upload field:

- `document`

Optional form field:

- `language`

Accepted formats:

- PDF
- JPG / JPEG
- PNG

Max file size:

- 10 MB

## Environment

Create `backend/.env` from the repo root `.env.example`.

Typical values:

```env
PORT=5000
NODE_ENV=development
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PYTHON_DETECTOR_URL=http://localhost:8000
PYTHON_DETECTOR_TIMEOUT_MS=120000
```

If `ANTHROPIC_API_KEY` is missing, the backend still works with Python-only analysis.

## Run

```powershell
cd backend
npm install
npm start
```

Then open `http://localhost:5000`.

## Frontend Notes

The UI now reflects the real analysis source:

- `Claude + Python fusion`
- `Python detector only`
- `Claude only`

It also shows:

- Python forensic dashboard
- suspicious regions
- suspicious crops
- reliability / OOD warning
- health status for backend and detector readiness

## Key Files

- `server.js`
- `routes/analyze.js`
- `utils/claudeAnalyzer.js`
- `utils/pythonDetector.js`
- `public/index.html`
