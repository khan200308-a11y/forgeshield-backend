"""
ForgeShield Python Detector API
FastAPI wrapper around EnhancedForgeryDetector — called by the Node.js backend.
Runs on port 8000 by default.
"""

import base64
import os
import sys
import tempfile

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="ForgeShield Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None


@app.on_event("startup")
async def startup():
    global detector
    print("[detector_api] Loading EnhancedForgeryDetector...")
    from unified_detector import EnhancedForgeryDetector
    detector = EnhancedForgeryDetector(regional_lang="en", use_gpu=True)
    print("[detector_api] Detector ready.")


class DetectRequest(BaseModel):
    image_base64: str  # base64-encoded PNG (no data URI prefix)


class DetectResponse(BaseModel):
    verdict: str           # FORGED | GENUINE
    confidence: float      # 0.0 – 1.0
    scores: dict           # per-expert scores
    explanations: dict     # human-readable explanations
    visualization_base64: str = None  # base64 PNG of ELA/mask viz


@app.get("/health")
async def health():
    return {"status": "ok", "detector_ready": detector is not None}


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized yet")

    try:
        img_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Save to a temp file so the detector (which uses file paths) can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()

        results, viz_b64 = detector.detect(tmp.name)

        return DetectResponse(
            verdict=results["verdict"],
            confidence=float(results["confidence"]),
            scores={k: float(v) for k, v in results["scores"].items()},
            explanations=results["explanations"],
            visualization_base64=viz_b64,
        )
    except Exception as e:
        print(f"[detector_api] Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


if __name__ == "__main__":
    port = int(os.environ.get("DETECTOR_PORT", 8000))
    uvicorn.run("detector_api:app", host="0.0.0.0", port=port, reload=False)
