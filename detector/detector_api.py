"""
ForgeShield Python Detector API.

FastAPI wrapper around the local EnhancedForgeryDetector used by the Node.js
backend.
"""

import base64
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

detector = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global detector
    print("[detector_api] Loading EnhancedForgeryDetector...")
    from unified_detector import EnhancedForgeryDetector

    use_gpu_setting = os.environ.get("DETECTOR_USE_GPU", "auto").lower()
    use_gpu = use_gpu_setting in {"auto", "true", "1", "yes"}
    detector = EnhancedForgeryDetector(regional_lang="en", use_gpu=use_gpu)
    print("[detector_api] Detector ready.")
    yield
    detector = None


app = FastAPI(title="ForgeShield Detector API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    image_base64: str


class DetectResponse(BaseModel):
    verdict: str
    confidence: float
    risk_level: str
    scores: dict[str, float]
    explanations: dict[str, str]
    summary: str
    flags: list[dict[str, Any]]
    recommendations: list[str]
    document_type_detected: str
    component_status: dict[str, dict[str, Any]]
    ocr_backend: str | None = None
    model_reliability: dict[str, Any] | None = None
    suspicious_regions: list[dict[str, Any]] | None = None
    region_crops: list[dict[str, Any]] | None = None
    offline_llm_summary: str | None = None
    visualization_base64: str | None = None


@app.get("/health")
async def health():
    component_state = None
    if detector is not None:
        component_state = {
            "device": str(detector.device),
            "layoutlm_available": detector.layout_model is not None,
            "layoutlm_enabled": getattr(detector, "layout_enabled", False),
            "noiseprint_enabled": getattr(detector.noiseprint_expert, "enabled", False),
            "text_model": getattr(detector.text_expert, "model_name", None),
            "ocr_backend": getattr(detector.ocr_backend, "backend_name", None),
            "offline_llm_enabled": getattr(detector.offline_llm_expert, "enabled", False),
            "offline_llm_model": getattr(detector.offline_llm_expert, "model_name", None),
        }
    return {"status": "ok", "detector_ready": detector is not None, "components": component_state}


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized yet")

    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        temp_file.write(image_bytes)
        temp_file.flush()
        temp_file.close()

        results, visualization_base64 = detector.detect(temp_file.name)
        return DetectResponse(
            verdict=results["verdict"],
            confidence=float(results["confidence"]),
            risk_level=results["risk_level"],
            scores={key: float(value) for key, value in results["scores"].items()},
            explanations=results["explanations"],
            summary=results["summary"],
            flags=results["flags"],
            recommendations=results["recommendations"],
            document_type_detected=results["document_type_detected"],
            component_status=results["component_status"],
            ocr_backend=results.get("ocr_backend"),
            model_reliability=results.get("model_reliability"),
            suspicious_regions=results.get("suspicious_regions"),
            region_crops=results.get("region_crops"),
            offline_llm_summary=results.get("offline_llm_summary"),
            visualization_base64=visualization_base64,
        )
    except Exception as exc:
        print(f"[detector_api] Detection error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    port = int(os.environ.get("DETECTOR_PORT", 8000))
    uvicorn.run("detector_api:app", host="0.0.0.0", port=port, reload=False)
