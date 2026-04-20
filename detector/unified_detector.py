import base64
import json
import os
import re
import subprocess
import tempfile
import warnings
from collections import Counter
from datetime import datetime
from io import BytesIO
from textwrap import fill
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import easyocr
import matplotlib
import numpy as np
import pytesseract
import scipy.io as sio
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, LayoutLMv3Model, LayoutLMv3Processor, ViTModel

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


FEATURE_NAMES = [
    "ela",
    "visual",
    "layout",
    "ocr",
    "font_gmm",
    "noiseprint",
    "copy_move",
    "semantic",
    "offline_llm",
    "texture",
    "text_perp",
]

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(DETECTOR_DIR, "lightgbm_model.txt")
DEFAULT_NOISEPRINT_PATH = os.environ.get("NOISEPRINT_PATH", os.path.join(DETECTOR_DIR, "Noiseprint"))


def clamp(value, minimum=0.0, maximum=1.0):
    return float(max(minimum, min(maximum, value)))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_map(values):
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def choose_auto(enabled_on_gpu=True, enabled_on_cpu=False):
    setting = os.environ.get(name := choose_auto.current_name, "auto").lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return enabled_on_gpu if choose_auto.current_device == "cuda" else enabled_on_cpu


choose_auto.current_name = ""
choose_auto.current_device = "cpu"


def auto_flag(name, device_type, enabled_on_gpu=True, enabled_on_cpu=False):
    choose_auto.current_name = name
    choose_auto.current_device = device_type
    return choose_auto(enabled_on_gpu=enabled_on_gpu, enabled_on_cpu=enabled_on_cpu)


TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class OCRBackend:
    def __init__(self, languages, use_gpu=False):
        self.languages = languages
        self.use_gpu = use_gpu
        self.backend_name = "easyocr"
        self.error = None
        requested = os.environ.get("FORGESHIELD_OCR_BACKEND", "auto").lower()

        self.paddle = None
        if requested in {"auto", "paddleocr"} and PaddleOCR is not None:
            try:
                lang = "en"
                self.paddle = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
                self.backend_name = "paddleocr"
                print("[detector] OCR backend: PaddleOCR.")
                return
            except Exception as exc:
                self.error = str(exc)
                self.paddle = None
                if requested == "paddleocr":
                    print(f"[detector] PaddleOCR unavailable, falling back to EasyOCR: {self.error}")

        self.easyocr_reader = easyocr.Reader(sorted(set(languages)), gpu=use_gpu)
        self.backend_name = "easyocr"
        print("[detector] OCR backend: EasyOCR.")

    def extract(self, image_path):
        if self.paddle is not None:
            try:
                result = self.paddle.ocr(image_path, cls=True)
                entries = []
                for page in result or []:
                    for item in page or []:
                        bbox, payload = item
                        text, confidence = payload
                        entries.append(
                            {
                                "bbox": bbox,
                                "text": text,
                                "confidence": safe_float(confidence, 0.0),
                            }
                        )
                text = " ".join(entry["text"] for entry in entries if entry["text"]).strip()
                return {"entries": entries, "text": text, "backend": self.backend_name, "error": None}
            except Exception as exc:
                self.error = str(exc)

        try:
            result = self.easyocr_reader.readtext(image_path, detail=1)
            entries = [
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": safe_float(confidence, 0.0),
                }
                for bbox, text, confidence in result
            ]
            text = " ".join(entry["text"] for entry in entries if entry["text"]).strip()
            return {"entries": entries, "text": text, "backend": self.backend_name, "error": self.error}
        except Exception as exc:
            return {"entries": [], "text": "", "backend": self.backend_name, "error": str(exc)}


class TextForensicsExpert:
    def __init__(self, device):
        self.device = device
        self.enabled = auto_flag("FORGESHIELD_ENABLE_TEXT_LM", device.type, enabled_on_gpu=True, enabled_on_cpu=False)
        default_model = "distilgpt2" if self.device.type == "cpu" else "gpt2"
        self.model_name = os.environ.get("FORGESHIELD_TEXT_MODEL", default_model)
        self.max_length = int(os.environ.get("FORGESHIELD_TEXT_MAX_LENGTH", "192" if self.device.type == "cpu" else "384"))
        self.available = False
        self.error = None

        if not self.enabled:
            self.error = "disabled for current runtime"
            print("[detector] Text LM disabled for CPU-friendly runtime.")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            self.available = True
            print(f"[detector] Text forensics ready with {self.model_name}.")
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            print(f"[detector] Text forensics fallback enabled: {self.error}")

    def analyze(self, text):
        cleaned = (text or "").strip()
        token_count = len(cleaned.split())

        weird_punctuation = cleaned.count("|") + cleaned.count("_") + cleaned.count("~")
        digit_chunks = re.findall(r"\d+", cleaned)
        repeated_digits = sum(1 for token in digit_chunks if len(token) >= 8)
        heuristic = clamp((weird_punctuation * 0.08) + (repeated_digits * 0.12) + (token_count < 3) * 0.15)

        if not cleaned:
            return 0.2, {"perplexity": None, "token_count": 0, "fallback": True, "disabled": not self.enabled}

        if not self.available:
            return clamp(0.2 + heuristic), {
                "perplexity": None,
                "token_count": token_count,
                "fallback": True,
                "disabled": not self.enabled,
                "error": self.error,
            }

        try:
            encoded = self.tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded, labels=encoded["input_ids"])
            perplexity = min(torch.exp(outputs.loss).item(), 1000.0)
            score = clamp((perplexity - 40.0) / 120.0)
            return score, {
                "perplexity": perplexity,
                "token_count": token_count,
                "fallback": False,
                "disabled": False,
            }
        except Exception as exc:
            return clamp(0.25 + heuristic), {
                "perplexity": None,
                "token_count": token_count,
                "fallback": True,
                "disabled": False,
                "error": str(exc),
            }


class OfflineLLMExpert:
    def __init__(self):
        self.enabled = env_flag("FORGESHIELD_ENABLE_OFFLINE_LLM", False)
        self.endpoint = os.environ.get("FORGESHIELD_OFFLINE_LLM_URL", "http://127.0.0.1:11434/api/generate")
        self.model_name = os.environ.get("FORGESHIELD_OFFLINE_LLM_MODEL", "qwen2.5:3b-instruct")
        self.timeout_sec = int(os.environ.get("FORGESHIELD_OFFLINE_LLM_TIMEOUT_SEC", "20"))
        self.available = self.enabled
        self.error = None

        if self.enabled:
            print(f"[detector] Offline LLM audit enabled with {self.model_name}.")
        else:
            print("[detector] Offline LLM audit disabled.")

    def _prompt(self, text, document_type):
        trimmed = re.sub(r"\s+", " ", (text or "").strip())[:2400]
        return f"""
You are auditing OCR text from a possibly forged {document_type or 'document'}.
Return strict JSON only with this schema:
{{
  "score": <number 0 to 1>,
  "summary": "<short sentence>",
  "issues": ["<issue>"],
  "suspicious_fields": ["<short token or field excerpt>"]
}}

Focus on impossible dates, repeated identifiers, mismatched totals, inconsistent official wording, and suspicious edits in numeric fields.
If the text is too short or noisy, return a low score and explain why.

OCR TEXT:
{trimmed}
""".strip()

    def analyze(self, text, document_type="unknown"):
        cleaned = (text or "").strip()
        if not cleaned:
            return 0.15, {
                "fallback": True,
                "disabled": not self.enabled,
                "summary": "No OCR text was available for offline semantic audit.",
                "issues": [],
                "suspicious_fields": [],
            }

        if not self.enabled:
            return 0.2, {
                "fallback": True,
                "disabled": True,
                "summary": "Offline LLM audit disabled.",
                "issues": [],
                "suspicious_fields": [],
            }

        try:
            payload = json.dumps(
                {
                    "model": self.model_name,
                    "prompt": self._prompt(cleaned, document_type),
                    "stream": False,
                    "format": "json",
                }
            ).encode("utf-8")
            req = urllib_request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self.timeout_sec) as response:
                raw = json.loads(response.read().decode("utf-8"))
            parsed = json.loads((raw.get("response") or "{}").strip())
            score = clamp(parsed.get("score", 0.2))
            issues = [str(item)[:180] for item in parsed.get("issues", [])[:5]]
            suspicious_fields = [str(item)[:80] for item in parsed.get("suspicious_fields", [])[:6]]
            return score, {
                "fallback": False,
                "disabled": False,
                "summary": str(parsed.get("summary") or "Offline LLM audit completed."),
                "issues": issues,
                "suspicious_fields": suspicious_fields,
                "model": self.model_name,
            }
        except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self.error = str(exc)
            self.available = False
            return 0.2, {
                "fallback": True,
                "disabled": False,
                "summary": "Offline LLM audit unavailable; using rule-based semantic analysis only.",
                "issues": [],
                "suspicious_fields": [],
                "error": self.error,
                "model": self.model_name,
            }


class FontForensicsExpert:
    def __init__(self, device):
        self.device = device
        self.max_patches = 24
        self.enabled = auto_flag("FORGESHIELD_ENABLE_FONT_VIT", device.type, enabled_on_gpu=True, enabled_on_cpu=False)
        self.available = False
        self.error = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if not self.enabled:
            self.error = "disabled for current runtime"
            print("[detector] Font ViT disabled for CPU-friendly runtime.")
            return

        try:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.vit.eval()
            self.vit.to(self.device)
            self.available = True
            print("[detector] Font forensics ready.")
        except Exception as exc:
            self.error = str(exc)
            print(f"[detector] Font forensics fallback enabled: {self.error}")

    def extract_patches(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patches = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if width < 8 or height < 8 or area < 100 or area > 7000:
                continue
            patches.append(image[y : y + height, x : x + width])

        if len(patches) > self.max_patches:
            indexes = np.linspace(0, len(patches) - 1, num=self.max_patches, dtype=int)
            patches = [patches[index] for index in indexes]
        return patches

    def _fallback_score(self, patches):
        if not patches:
            return 0.25, {"num_patches": 0, "fallback": True, "disabled": not self.enabled}

        heights = np.array([patch.shape[0] for patch in patches], dtype=np.float32)
        widths = np.array([patch.shape[1] for patch in patches], dtype=np.float32)
        aspect = widths / np.maximum(heights, 1.0)
        ink_density = np.array([np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) < 180) for patch in patches], dtype=np.float32)
        score = clamp(
            0.35 * (np.std(heights) / (np.mean(heights) + 1e-6))
            + 0.35 * (np.std(aspect) / (np.mean(aspect) + 1e-6))
            + 0.30 * (np.std(ink_density) / (np.mean(ink_density) + 1e-6))
        )
        return score, {
            "num_patches": int(len(patches)),
            "fallback": True,
            "disabled": not self.enabled,
            "error": self.error,
        }

    def _embedding(self, patch):
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(patch_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.vit(pixel_values=tensor).last_hidden_state.mean(dim=1)
        vector = hidden.squeeze(0).detach().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-8)

    def analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        patches = self.extract_patches(image)
        if not patches or not self.available:
            return self._fallback_score(patches)

        try:
            embeddings = np.stack([self._embedding(patch) for patch in patches], axis=0)
            centroid = np.median(embeddings, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            distance = 1.0 - np.dot(embeddings, centroid)
            score = clamp(np.percentile(distance, 85) / 0.45)
            return score, {
                "num_patches": int(len(patches)),
                "mean_distance": float(np.mean(distance)),
                "fallback": False,
                "disabled": False,
            }
        except Exception as exc:
            score, details = self._fallback_score(patches)
            details["error"] = str(exc)
            return score, details


class NoiseprintExpert:
    def __init__(self, noiseprint_path=DEFAULT_NOISEPRINT_PATH, device_type="cpu"):
        default_enabled = device_type == "cuda"
        self.enabled = auto_flag("FORGESHIELD_ENABLE_NOISEPRINT", device_type, enabled_on_gpu=True, enabled_on_cpu=default_enabled)
        self.noiseprint_path = os.path.abspath(noiseprint_path)
        self.python_path = os.environ.get("NOISEPRINT_PYTHON", r"D:\anaconda3\envs\noiseprint_env\python.exe")
        self.script_path = os.path.join(self.noiseprint_path, "main_blind.py")
        self.failure_count = 0
        self.disabled_reason = None

        if not self.enabled:
            self.disabled_reason = "disabled for current runtime"
            print("[detector] Noiseprint fallback enabled: disabled for CPU-friendly runtime.")
            return

        self.enabled = os.path.exists(self.python_path) and os.path.exists(self.script_path)
        if not self.enabled:
            self.disabled_reason = "Noiseprint executable or script not found."
            print(f"[detector] Noiseprint fallback enabled: {self.disabled_reason}")
        else:
            print("[detector] Noiseprint subprocess enabled.")

    def _fallback_score(self, image_path, reason=None):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.5, {"fallback": True, "disabled": not self.enabled, "reason": reason or "image_unreadable"}

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blur)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        score = clamp(0.6 * (float(np.mean(residual)) / 42.0) + 0.4 * (float(np.std(laplacian)) / 65.0))
        return score, {
            "fallback": True,
            "disabled": not self.enabled,
            "reason": reason or self.disabled_reason or "noiseprint_disabled",
            "residual_mean": float(np.mean(residual)),
            "edge_std": float(np.std(laplacian)),
        }

    def analyze(self, image_path):
        if not self.enabled:
            return self._fallback_score(image_path, self.disabled_reason)

        resized_temp = None
        mat_path = None
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._fallback_score(image_path, "image_unreadable")

            height, width = image.shape[:2]
            max_dim = int(os.environ.get("NOISEPRINT_MAX_DIM", "768"))
            run_path = image_path

            if max(height, width) > max_dim:
                scale = max_dim / float(max(height, width))
                resized = cv2.resize(image, (int(width * scale), int(height * scale)))
                resized_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                resized_temp.close()
                cv2.imwrite(resized_temp.name, resized)
                run_path = resized_temp.name

            mat_file = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
            mat_file.close()
            mat_path = mat_file.name
            result = subprocess.run(
                [self.python_path, self.script_path, run_path, mat_path],
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("NOISEPRINT_TIMEOUT_SEC", "45")),
            )

            if result.returncode != 0:
                combined = (result.stderr or result.stdout or "").strip()
                short_error = combined.splitlines()[0] if combined else "noiseprint subprocess failed"
                self.failure_count += 1
                lowered = combined.lower()
                if (
                    "paging file is too small" in lowered
                    or "cudart64_100.dll not found" in lowered
                    or "failed to load the native tensorflow runtime" in lowered
                    or self.failure_count >= 2
                ):
                    self.enabled = False
                    self.disabled_reason = short_error
                return self._fallback_score(image_path, short_error)

            mat = sio.loadmat(mat_path)
            heatmap = next((mat[key] for key in ("heatmap", "map", "out", "scoremap") if key in mat), None)
            if heatmap is None:
                return self._fallback_score(image_path, "noiseprint_output_missing")

            heatmap = heatmap.astype(np.float32)
            heatmap -= np.min(heatmap)
            heatmap /= np.max(heatmap) + 1e-8
            score = clamp(float(np.mean(heatmap)) + 0.25 * float(np.std(heatmap)))
            return score, {"fallback": False, "disabled": False, "heatmap_mean": float(np.mean(heatmap))}
        except Exception as exc:
            self.failure_count += 1
            if self.failure_count >= 2:
                self.enabled = False
                self.disabled_reason = str(exc)
            return self._fallback_score(image_path, str(exc))
        finally:
            if resized_temp and os.path.exists(resized_temp.name):
                os.unlink(resized_temp.name)
            if mat_path and os.path.exists(mat_path):
                os.unlink(mat_path)


class PixelLocalizer:
    def fuse_maps(self, image_shape, maps, region_proposals=None):
        height, width = image_shape[:2]
        fused = np.zeros((height, width), dtype=np.float32)
        total_weight = 0.0

        for evidence_map, weight in maps:
            if evidence_map is None:
                continue
            resized = cv2.resize(np.asarray(evidence_map, dtype=np.float32), (width, height))
            fused += weight * normalize_map(resized)
            total_weight += float(weight)

        if total_weight > 0:
            fused /= total_weight

        for region in region_proposals or []:
            x = int((safe_float(region.get("x_percent"), 0.0) / 100.0) * width)
            y = int((safe_float(region.get("y_percent"), 0.0) / 100.0) * height)
            box_width = max(4, int((safe_float(region.get("width_percent"), 0.0) / 100.0) * width))
            box_height = max(4, int((safe_float(region.get("height_percent"), 0.0) / 100.0) * height))
            confidence = safe_float(region.get("confidence"), 0.0) / 100.0
            x2 = min(width, x + box_width)
            y2 = min(height, y + box_height)
            fused[y:y2, x:x2] = np.maximum(fused[y:y2, x:x2], confidence)

        fused = cv2.GaussianBlur(fused, (0, 0), 1.2)
        return normalize_map(fused)

    def generate_mask(self, evidence_map):
        normalized = (normalize_map(evidence_map) * 255.0).astype(np.uint8)
        threshold = max(40, int(np.percentile(normalized, 89)))
        _, mask = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _iou(self, first, second):
        x1 = max(first["x"], second["x"])
        y1 = max(first["y"], second["y"])
        x2 = min(first["x"] + first["w"], second["x"] + second["w"])
        y2 = min(first["y"] + first["h"], second["y"] + second["h"])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = (first["w"] * first["h"]) + (second["w"] * second["h"]) - intersection
        return intersection / max(union, 1)

    def extract_regions(self, mask, image_shape, confidence, region_proposals=None, evidence_map=None):
        height, width = image_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < 0.0015 * height * width:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            local_confidence = confidence
            if evidence_map is not None:
                patch = evidence_map[y : y + box_height, x : x + box_width]
                if patch.size:
                    local_confidence = max(local_confidence, float(np.mean(patch)))
            regions.append(
                {
                    "description": "Composite anomaly hotspot",
                    "x": x,
                    "y": y,
                    "w": box_width,
                    "h": box_height,
                    "confidence": int(clamp(local_confidence) * 100),
                    "source": "evidence_map",
                }
            )
            if len(regions) == 5:
                break

        for proposal in region_proposals or []:
            rect = {
                "x": int((safe_float(proposal.get("x_percent"), 0.0) / 100.0) * width),
                "y": int((safe_float(proposal.get("y_percent"), 0.0) / 100.0) * height),
                "w": max(4, int((safe_float(proposal.get("width_percent"), 0.0) / 100.0) * width)),
                "h": max(4, int((safe_float(proposal.get("height_percent"), 0.0) / 100.0) * height)),
                "confidence": int(proposal.get("confidence", int(clamp(confidence) * 100))),
                "description": proposal.get("description", "Suspicious field region"),
                "source": proposal.get("source", "expert"),
            }
            if any(self._iou(rect, existing) > 0.4 for existing in regions):
                continue
            regions.append(rect)

        regions = sorted(regions, key=lambda item: item.get("confidence", 0), reverse=True)[:6]
        normalized = []
        for region in regions:
            normalized.append(
                {
                    "description": region["description"],
                    "x_percent": round((region["x"] / width) * 100, 2),
                    "y_percent": round((region["y"] / height) * 100, 2),
                    "width_percent": round((region["w"] / width) * 100, 2),
                    "height_percent": round((region["h"] / height) * 100, 2),
                    "confidence": int(region["confidence"]),
                    "source": region.get("source", "expert"),
                }
            )
        return normalized


class LightGBMFusion:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.metadata_path = model_path.replace(".txt", "_meta.json")
        self.use_weighted = not os.path.exists(self.model_path)
        self.feature_names = list(FEATURE_NAMES)
        self.weights = {
            "ela": 0.16,
            "visual": 0.10,
            "layout": 0.09,
            "ocr": 0.11,
            "font_gmm": 0.08,
            "noiseprint": 0.06,
            "copy_move": 0.15,
            "semantic": 0.12,
            "offline_llm": 0.10,
            "texture": 0.09,
            "text_perp": 0.02,
        }
        self.threshold = 0.62
        self.suspicious_threshold = 0.42
        self.metadata = {}
        self.distribution_stats = {}
        self.minimum_auc = safe_float(os.environ.get("FORGESHIELD_MIN_FUSION_AUC", 0.6), 0.6)
        self.minimum_f1 = safe_float(os.environ.get("FORGESHIELD_MIN_FUSION_F1", 0.6), 0.6)
        self.force_trained = env_flag("FORGESHIELD_FORCE_TRAINED_FUSION", False)

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
                self.threshold = safe_float(self.metadata.get("threshold", self.threshold), self.threshold)
                self.suspicious_threshold = safe_float(
                    self.metadata.get("suspicious_threshold", self.suspicious_threshold),
                    self.suspicious_threshold,
                )
                metadata_features = self.metadata.get("feature_names")
                if metadata_features:
                    self.feature_names = metadata_features
                self.distribution_stats = self.metadata.get("distribution_stats", {}) or {}
            except Exception as exc:
                print(f"[detector] Could not load fusion metadata: {exc}")

        if self.use_weighted:
            self.model = None
            print(f"[detector] LightGBM model not found at {self.model_path}; using weighted fusion.")
            return

        try:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=self.model_path)
            metrics = self.metadata.get("metrics", {}) if isinstance(self.metadata, dict) else {}
            auc = safe_float(metrics.get("auc", 0.0), 0.0)
            f1 = safe_float(metrics.get("f1", 0.0), 0.0)

            if not self.force_trained and metrics and (auc < self.minimum_auc or f1 < self.minimum_f1):
                self.model = None
                self.use_weighted = True
                print(
                    f"[detector] Trained LightGBM at {self.model_path} did not meet quality gates "
                    f"(AUC={auc:.3f}, F1={f1:.3f}); using weighted fusion instead."
                )
                return

            print(f"[detector] Using LightGBM model from {self.model_path}.")
        except Exception as exc:
            self.model = None
            self.use_weighted = True
            print(f"[detector] LightGBM load failed, using weighted fusion: {exc}")

    def _effective_weights(self, component_status=None):
        effective = {name: self.weights.get(name, 0.05) for name in self.feature_names}
        if not component_status:
            return effective

        for name in self.feature_names:
            status = component_status.get(name, {})
            if status.get("disabled"):
                effective[name] = 0.0
            elif status.get("status") == "fallback" or status.get("fallback"):
                effective[name] *= 0.55
        return effective

    def predict(self, features_dict, component_status=None):
        if self.use_weighted or self.model is None:
            effective = self._effective_weights(component_status)
            total_weight = sum(effective.values()) or 1.0
            weighted = sum(effective[name] * safe_float(features_dict.get(name, 0.5), 0.5) for name in self.feature_names)
            return clamp(weighted / total_weight)

        values = [safe_float(features_dict.get(name, 0.5), 0.5) for name in self.feature_names]
        array = np.array([values], dtype=np.float32)
        return clamp(self.model.predict(array)[0])

    def reliability(self, features_dict):
        metrics = self.metadata.get("metrics", {}) if isinstance(self.metadata, dict) else {}
        reliability = {
            "fusion_mode": "weighted" if self.use_weighted or self.model is None else "lightgbm",
            "quality_gated": bool(self.use_weighted and os.path.exists(self.model_path)),
            "trained_auc": safe_float(metrics.get("auc", 0.0), 0.0),
            "trained_f1": safe_float(metrics.get("f1", 0.0), 0.0),
            "out_of_distribution": False,
            "distribution_shift_score": 0.0,
            "distribution_shift_distance": 0.0,
            "distribution_shift_threshold": 0.0,
        }

        mean_map = self.distribution_stats.get("feature_mean", {})
        std_map = self.distribution_stats.get("feature_std", {})
        if not mean_map or not std_map:
            return reliability

        z_values = []
        for name in self.feature_names:
            mean_value = safe_float(mean_map.get(name), 0.5)
            std_value = max(safe_float(std_map.get(name), 0.1), 1e-4)
            current_value = safe_float(features_dict.get(name, mean_value), mean_value)
            z_values.append(abs(current_value - mean_value) / std_value)

        if not z_values:
            return reliability

        distance = float(np.mean(z_values))
        mean_distance = safe_float(self.distribution_stats.get("distance_mean", 0.0), 0.0)
        std_distance = max(safe_float(self.distribution_stats.get("distance_std", 0.2), 0.2), 1e-4)
        threshold = max(
            safe_float(self.distribution_stats.get("distance_p95", 0.0), 0.0),
            mean_distance + (2.5 * std_distance),
        )
        shift_score = clamp((distance - mean_distance) / (3.0 * std_distance))

        reliability.update(
            {
                "out_of_distribution": bool(distance > threshold),
                "distribution_shift_score": shift_score,
                "distribution_shift_distance": distance,
                "distribution_shift_threshold": threshold,
            }
        )
        return reliability


class EnhancedForgeryDetector:
    def __init__(self, regional_lang="en", use_gpu=True):
        has_cuda = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if has_cuda else "cpu")
        self.ocr_langs = sorted({regional_lang, "en"})
        self.ocr_backend = OCRBackend(self.ocr_langs, use_gpu=has_cuda)
        self.tesseract_ok = tesseract_available()
        self.layout_enabled = auto_flag("FORGESHIELD_ENABLE_LAYOUTLM", self.device.type, enabled_on_gpu=True, enabled_on_cpu=False)
        self.layout_enabled = self.layout_enabled and self.tesseract_ok

        self.layout_processor = None
        self.layout_model = None
        if self.layout_enabled:
            try:
                self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
                self.layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(self.device)
                self.layout_model.eval()
                print("[detector] LayoutLMv3 ready.")
            except Exception as exc:
                print(f"[detector] LayoutLMv3 fallback enabled: {exc}")
                self.layout_enabled = False
        elif not self.tesseract_ok:
            print("[detector] LayoutLMv3 fallback enabled: Tesseract is unavailable.")
        else:
            print("[detector] LayoutLMv3 fallback enabled: disabled for CPU-friendly runtime. Set FORGESHIELD_ENABLE_LAYOUTLM=1 to force it.")

        self.visual_model = self._build_visual_model()
        self.vis_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.text_expert = TextForensicsExpert(self.device)
        self.offline_llm_expert = OfflineLLMExpert()
        self.font_expert = FontForensicsExpert(self.device)
        self.noiseprint_expert = NoiseprintExpert(device_type=self.device.type)
        self.pixel_localizer = PixelLocalizer()
        self.fusion = LightGBMFusion()

        print(f"[detector] Using device: {self.device}")

    def _build_visual_model(self):
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(backbone.children())[:-1])
        model.eval()
        model.to(self.device)
        return model

    def _ocr_data(self, image_path):
        return self.ocr_backend.extract(image_path)

    def _entry_to_region(self, entry, image_shape, description, confidence, source):
        if not entry or "bbox" not in entry:
            return None
        height, width = image_shape[:2]
        points = np.array(entry["bbox"], dtype=np.float32)
        xs = points[:, 0]
        ys = points[:, 1]
        x = max(0.0, float(np.min(xs)))
        y = max(0.0, float(np.min(ys)))
        box_width = max(4.0, float(np.max(xs) - np.min(xs)))
        box_height = max(4.0, float(np.max(ys) - np.min(ys)))
        return {
            "description": description,
            "x_percent": round((x / max(width, 1)) * 100, 2),
            "y_percent": round((y / max(height, 1)) * 100, 2),
            "width_percent": round((box_width / max(width, 1)) * 100, 2),
            "height_percent": round((box_height / max(height, 1)) * 100, 2),
            "confidence": int(clamp(confidence) * 100),
            "source": source,
        }

    def _merge_region_proposals(self, details):
        merged = []
        for name, component in details.items():
            for region in component.get("regions", []) or []:
                proposal = dict(region)
                proposal.setdefault("source", name)
                merged.append(proposal)
        return merged

    def _ocr_region_map(self, image_shape, regions):
        height, width = image_shape[:2]
        region_map = np.zeros((height, width), dtype=np.float32)
        for region in regions or []:
            x = int((safe_float(region.get("x_percent"), 0.0) / 100.0) * width)
            y = int((safe_float(region.get("y_percent"), 0.0) / 100.0) * height)
            box_width = max(4, int((safe_float(region.get("width_percent"), 0.0) / 100.0) * width))
            box_height = max(4, int((safe_float(region.get("height_percent"), 0.0) / 100.0) * height))
            confidence = safe_float(region.get("confidence"), 0.0) / 100.0
            x2 = min(width, x + box_width)
            y2 = min(height, y + box_height)
            region_map[y:y2, x:x2] = np.maximum(region_map[y:y2, x:x2], confidence)
        return cv2.GaussianBlur(region_map, (0, 0), 3.0)

    def _texture_evidence_map(self, gray):
        tile_size = 32
        evidence = np.zeros_like(gray, dtype=np.float32)
        for y in range(0, gray.shape[0] - tile_size + 1, tile_size):
            for x in range(0, gray.shape[1] - tile_size + 1, tile_size):
                tile = gray[y : y + tile_size, x : x + tile_size]
                blur = cv2.GaussianBlur(tile, (3, 3), 0)
                residual = cv2.absdiff(tile, blur)
                local_score = 0.55 * (float(np.mean(residual)) / 20.0) + 0.45 * (float(np.std(tile)) / 45.0)
                evidence[y : y + tile_size, x : x + tile_size] = clamp(local_score)
        return normalize_map(evidence)

    def _residual_evidence_map(self, gray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blur).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        combined = normalize_map(residual) * 0.6 + normalize_map(np.abs(laplacian)) * 0.4
        return normalize_map(combined)

    def _extract_region_crops(self, image, regions, flags=None, limit=3):
        if image is None:
            return []
        height, width = image.shape[:2]
        crops = []
        for index, region in enumerate((regions or [])[:limit], start=1):
            x = int((safe_float(region.get("x_percent"), 0.0) / 100.0) * width)
            y = int((safe_float(region.get("y_percent"), 0.0) / 100.0) * height)
            box_width = max(12, int((safe_float(region.get("width_percent"), 0.0) / 100.0) * width))
            box_height = max(12, int((safe_float(region.get("height_percent"), 0.0) / 100.0) * height))
            margin_x = max(10, box_width // 6)
            margin_y = max(10, box_height // 6)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + box_width + margin_x)
            y2 = min(height, y + box_height + margin_y)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (x - x1, y - y1), (min(crop.shape[1] - 1, x - x1 + box_width), min(crop.shape[0] - 1, y - y1 + box_height)), (40, 200, 255), 2)
            success, encoded = cv2.imencode(".png", crop)
            if not success:
                continue
            flag = (flags or [None] * limit)[index - 1] if flags else None
            crops.append(
                {
                    "id": f"crop_{index}",
                    "title": flag.get("category") if flag else region.get("description", f"Suspicious crop {index}"),
                    "description": region.get("description", "Suspicious crop"),
                    "confidence": int(region.get("confidence", 0)),
                    "image_base64": base64.b64encode(encoded.tobytes()).decode("utf-8"),
                }
            )
        return crops

    def ela_analysis(self, image_path, quality_levels=(68, 82, 94)):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        maps = []
        for quality in quality_levels:
            _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            decoded = cv2.imdecode(encoded, 1)
            diff = cv2.absdiff(image, decoded)
            maps.append(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32))

        ela_map = np.mean(maps, axis=0)
        normalized = cv2.normalize(ela_map, None, 0, 255, cv2.NORM_MINMAX)
        mean_score = float(np.mean(normalized) / 255.0)
        peak_score = float(np.percentile(normalized, 95) / 255.0)
        ela_score = clamp(0.65 * mean_score + 0.35 * peak_score)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return ela_score, heatmap, normalized

    def _layout_heuristic(self, ocr_data, image_shape):
        entries = ocr_data.get("entries", [])
        if len(entries) < 2:
            return 0.25, {"box_count": len(entries), "fallback": True}

        image_height, image_width = image_shape[:2]
        left_positions = []
        heights = []
        baselines = []
        widths = []
        for entry in entries:
            points = np.array(entry["bbox"], dtype=np.float32)
            xs = points[:, 0]
            ys = points[:, 1]
            left_positions.append(float(np.min(xs) / max(image_width, 1)))
            heights.append(float(np.max(ys) - np.min(ys)))
            widths.append(float(np.max(xs) - np.min(xs)))
            baselines.append(float(np.mean(ys) / max(image_height, 1)))

        score = clamp(
            0.30 * (np.std(left_positions) / (np.mean(left_positions) + 1e-6))
            + 0.35 * (np.std(heights) / (np.mean(heights) + 1e-6))
            + 0.20 * (np.std(widths) / (np.mean(widths) + 1e-6))
            + 0.15 * (np.std(baselines) * 3.0)
        )
        return score, {"box_count": len(entries), "fallback": True}

    def layout_analysis(self, image_path, ocr_data=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        ocr_data = ocr_data or self._ocr_data(image_path)
        heuristic_score, heuristic_details = self._layout_heuristic(ocr_data, image.shape)
        if self.layout_processor is None or self.layout_model is None:
            return heuristic_score, {**heuristic_details, "disabled": not self.layout_enabled}

        try:
            rgb_image = Image.open(image_path).convert("RGB")
            encoding = self.layout_processor(rgb_image, return_tensors="pt", truncation=True)
            encoding = {key: value.to(self.device) for key, value in encoding.items()}
            with torch.no_grad():
                outputs = self.layout_model(**encoding)
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            transformer_score = clamp(np.std(pooled) / (np.mean(np.abs(pooled)) + 1e-6) / 2.5)
            score = clamp(0.65 * transformer_score + 0.35 * heuristic_score)
            return score, {"box_count": heuristic_details["box_count"], "fallback": False, "disabled": False}
        except Exception as exc:
            return heuristic_score, {**heuristic_details, "error": str(exc), "disabled": False}

    def _blockiness_score(self, gray):
        horizontal = np.abs(gray[:, 8::8] - gray[:, 7::8])
        vertical = np.abs(gray[8::8, :] - gray[7::8, :])
        boundary_mean = 0.0
        if horizontal.size:
            boundary_mean += float(np.mean(horizontal))
        if vertical.size:
            boundary_mean += float(np.mean(vertical))
        return clamp(boundary_mean / 32.0)

    def visual_analysis(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.vis_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.visual_model(tensor).cpu().numpy().flatten()

        feature_score = clamp(np.std(features) / (np.mean(np.abs(features)) + 1e-6) / 10.0)
        gray = np.array(image.convert("L"), dtype=np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edge_score = clamp(float(np.std(laplacian)) / 80.0)
        blockiness = self._blockiness_score(gray)
        score = clamp(0.35 * feature_score + 0.30 * edge_score + 0.35 * blockiness)
        return score, {"feature_score": feature_score, "edge_score": edge_score, "blockiness": blockiness}

    def ocr_consistency_analysis(self, image_path, ocr_data=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        ocr_data = ocr_data or self._ocr_data(image_path)
        entries = ocr_data.get("entries", [])
        if not entries:
            return 0.2, {"tokens": 0, "mean_confidence": 0.0, "fallback": True}

        heights = []
        widths = []
        confidences = []
        centers = []
        digit_low_conf = 0
        suspicious_regions = []

        for entry in entries:
            points = np.array(entry["bbox"], dtype=np.float32)
            xs = points[:, 0]
            ys = points[:, 1]
            height = float(np.max(ys) - np.min(ys))
            width = float(np.max(xs) - np.min(xs))
            heights.append(height)
            widths.append(width)
            confidences.append(float(entry["confidence"]))
            centers.append(float(np.mean(ys) / max(image.shape[0], 1)))
            if any(char.isdigit() for char in entry["text"]) and entry["confidence"] < 0.45:
                digit_low_conf += 1

        height_cv = np.std(heights) / (np.mean(heights) + 1e-6)
        width_cv = np.std(widths) / (np.mean(widths) + 1e-6)
        baseline_jitter = np.std(centers) * 2.5
        low_conf_ratio = float(np.mean([confidence < 0.6 for confidence in confidences]))
        mean_confidence = float(np.mean(confidences))
        height_mean = np.mean(heights) + 1e-6
        width_mean = np.mean(widths) + 1e-6
        height_std = np.std(heights) + 1e-6
        width_std = np.std(widths) + 1e-6

        for index, entry in enumerate(entries):
            anomaly = clamp(
                0.45 * (1.0 - confidences[index])
                + 0.20 * min(abs(heights[index] - height_mean) / (2.5 * height_std), 1.0)
                + 0.15 * min(abs(widths[index] - width_mean) / (2.5 * width_std), 1.0)
                + 0.20 * float(any(char.isdigit() for char in entry["text"]) and confidences[index] < 0.55)
            )
            if anomaly >= 0.45:
                region = self._entry_to_region(
                    entry,
                    image.shape,
                    f"OCR anomaly around '{(entry['text'] or '').strip()[:26]}'",
                    anomaly,
                    "ocr",
                )
                if region:
                    suspicious_regions.append(region)

        score = clamp(
            0.25 * height_cv
            + 0.15 * width_cv
            + 0.20 * baseline_jitter
            + 0.20 * low_conf_ratio
            + 0.20 * (digit_low_conf / max(len(entries), 1))
        )
        return score, {
            "tokens": len(entries),
            "mean_confidence": mean_confidence,
            "low_conf_ratio": low_conf_ratio,
            "digit_low_conf": digit_low_conf,
            "suspicious_box_count": len(suspicious_regions),
            "regions": suspicious_regions[:6],
            "fallback": False,
        }

    def copy_move_analysis(self, image_path):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read image: {image_path}")

        orb = cv2.ORB_create(nfeatures=600)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 12:
            return 0.15, {"match_count": 0, "fallback": True}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(descriptors, descriptors, k=2)
        valid_matches = 0
        strong_distances = []
        suspicious_regions = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.queryIdx == first.trainIdx:
                continue
            if first.distance >= 0.75 * second.distance:
                continue

            p1 = np.array(keypoints[first.queryIdx].pt)
            p2 = np.array(keypoints[first.trainIdx].pt)
            spatial_distance = np.linalg.norm(p1 - p2)
            if spatial_distance < 32:
                continue
            valid_matches += 1
            strong_distances.append(first.distance)
            if len(suspicious_regions) < 8:
                for point in (p1, p2):
                    x = max(0.0, point[0] - 28.0)
                    y = max(0.0, point[1] - 20.0)
                    suspicious_regions.append(
                        {
                            "description": "Repeated visual pattern match",
                            "x_percent": round((x / max(gray.shape[1], 1)) * 100, 2),
                            "y_percent": round((y / max(gray.shape[0], 1)) * 100, 2),
                            "width_percent": round((56.0 / max(gray.shape[1], 1)) * 100, 2),
                            "height_percent": round((40.0 / max(gray.shape[0], 1)) * 100, 2),
                            "confidence": int(clamp(1.0 - first.distance / 80.0) * 100),
                            "source": "copy_move",
                        }
                    )

        if not strong_distances:
            return 0.15, {"match_count": 0, "regions": [], "fallback": False}

        score = clamp(0.55 * (valid_matches / 45.0) + 0.45 * (1.0 - np.mean(strong_distances) / 80.0))
        return score, {"match_count": int(valid_matches), "regions": suspicious_regions[:6], "fallback": False}

    def texture_analysis(self, image_path):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read image: {image_path}")

        tile_size = 32
        residual_means = []
        texture_stds = []
        tile_meta = []
        for y in range(0, gray.shape[0] - tile_size + 1, tile_size):
            for x in range(0, gray.shape[1] - tile_size + 1, tile_size):
                tile = gray[y : y + tile_size, x : x + tile_size]
                blur = cv2.GaussianBlur(tile, (3, 3), 0)
                residual = cv2.absdiff(tile, blur)
                residual_means.append(float(np.mean(residual)))
                texture_stds.append(float(np.std(tile)))
                tile_meta.append((x, y))

        if not residual_means:
            return 0.2, {"tile_count": 0, "regions": [], "fallback": True}

        residual_means = np.array(residual_means, dtype=np.float32)
        texture_stds = np.array(texture_stds, dtype=np.float32)
        residual_z = np.abs((residual_means - np.mean(residual_means)) / (np.std(residual_means) + 1e-6))
        texture_z = np.abs((texture_stds - np.mean(texture_stds)) / (np.std(texture_stds) + 1e-6))
        outlier_fraction = float(np.mean((residual_z > 2.0) | (texture_z > 2.0)))
        score = clamp(0.6 * outlier_fraction + 0.4 * (np.std(residual_means) / (np.mean(residual_means) + 1e-6)))
        tile_scores = 0.6 * normalize_map(residual_z) + 0.4 * normalize_map(texture_z)
        suspicious_regions = []
        for index in np.argsort(tile_scores)[::-1][:6]:
            if tile_scores[index] < 0.55:
                continue
            x, y = tile_meta[index]
            suspicious_regions.append(
                {
                    "description": "Texture outlier tile",
                    "x_percent": round((x / max(gray.shape[1], 1)) * 100, 2),
                    "y_percent": round((y / max(gray.shape[0], 1)) * 100, 2),
                    "width_percent": round((tile_size / max(gray.shape[1], 1)) * 100, 2),
                    "height_percent": round((tile_size / max(gray.shape[0], 1)) * 100, 2),
                    "confidence": int(clamp(tile_scores[index]) * 100),
                    "source": "texture",
                }
            )
        return score, {
            "tile_count": int(len(residual_means)),
            "outlier_fraction": outlier_fraction,
            "regions": suspicious_regions,
            "fallback": False,
        }

    def semantic_analysis(self, image_path, ocr_data=None, document_type="unknown"):
        ocr_data = ocr_data or self._ocr_data(image_path)
        text = (ocr_data.get("text") or "").strip()
        entries = ocr_data.get("entries", [])

        invalid_dates = 0
        impossible_percentages = 0
        duplicate_serials = 0
        suspicious_numeric_tokens = 0
        inconsistent_totals = 0
        suspicious_regions = []
        image = cv2.imread(image_path)
        image_shape = image.shape if image is not None else (1000, 1000, 3)

        date_matches = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
        for date_text in date_matches:
            parsed = False
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%m/%d/%Y", "%m-%d-%Y"):
                try:
                    datetime.strptime(date_text, fmt)
                    parsed = True
                    break
                except Exception:
                    continue
            if not parsed:
                invalid_dates += 1

        for match in re.findall(r"\b\d{1,3}(?:\.\d+)?\s*%", text):
            value = safe_float(match.replace("%", "").strip(), 0.0)
            if value > 100.0:
                impossible_percentages += 1

        serial_candidates = re.findall(r"\b[A-Z0-9]{8,}\b", text.upper())
        serial_counts = Counter(serial_candidates)
        duplicate_serials = sum(1 for token, count in serial_counts.items() if count > 1)

        numeric_keywords = {"total", "marks", "score", "roll", "date", "dob", "no", "id"}
        for entry in entries:
            lowered = (entry["text"] or "").lower()
            if entry["confidence"] < 0.45 and (any(keyword in lowered for keyword in numeric_keywords) or any(char.isdigit() for char in lowered)):
                suspicious_numeric_tokens += 1
                region = self._entry_to_region(
                    entry,
                    image_shape,
                    f"Low-confidence numeric field '{lowered[:28]}'",
                    0.62,
                    "semantic",
                )
                if region:
                    suspicious_regions.append(region)

        if document_type == "marksheet":
            percent_values = [safe_float(match.replace("%", "").strip(), 0.0) for match in re.findall(r"\b\d{1,3}(?:\.\d+)?\s*%", text)]
            if percent_values and any(value > 100.0 for value in percent_values):
                inconsistent_totals += 1
            numeric_lines = []
            for entry in entries:
                token = (entry.get("text") or "").strip()
                if re.fullmatch(r"\d{1,3}", token):
                    numeric_lines.append((int(token), entry))
            if len(numeric_lines) >= 3:
                max_number = max(value for value, _ in numeric_lines)
                if max_number > 1000:
                    inconsistent_totals += 1
                    for _, entry in numeric_lines[:2]:
                        region = self._entry_to_region(entry, image_shape, "Unusually large marks value", 0.68, "semantic")
                        if region:
                            suspicious_regions.append(region)

        for entry in entries:
            token = (entry.get("text") or "").strip()
            if token in date_matches:
                region = self._entry_to_region(entry, image_shape, f"Date field '{token}' needs verification", 0.58, "semantic")
                if region:
                    suspicious_regions.append(region)

        marksheet_bias = 0.1 if document_type == "marksheet" and (impossible_percentages or suspicious_numeric_tokens) else 0.0
        score = clamp(
            0.30 * min(invalid_dates, 3) / 3.0
            + 0.25 * min(impossible_percentages, 3) / 3.0
            + 0.20 * min(duplicate_serials, 3) / 3.0
            + 0.25 * min(suspicious_numeric_tokens, 4) / 4.0
            + 0.15 * min(inconsistent_totals, 2) / 2.0
            + marksheet_bias
        )

        return score, {
            "invalid_dates": int(invalid_dates),
            "impossible_percentages": int(impossible_percentages),
            "duplicate_serials": int(duplicate_serials),
            "suspicious_numeric_tokens": int(suspicious_numeric_tokens),
            "inconsistent_totals": int(inconsistent_totals),
            "regions": suspicious_regions[:6],
            "fallback": False,
        }

    def get_full_text(self, image_path, ocr_data=None):
        ocr_data = ocr_data or self._ocr_data(image_path)
        return ocr_data.get("text", "")

    def _run_expert(self, label, fn, default_score=0.5, default_details=None):
        default_details = default_details or {}
        try:
            score, details = fn()
            return clamp(score), {**default_details, **details, "status": "ok"}
        except Exception as exc:
            print(f"[detector] {label} fallback: {exc}")
            return default_score, {**default_details, "status": "fallback", "error": str(exc)}

    def _risk_level(self, final_score):
        if final_score >= 0.75:
            return "HIGH"
        if final_score >= 0.45:
            return "MEDIUM"
        return "LOW"

    def _document_type(self, text):
        lowered = (text or "").lower()
        mapping = [
            ("certificate", "certificate"),
            ("marksheet", "marksheet"),
            ("mark sheet", "marksheet"),
            ("admit", "admit card"),
            ("identity", "ID card"),
            ("id card", "ID card"),
            ("government", "government record"),
            ("letter", "letter"),
            ("bonafide", "bonafide"),
        ]
        for needle, label in mapping:
            if needle in lowered:
                return label
        return "unknown"

    def _dominant_experts(self, scores):
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [name for name, value in ranked if value >= 0.42][:4]

    def _explanations(self, scores, details):
        return {
            "ela": f"ELA anomaly score {scores['ela']:.2f} from compression inconsistency analysis.",
            "visual": f"Visual score {scores['visual']:.2f} from CNN texture spread, edges, and JPEG blockiness.",
            "layout": f"Layout score {scores['layout']:.2f} from OCR geometry and document structure.",
            "ocr": f"OCR score {scores['ocr']:.2f}; mean confidence {details['ocr'].get('mean_confidence', 0.0):.2f}.",
            "font_gmm": f"Font consistency score {scores['font_gmm']:.2f} across {details['font_gmm'].get('num_patches', 0)} text patches.",
            "noiseprint": "Noiseprint fallback used." if details["noiseprint"].get("fallback") else "Noise residual analysis detected inconsistent camera noise.",
            "copy_move": f"Copy-move score {scores['copy_move']:.2f}; repeated-region matches {details['copy_move'].get('match_count', 0)}.",
            "semantic": f"Semantic consistency score {scores['semantic']:.2f}; invalid dates {details['semantic'].get('invalid_dates', 0)}, numeric anomalies {details['semantic'].get('suspicious_numeric_tokens', 0)}.",
            "offline_llm": (
                "Offline LLM audit disabled or unavailable."
                if details["offline_llm"].get("disabled") or details["offline_llm"].get("fallback")
                else f"Offline LLM audit flagged {len(details['offline_llm'].get('issues', []))} semantic issues."
            ),
            "texture": f"Texture consistency score {scores['texture']:.2f}; outlier tiles {details['texture'].get('outlier_fraction', 0.0):.2f}.",
            "text_perp": (
                "Text LM disabled or unavailable."
                if details["text_perp"].get("perplexity") is None
                else f"Text perplexity {details['text_perp']['perplexity']:.1f}."
            ),
        }

    def _flags(self, final_score, scores, explanations, regions):
        category_map = {
            "ela": "IMAGE_ARTIFACT",
            "visual": "IMAGE_ARTIFACT",
            "layout": "LAYOUT_ANOMALY",
            "ocr": "TEXT_TAMPERING",
            "font_gmm": "FONT_INCONSISTENCY",
            "noiseprint": "IMAGE_ARTIFACT",
            "copy_move": "IMAGE_ARTIFACT",
            "semantic": "METADATA",
            "offline_llm": "METADATA",
            "texture": "IMAGE_ARTIFACT",
            "text_perp": "METADATA",
        }

        dominant = self._dominant_experts(scores)
        if not regions and final_score >= 0.45:
            regions = [
                {
                    "description": "Whole document review region",
                    "x_percent": 5.0,
                    "y_percent": 5.0,
                    "width_percent": 90.0,
                    "height_percent": 90.0,
                    "confidence": int(final_score * 100),
                }
            ]

        flags = []
        for index, region in enumerate(regions[: max(1, len(dominant))]):
            expert_name = dominant[min(index, len(dominant) - 1)] if dominant else "ela"
            confidence = max(region.get("confidence", 0), int(scores[expert_name] * 100))
            severity = "HIGH" if confidence >= 75 else "MEDIUM" if confidence >= 45 else "LOW"
            flags.append(
                {
                    "id": f"python_flag_{index + 1}",
                    "category": category_map.get(expert_name, "IMAGE_ARTIFACT"),
                    "description": explanations[expert_name],
                    "severity": severity,
                    "confidence": int(confidence),
                    "region": {
                        "description": region["description"],
                        "x_percent": region["x_percent"],
                        "y_percent": region["y_percent"],
                        "width_percent": region["width_percent"],
                        "height_percent": region["height_percent"],
                    },
                }
            )
        return flags

    def _recommendations(self, final_score, scores, details, reliability=None):
        dominant = self._dominant_experts(scores)
        recommendations = []
        if "semantic" in dominant or "ocr" in dominant:
            recommendations.append("Verify dates, identifiers, and numeric fields against the issuer's official records.")
        if "offline_llm" in dominant:
            recommendations.append("Review the OCR text and extracted fields manually, because the offline semantic audit found content-level inconsistencies.")
        if "layout" in dominant or "font_gmm" in dominant:
            recommendations.append("Compare alignment, font appearance, and spacing against a genuine sample of the same document type.")
        if "ela" in dominant or "visual" in dominant or "texture" in dominant or "copy_move" in dominant:
            recommendations.append("Inspect highlighted areas for pasted regions, duplicated content, and inconsistent compression or texture.")
        if details["noiseprint"].get("fallback"):
            recommendations.append("Noiseprint is running in fallback mode, so a manual image-forensics review is still recommended.")
        if reliability and reliability.get("out_of_distribution"):
            recommendations.append("This document looks unlike the training distribution, so treat the score as lower-confidence and review it against a matched genuine sample.")
        if final_score < 0.45:
            recommendations.append("No strong forgery signal was detected, but high-stakes documents should still be manually reviewed.")

        deduped = []
        for recommendation in recommendations:
            if recommendation not in deduped:
                deduped.append(recommendation)
        return deduped[:4]

    def _summary(self, verdict, final_score, scores, details, reliability=None):
        dominant = self._dominant_experts(scores)
        readable = ", ".join(name.replace("_", " ") for name in dominant) if dominant else "no dominant signals"
        summary = f"Python detector classified the document as {verdict} with {int(final_score * 100)}% confidence."
        summary += f" Strongest signals came from {readable}."
        if details["noiseprint"].get("fallback"):
            summary += " Noise residual analysis used a fallback path because the dedicated Noiseprint runtime was unavailable."
        if details["offline_llm"].get("issues"):
            summary += f" Offline semantic audit highlighted {len(details['offline_llm'].get('issues', []))} content-level concerns."
        if reliability and reliability.get("out_of_distribution"):
            summary += " The feature profile is outside the training distribution, so this result should be treated cautiously on unseen document styles."
        return summary

    def detect(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        ocr_data = self._ocr_data(image_path)
        full_text = self.get_full_text(image_path, ocr_data)
        document_type = self._document_type(full_text)
        details = {}

        print("Running ELA...")
        try:
            ela_score, ela_heatmap, ela_raw = self.ela_analysis(image_path)
            details["ela"] = {"status": "ok"}
        except Exception as exc:
            ela_score = 0.5
            ela_heatmap = np.zeros_like(image)
            ela_raw = np.zeros(image.shape[:2], dtype=np.float32)
            details["ela"] = {"status": "fallback", "error": str(exc)}

        print("Running Visual CNN...")
        visual_score, details["visual"] = self._run_expert("Visual CNN", lambda: self.visual_analysis(image_path))

        print("Running LayoutLMv3...")
        layout_score, details["layout"] = self._run_expert("LayoutLMv3", lambda: self.layout_analysis(image_path, ocr_data))

        print("Running OCR consistency...")
        ocr_score, details["ocr"] = self._run_expert("OCR consistency", lambda: self.ocr_consistency_analysis(image_path, ocr_data))

        print("Running Font Forensics...")
        font_score, details["font_gmm"] = self._run_expert("Font Forensics", lambda: self.font_expert.analyze(image_path))

        print("Running Noiseprint++...")
        noise_score, details["noiseprint"] = self._run_expert("Noiseprint", lambda: self.noiseprint_expert.analyze(image_path))

        print("Running Copy-Move analysis...")
        copy_move_score, details["copy_move"] = self._run_expert("Copy-Move", lambda: self.copy_move_analysis(image_path))

        print("Running Semantic checks...")
        semantic_score, details["semantic"] = self._run_expert(
            "Semantic checks",
            lambda: self.semantic_analysis(image_path, ocr_data, document_type),
        )

        print("Running Offline LLM audit...")
        offline_llm_score, details["offline_llm"] = self._run_expert(
            "Offline LLM audit",
            lambda: self.offline_llm_expert.analyze(full_text, document_type),
            default_score=0.2,
        )

        print("Running Texture analysis...")
        texture_score, details["texture"] = self._run_expert("Texture analysis", lambda: self.texture_analysis(image_path))

        print("Running Text Forensics...")
        text_score, details["text_perp"] = self._run_expert("Text Forensics", lambda: self.text_expert.analyze(full_text))

        scores = {
            "ela": ela_score,
            "visual": visual_score,
            "layout": layout_score,
            "ocr": ocr_score,
            "font_gmm": font_score,
            "noiseprint": noise_score,
            "copy_move": copy_move_score,
            "semantic": semantic_score,
            "offline_llm": offline_llm_score,
            "texture": texture_score,
            "text_perp": text_score,
        }

        final_score = self.fusion.predict(scores, details)
        reliability = self.fusion.reliability(scores)
        verdict = (
            "FORGED"
            if final_score >= self.fusion.threshold
            else "SUSPICIOUS"
            if final_score >= self.fusion.suspicious_threshold
            else "GENUINE"
        )
        risk_level = self._risk_level(final_score)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        region_proposals = self._merge_region_proposals(details)
        region_map = self._ocr_region_map(image.shape, region_proposals)
        texture_map = self._texture_evidence_map(gray)
        residual_map = self._residual_evidence_map(gray)
        evidence_map = self.pixel_localizer.fuse_maps(
            image.shape,
            [
                (ela_raw, 0.32),
                (texture_map, 0.20),
                (residual_map, 0.18),
                (region_map, 0.30),
            ],
            region_proposals=region_proposals,
        )
        mask = self.pixel_localizer.generate_mask(evidence_map)
        regions = self.pixel_localizer.extract_regions(
            mask,
            image.shape,
            final_score,
            region_proposals=region_proposals,
            evidence_map=evidence_map,
        )
        explanations = self._explanations(scores, details)
        flags = self._flags(final_score, scores, explanations, regions)
        region_crops = self._extract_region_crops(image, regions, flags)
        recommendations = self._recommendations(final_score, scores, details, reliability)
        summary = self._summary(verdict, final_score, scores, details, reliability)

        component_status = {}
        for name in FEATURE_NAMES:
            component_status[name] = {
                "score": clamp(scores[name]),
                **details[name],
            }

        results = {
            "verdict": verdict,
            "confidence": float(final_score),
            "risk_level": risk_level,
            "scores": scores,
            "explanations": explanations,
            "summary": summary,
            "flags": flags,
            "recommendations": recommendations,
            "document_type_detected": document_type,
            "component_status": component_status,
            "ocr_backend": ocr_data.get("backend"),
            "model_reliability": reliability,
            "suspicious_regions": regions,
            "region_crops": region_crops,
            "offline_llm_summary": details["offline_llm"].get("summary"),
        }

        visualization_base64 = self.create_visualization(image_path, ela_heatmap, mask, flags, results, evidence_map)
        return results, visualization_base64

    def create_visualization(self, image_path, ela_heatmap, mask, flags=None, results=None, evidence_map=None):
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        overlay_ela = cv2.addWeighted(original, 0.58, ela_heatmap, 0.42, 0)
        overlay_ela_rgb = cv2.cvtColor(overlay_ela, cv2.COLOR_BGR2RGB)

        evidence_map = normalize_map(evidence_map if evidence_map is not None else mask)
        evidence_overlay = cv2.applyColorMap((evidence_map * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        evidence_overlay = cv2.addWeighted(original, 0.52, evidence_overlay, 0.48, 0)
        evidence_overlay_rgb = cv2.cvtColor(evidence_overlay, cv2.COLOR_BGR2RGB)

        if mask is not None:
            contour_overlay = original.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_overlay, contours, -1, (255, 190, 40), 2)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
            mask_overlay = cv2.addWeighted(contour_overlay, 0.78, mask_color, 0.22, 0)
        else:
            mask_overlay = original.copy()

        region_overlay = original.copy()
        if flags:
            height, width = original.shape[:2]
            for flag in flags:
                region = flag.get("region") or {}
                x = int((region.get("x_percent", 0) / 100.0) * width)
                y = int((region.get("y_percent", 0) / 100.0) * height)
                box_width = int((region.get("width_percent", 0) / 100.0) * width)
                box_height = int((region.get("height_percent", 0) / 100.0) * height)
                severity = (flag.get("severity") or "LOW").upper()
                color = (220, 70, 70) if severity == "HIGH" else (245, 165, 55) if severity == "MEDIUM" else (70, 170, 70)
                cv2.rectangle(region_overlay, (x, y), (x + box_width, y + box_height), color, 2)
                label = f"{severity} {int(flag.get('confidence', 0))}%"
                cv2.putText(
                    region_overlay,
                    label,
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        mask_overlay_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
        region_overlay_rgb = cv2.cvtColor(region_overlay, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(17, 11), dpi=150)
        grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.92], height_ratios=[1.0, 1.0], hspace=0.18, wspace=0.14)

        ax_original = fig.add_subplot(grid[0, 0])
        ax_original.imshow(region_overlay_rgb)
        ax_original.set_title("Document With Flagged Regions", fontsize=13, fontweight="bold")
        ax_original.axis("off")

        ax_ela = fig.add_subplot(grid[0, 1])
        ax_ela.imshow(overlay_ela_rgb)
        ax_ela.set_title("ELA Compression Anomalies", fontsize=13, fontweight="bold")
        ax_ela.axis("off")

        ax_evidence = fig.add_subplot(grid[0, 2])
        ax_evidence.imshow(evidence_overlay_rgb)
        ax_evidence.set_title("Composite Evidence Map", fontsize=13, fontweight="bold")
        ax_evidence.axis("off")

        ax_mask = fig.add_subplot(grid[1, 0])
        ax_mask.imshow(mask_overlay_rgb)
        ax_mask.set_title("Localized Anomaly Mask", fontsize=13, fontweight="bold")
        ax_mask.axis("off")

        ax_panel = fig.add_subplot(grid[1, 1:])

        results = results or {}
        scores = results.get("scores", {})
        reliability = results.get("model_reliability", {})
        top_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:6]
        verdict = results.get("verdict", "UNKNOWN")
        confidence = int(100 * safe_float(results.get("confidence", 0.0), 0.0))
        risk_level = results.get("risk_level", "UNKNOWN")
        ood_note = "Yes" if reliability.get("out_of_distribution") else "No"
        fusion_mode = reliability.get("fusion_mode", "unknown")

        verdict_color = "#b42318" if verdict == "FORGED" else "#b54708" if verdict == "SUSPICIOUS" else "#027a48"
        ax_panel.text(
            0.0,
            1.02,
            f"{verdict}  |  {confidence}% confidence  |  Risk {risk_level}",
            fontsize=16,
            fontweight="bold",
            color=verdict_color,
            transform=ax_panel.transAxes,
        )
        ax_panel.text(
            0.0,
            0.95,
            f"Fusion: {fusion_mode}   OCR: {results.get('ocr_backend', 'unknown')}   OOD warning: {ood_note}",
            fontsize=10.5,
            color="#344054",
            transform=ax_panel.transAxes,
        )

        if top_scores:
            labels = [name.replace("_", " ") for name, _ in reversed(top_scores)]
            values = [score for _, score in reversed(top_scores)]
            y_pos = np.arange(len(labels))
            bars = ax_panel.barh(y_pos, values, color=["#0ba5ec" if value < 0.45 else "#f79009" if value < 0.75 else "#f04438" for value in values], height=0.6)
            ax_panel.set_yticks(y_pos)
            ax_panel.set_yticklabels(labels, fontsize=10)
            ax_panel.set_xlim(0, 1)
            ax_panel.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax_panel.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=9)
            ax_panel.grid(axis="x", alpha=0.18)
            ax_panel.spines["top"].set_visible(False)
            ax_panel.spines["right"].set_visible(False)
            ax_panel.spines["left"].set_visible(False)
            for bar, value in zip(bars, values):
                ax_panel.text(min(value + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
        else:
            ax_panel.set_axis_off()

        detail_lines = []
        for flag in (flags or [])[:3]:
            detail_lines.append(
                f"- {flag.get('category', 'FLAG')}: {flag.get('description', '')[:110]}"
            )
        offline_summary = results.get("offline_llm_summary")
        if offline_summary:
            detail_lines.append(f"- Offline audit: {offline_summary[:120]}")
        if reliability.get("out_of_distribution"):
            distance = reliability.get("distribution_shift_distance", 0.0)
            threshold = reliability.get("distribution_shift_threshold", 0.0)
            detail_lines.append(f"- OOD warning: feature distance {distance:.2f} exceeded threshold {threshold:.2f}.")
        if not detail_lines:
            detail_lines.append("- No strong localized anomaly regions were found.")

        panel_text = fill("\n".join(detail_lines), width=58, break_long_words=False, replace_whitespace=False)
        ax_panel.text(0.0, -0.18, panel_text, fontsize=9.6, color="#101828", transform=ax_panel.transAxes, va="top")

        fig.suptitle("ForgeShield Forensic Dashboard", fontsize=18, fontweight="bold", y=0.98)
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64

    def generate_html_report(self, image_path, output_path="enhanced_report.html"):
        results, viz_b64 = self.detect(image_path)
        rows = "".join(
            f"<tr><td>{name}</td><td>{score:.2%}</td><td>{results['explanations'].get(name, '')}</td></tr>"
            for name, score in results["scores"].items()
        )
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <title>ForgeShield Python Detector Report</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; }}
            .verdict {{ font-size: 28px; font-weight: bold; padding: 18px; border-radius: 10px; }}
            .FORGED {{ background: #ffe0e0; color: #8b0000; }}
            .SUSPICIOUS {{ background: #fff1d6; color: #8a5a00; }}
            .GENUINE {{ background: #ddf7e6; color: #0b6c35; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #f5f5f5; }}
          </style>
        </head>
        <body>
          <h1>ForgeShield Python Detector Report</h1>
          <div class="verdict {results['verdict']}">Verdict: {results['verdict']}</div>
          <p>Confidence: <strong>{results['confidence']:.2%}</strong></p>
          <p>Risk: <strong>{results['risk_level']}</strong></p>
          <p>{results['summary']}</p>
          <table>
            <tr><th>Expert</th><th>Score</th><th>Explanation</th></tr>
            {rows}
          </table>
          <h2>Visualization</h2>
          <img src="data:image/png;base64,{viz_b64}" style="max-width:100%" />
        </body>
        </html>
        """
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return output_path


if __name__ == "__main__":
    detector = EnhancedForgeryDetector(regional_lang="en", use_gpu=torch.cuda.is_available())
    test_image = r"data\forged\genuine_002_layout_shift.png"
    report_path = detector.generate_html_report(test_image, "enhanced_unified_report.html")
    print(f"Report saved to {report_path}")
