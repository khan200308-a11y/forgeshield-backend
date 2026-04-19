# =============================================================================
# ENHANCED DOCUMENT FORGERY DETECTOR - FULLY CORRECTED
# Includes Tesseract path, GPT-2 fix, and all experts.
# =============================================================================

import os
import sys
import cv2
import numpy as np
from PIL import Image
import base64
import io
import warnings
warnings.filterwarnings("ignore")

# ------------------------- AUTO-INSTALL MISSING PACKAGES -----------------
def install_missing_packages():
    required = {
        'torch': 'torch torchvision',
        'transformers': 'transformers',
        'easyocr': 'easyocr',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'shap': 'shap',
        'tqdm': 'tqdm'
    }
    missing = []
    for pkg, install_name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install_name)
    if missing:
        print(f"Installing missing packages: {missing}")
        for m in missing:
            os.system(f"{sys.executable} -m pip install {m}")
        print("Installation done. Please restart kernel if in Jupyter.")

install_missing_packages()

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import LayoutLMv3Processor, LayoutLMv3Model, AutoModelForCausalLM, AutoTokenizer
import easyocr
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ------------------------- PATHS -------------------------
_DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
_NOISEPRINT_PATH = os.environ.get(
    "NOISEPRINT_PATH",
    os.path.join(_DETECTOR_DIR, "Noiseprint"),
)

# ------------------------- TESSERACT PATH -------------------------
import pytesseract
_TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
TESSERACT_OK = False
try:
    pytesseract.get_tesseract_version()
    TESSERACT_OK = True
    print("✅ Tesseract found.")
except Exception:
    print("⚠️ Tesseract not found. LayoutLMv3 will use fallback mode.")

# ------------------------- NOISEPRINT++ (optional) -----------------
try:
    sys.path.append(os.path.expanduser("~/Noiseprint"))
    from noiseprint import Noiseprint
    NOISEPRINT_AVAILABLE = True
except:
    NOISEPRINT_AVAILABLE = False
    print("Noiseprint++ not found. Using fallback.")

# ------------------------- TEXT FORENSICS (FIXED) -----------------
class TextForensicsExpert:
    def __init__(self):
        # Use AutoModelForCausalLM to get loss (perplexity)
        self.lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.lm_model.eval()
        if torch.cuda.is_available():
            self.lm_model = self.lm_model.cuda()
        print("Text forensics expert ready (GPT-2).")
    
    def get_perplexity(self, text):
        if not text.strip():
            return 100.0
        inputs = self.lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # now works
        perplexity = torch.exp(loss).item()
        return min(perplexity, 1000)
    
    def analyze(self, text):
        ppl = self.get_perplexity(text)
        ppl_score = min(1.0, ppl / 100.0)
        cnn_score = 0.5  # placeholder
        combined = 0.7 * ppl_score + 0.3 * cnn_score
        return combined, {"perplexity": ppl}

# ------------------------- FONT FORENSICS (ViT + GMM) -----------------
class FontForensicsExpert:
    def __init__(self):
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit.eval()
        if torch.cuda.is_available():
            self.vit = self.vit.cuda()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Dummy GMM (replace with real training)
        np.random.seed(42)
        dummy_features = np.random.randn(100, 768)
        self.scaler = StandardScaler().fit(dummy_features)
        self.gmm = GaussianMixture(n_components=3, random_state=42)
        self.gmm.fit(self.scaler.transform(dummy_features))
        print("Font forensics expert ready.")
    
    def extract_patches(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patches = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 10 and w*h < 5000:
                patches.append(image[y:y+h, x:x+w])
        return patches
    
    def get_vit_embedding(self, patch):
        pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        inp = self.transform(pil_patch).unsqueeze(0)
        if torch.cuda.is_available():
            inp = inp.cuda()
        with torch.no_grad():
            emb = self.vit(pixel_values=inp).last_hidden_state.mean(dim=1).cpu().numpy()
        return emb.flatten()
    
    def analyze(self, image_path):
        img = cv2.imread(image_path)
        patches = self.extract_patches(img)
        if len(patches) == 0:
            return 0.5, {"num_patches": 0}
        scores = []
        for patch in patches:
            feat = self.get_vit_embedding(patch)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            log_lik = self.gmm.score_samples(feat_scaled)[0]
            anomaly = 1.0 / (1.0 + np.exp(log_lik + 5))
            scores.append(anomaly)
        return np.mean(scores), {"num_patches": len(patches)}

# ------------------------- NOISEPRINT EXPERT -------------------------
import subprocess
import tempfile

import subprocess
import tempfile
import scipy.io as sio

class NoiseprintExpert:
    def __init__(self, noiseprint_path=r"./Noiseprint", tf1_env_python=None):
        self.noiseprint_path = os.path.abspath(noiseprint_path)
        self.python_path = tf1_env_python or r"D:\anaconda3\envs\noiseprint_env\python.exe"
        self.available = os.path.exists(self.python_path) and os.path.exists(os.path.join(self.noiseprint_path, "main_blind.py"))
        if not self.available:
            print(f"Noiseprint not available. Using fallback.")
        else:
            print("Noiseprint++ ready (TF1 subprocess).")
    
    def analyze(self, image_path):
        if not self.available:
            return 0.5, {"fallback": True}
        
        # Resize large images to avoid memory issues
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
            cv2.imwrite(temp_path, img)
            image_path = temp_path
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            out_path = tmp.name
        
        try:
            # Positional arguments: input_image output.mat
            cmd = [
                self.python_path,
                os.path.join(self.noiseprint_path, "main_blind.py"),
                image_path,
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            
            if result.returncode != 0:
                print(f"Noiseprint error: {result.stderr}")
                return 0.5, {}
            
            # Read .mat file
            mat = sio.loadmat(out_path)
            # The variable name varies; try common keys
            heatmap = None
            for key in ['heatmap', 'map', 'out', 'scoremap']:
                if key in mat:
                    heatmap = mat[key]
                    break
            if heatmap is None:
                return 0.5, {}
            
            heatmap = heatmap.astype(np.float32)
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            score = float(np.mean(heatmap_norm))
            return score, {"heatmap_path": out_path}
        
        except Exception as e:
            print(f"Noiseprint failed: {e}")
            return 0.5, {}
        finally:
            if cleanup_temp and os.path.exists(image_path):
                os.unlink(image_path)
            # Keep out_path for optional visualization? Or delete:
            # if os.path.exists(out_path): os.unlink(out_path)

# ------------------------- PIXEL LOCALIZER -------------------------
class PixelLocalizer:
    def generate_mask(self, image_path, ela_map):
        ela_norm = ela_map / 255.0
        mask = (ela_norm > 0.3).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

# ------------------------- LIGHTGBM FUSION -------------------------
class LightGBMFusion:
    def __init__(self, model_path="lightgbm_model.txt"):
        if os.path.exists(model_path):
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=model_path)
            self.use_weighted = False
            print(f"✅ Using trained LightGBM from {model_path}")
        else:
            # Fallback to weighted average
            self.use_weighted = True
            self.weights = {...}  # keep your fallback
            print("⚠️ Trained model not found, using weighted average fallback")
    
    def predict(self, features_dict):
        if not self.use_weighted:
            X = np.array([[features_dict.get(name, 0.0) for name in 
                          ['ela', 'visual', 'layout', 'ocr', 'text_perp', 'font_gmm', 'noiseprint']]])
            return self.model.predict(X)[0]
        else:
            # Weighted average fallback
            score = sum(self.weights[k] * features_dict.get(k, 0) for k in self.weights)
            return np.clip(score / sum(self.weights.values()), 0, 1)

# ------------------------- MAIN DETECTOR -------------------------
class EnhancedForgeryDetector:
    def __init__(self, regional_lang='en', use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.reader = easyocr.Reader([regional_lang, 'en'])
        
        # LayoutLMv3 (only if Tesseract works)
        self.layout_processor = None
        self.layout_model = None
        if TESSERACT_OK:
            try:
                self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
                self.layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(self.device)
                self.layout_model.eval()
                print("LayoutLMv3 loaded.")
            except Exception as e:
                print(f"LayoutLMv3 init failed: {e}")
        
        # Visual CNN
        self.visual_model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.visual_model.to(self.device)
        self.visual_model.eval()
        self.vis_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.text_expert = TextForensicsExpert()
        self.font_expert = FontForensicsExpert()
        self.pixel_localizer = PixelLocalizer()
        self.pixel_localizer = PixelLocalizer()
        self.noiseprint_expert = NoiseprintExpert(noiseprint_path=_NOISEPRINT_PATH)
        self.fusion = LightGBMFusion()
        self.pixel_localizer = PixelLocalizer()
    
    def ela_analysis(self, image_path, quality_levels=[75, 85, 95]):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        ela_maps = []
        for q in quality_levels:
            _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            resaved = cv2.imdecode(encimg, 1)
            diff = cv2.absdiff(img, resaved)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ela_maps.append(diff_gray)
        ela_map = np.mean(ela_maps, axis=0)
        ela_map = cv2.normalize(ela_map, None, 0, 255, cv2.NORM_MINMAX)
        ela_score = np.mean(ela_map) / 255.0
        heatmap = cv2.applyColorMap(ela_map.astype(np.uint8), cv2.COLORMAP_JET)
        return ela_score, heatmap, ela_map
    
    def layout_analysis(self, image_path):
        if self.layout_processor is None or self.layout_model is None:
            return 0.5, np.zeros(768)
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = self.layout_processor(image, return_tensors="pt", truncation=True)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = self.layout_model(**encoding)
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            layout_score = min(1.0, np.std(pooled) / 2.0)
            return layout_score, pooled
        except Exception as e:
            print(f"Layout analysis failed: {e}")
            return 0.5, np.zeros(768)
    
    def visual_analysis(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.vis_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.visual_model(input_tensor).cpu().numpy().flatten()
        entropy = -np.sum(features * np.log(np.abs(features) + 1e-8))
        visual_score = min(1.0, entropy / 10.0)
        return visual_score, features
    
    def ocr_consistency_analysis(self, image_path):
        result = self.reader.readtext(image_path, detail=1)
        if not result:
            return 0.0, []
        heights = []
        confidences = []
        for bbox, _, conf in result:
            pts = np.array(bbox, dtype=np.int32)
            h = np.linalg.norm(pts[1] - pts[0])
            heights.append(h)
            confidences.append(conf)
        if len(heights) < 2:
            return 0.0, []
        cv_height = np.std(heights) / (np.mean(heights) + 1e-5)
        low_conf_ratio = np.mean([c < 0.6 for c in confidences])
        ocr_score = min(1.0, (cv_height * 0.5 + low_conf_ratio * 0.5))
        return ocr_score, confidences
    
    def get_full_text(self, image_path):
        result = self.reader.readtext(image_path, detail=0)
        return " ".join(result)
    
    def detect(self, image_path):
        print("Running ELA...")
        ela_score, ela_heatmap, ela_raw = self.ela_analysis(image_path)
        print("Running Visual CNN...")
        visual_score, _ = self.visual_analysis(image_path)
        print("Running LayoutLMv3...")
        layout_score, _ = self.layout_analysis(image_path)
        print("Running OCR consistency...")
        ocr_score, _ = self.ocr_consistency_analysis(image_path)
        print("Running Text Forensics...")
        full_text = self.get_full_text(image_path)
        text_score, text_details = self.text_expert.analyze(full_text)
        print("Running Font Forensics...")
        font_score, font_details = self.font_expert.analyze(image_path)
        print("Running Noiseprint++...")
        noise_score, _ = self.noiseprint_expert.analyze(image_path)
        
        scores_dict = {
            'ela': ela_score, 'visual': visual_score, 'layout': layout_score,
            'ocr': ocr_score, 'text_perp': text_score, 'font_gmm': font_score,
            'noiseprint': noise_score
        }
        final_score = self.fusion.predict(scores_dict)
        verdict = "FORGED" if final_score > 0.4 else "GENUINE"
        mask = self.pixel_localizer.generate_mask(image_path, ela_raw)
        
        results = {
            "verdict": verdict, "confidence": float(final_score), "scores": scores_dict,
            "explanations": {
                "ELA": "JPEG compression inconsistencies.",
                "Visual CNN": "Unusual visual patterns.",
                "LayoutLMv3": "Abnormal layout/text arrangement.",
                "OCR": "Inconsistent font sizes/low confidence.",
                "Text Perplexity": f"Perplexity={text_details.get('perplexity',0):.1f}",
                "Font GMM": f"Anomaly from {font_details.get('num_patches',0)} patches.",
                "Noiseprint++": "Camera noise fingerprint anomalies."
            }
        }
        viz_base64 = self.create_visualization(image_path, ela_heatmap, mask)
        return results, viz_base64
    
    def create_visualization(self, image_path, ela_heatmap, mask):
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        overlay_ela = cv2.addWeighted(original, 0.5, ela_heatmap, 0.5, 0)
        overlay_ela_rgb = cv2.cvtColor(overlay_ela, cv2.COLOR_BGR2RGB)
        if mask is not None:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_overlay = cv2.addWeighted(original, 0.7, mask_3ch, 0.3, 0)
            mask_overlay_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
        else:
            mask_overlay_rgb = overlay_ela_rgb
        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        axes[0].imshow(original_rgb); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(overlay_ela_rgb); axes[1].set_title("ELA Heatmap"); axes[1].axis("off")
        axes[2].imshow(mask_overlay_rgb); axes[2].set_title("Tampering Mask"); axes[2].axis("off")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        return img_b64
    
    def generate_html_report(self, image_path, output_path="enhanced_report.html"):
        results, viz_b64 = self.detect(image_path)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Enhanced Forgery Detection Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .verdict {{ font-size: 28px; font-weight: bold; padding: 20px; border-radius: 10px; }}
            .FORGED {{ background-color: #ffcccc; color: #990000; }}
            .GENUINE {{ background-color: #ccffcc; color: #006600; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
        </head>
        <body>
        <h1>🔬 Enhanced Document Forgery Detection</h1>
        <div class="verdict {results['verdict']}">Verdict: {results['verdict']}</div>
        <p>Overall Confidence: <b>{results['confidence']:.2%}</b></p>
        <h2>📊 Expert Scores</h2>
        <table>
            <tr><th>Expert</th><th>Score</th><th>Explanation</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v:.2%}</td><td>{results["explanations"].get(k,"")}</td></tr>' for k,v in results['scores'].items())}
        </table>
        <h2>🖼️ Visual Analysis</h2>
        <img src="data:image/png;base64,{viz_b64}" style="max-width:100%">
        <p><i>Report generated by Enhanced Forgery Detector (fixed GPT-2)</i></p>
        </body>
        </html>
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Report saved to {output_path}")
        return output_path

# ------------------------- RUN -------------------------
if __name__ == "__main__":
    detector = EnhancedForgeryDetector(regional_lang='en')
    # CHANGE THIS TO YOUR ACTUAL IMAGE PATH
    test_image = r"data\forged\genuine_002_layout_shift.png"
    report = detector.generate_html_report(test_image, "enhanced_unified_report.html")
    print(f"Open {report} in browser.")