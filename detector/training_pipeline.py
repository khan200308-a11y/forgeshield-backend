# =============================================================================
# COMPLETE TRAINING PIPELINE FOR LIGHTGBM FUSION - FIXED
# Handles different image sizes in splicing
# =============================================================================

import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# Import your existing detector
from unified_detector import EnhancedForgeryDetector

# ------------------------- 1. FORGERY GENERATION FUNCTIONS -------------------------
class ForgeryGenerator:
    """Creates realistic document forgeries from genuine images."""
    
    def __init__(self, genuine_folder, output_folder):
        self.genuine_folder = genuine_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "genuine"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "forged"), exist_ok=True)
    
    def copy_paste_forgery(self, src_img, dst_img, output_path):
        """Copy a region from src_img and paste into dst_img."""
        h, w = dst_img.shape[:2]
        # Ensure patch size is valid
        patch_h = random.randint(30, min(150, h//3))
        patch_w = random.randint(30, min(150, w//3))
        src_h, src_w = src_img.shape[:2]
        if src_h < patch_h or src_w < patch_w:
            patch_h = min(src_h, patch_h)
            patch_w = min(src_w, patch_w)
        if patch_h <= 0 or patch_w <= 0:
            cv2.imwrite(output_path, dst_img)
            return dst_img
        
        x_src = random.randint(0, src_w - patch_w)
        y_src = random.randint(0, src_h - patch_h)
        patch = src_img[y_src:y_src+patch_h, x_src:x_src+patch_w]
        
        x_dst = random.randint(0, w - patch_w)
        y_dst = random.randint(0, h - patch_h)
        result = dst_img.copy()
        result[y_dst:y_dst+patch_h, x_dst:x_dst+patch_w] = patch
        
        # Try seamless clone if possible
        try:
            mask = np.zeros_like(dst_img, dtype=np.uint8)
            mask[y_dst:y_dst+patch_h, x_dst:x_dst+patch_w] = 1
            result = cv2.seamlessClone(dst_img, patch, mask, (x_dst+patch_w//2, y_dst+patch_h//2), cv2.NORMAL_CLONE)
        except:
            pass
        cv2.imwrite(output_path, result)
        return result
    
    def text_insertion_forgery(self, img, output_path):
        """Insert fake text into document."""
        result = img.copy()
        h, w = img.shape[:2]
        rect_h = random.randint(20, 60)
        rect_w = random.randint(80, min(300, w-50))
        x = random.randint(20, w - rect_w - 20)
        y = random.randint(20, h - rect_h - 20)
        cv2.rectangle(result, (x, y), (x+rect_w, y+rect_h), (255, 255, 255), -1)
        cv2.rectangle(result, (x, y), (x+rect_w, y+rect_h), (0, 0, 0), 1)
        cv2.imwrite(output_path, result)
        return result
    
    def splicing_forgery(self, img1, img2, output_path):
        """Splice two images together, handling different sizes."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize images to same height (max height)
        target_h = max(h1, h2)
        if h1 != target_h:
            img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
        if h2 != target_h:
            img2 = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
        
        w1 = img1.shape[1]
        w2 = img2.shape[1]
        split = random.randint(w1//3, 2*w1//3) if w1 > 0 else w1//2
        
        left = img1[:, :split]
        right = img2[:, split:]
        
        # Ensure same width
        if left.shape[1] != right.shape[1]:
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = right[:, :min_w]
        
        result = np.hstack([left, right])
        cv2.imwrite(output_path, result)
        return result
    
    def generate_dataset(self, num_forged_per_genuine=3):
        """Generate forged versions for each genuine image."""
        genuine_images = [f for f in os.listdir(self.genuine_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(genuine_images) == 0:
            raise ValueError(f"No genuine images found in {self.genuine_folder}")
        
        # Copy genuine images to output
        for img_name in genuine_images:
            src = os.path.join(self.genuine_folder, img_name)
            dst = os.path.join(self.output_folder, "genuine", img_name)
            cv2.imwrite(dst, cv2.imread(src))
        
        # Generate forgeries
        all_images = []
        for f in genuine_images:
            img = cv2.imread(os.path.join(self.genuine_folder, f))
            if img is not None:
                all_images.append(img)
        
        for i, img_name in enumerate(tqdm(genuine_images, desc="Generating forgeries")):
            original = all_images[i]
            base_name = os.path.splitext(img_name)[0]
            
            for j in range(num_forged_per_genuine):
                forgery_type = random.choice(['copy_paste', 'text_insertion', 'splicing'])
                out_path = os.path.join(self.output_folder, "forged", f"{base_name}_forged_{j}_{forgery_type}.png")
                
                try:
                    if forgery_type == 'copy_paste':
                        other_img = random.choice([img for idx, img in enumerate(all_images) if idx != i])
                        self.copy_paste_forgery(other_img, original, out_path)
                    elif forgery_type == 'text_insertion':
                        self.text_insertion_forgery(original, out_path)
                    else:  # splicing
                        other_img = random.choice([img for idx, img in enumerate(all_images) if idx != i])
                        self.splicing_forgery(original, other_img, out_path)
                except Exception as e:
                    print(f"Warning: Failed to create {forgery_type} forgery for {img_name}: {e}")
                    # Fallback to text insertion
                    self.text_insertion_forgery(original, out_path)
        
        print(f"✅ Dataset created: {len(genuine_images)} genuine, {len(genuine_images)*num_forged_per_genuine} forged")
        return self.output_folder

# ------------------------- 2. FEATURE EXTRACTOR -------------------------
class FeatureExtractor:
    def __init__(self, detector):
        self.detector = detector
    
    def extract_features_for_image(self, image_path):
        try:
            ela_score, _, _ = self.detector.ela_analysis(image_path)
            visual_score, _ = self.detector.visual_analysis(image_path)
            layout_score, _ = self.detector.layout_analysis(image_path)
            ocr_score, _ = self.detector.ocr_consistency_analysis(image_path)
            full_text = self.detector.get_full_text(image_path)
            text_score, _ = self.detector.text_expert.analyze(full_text)
            font_score, _ = self.detector.font_expert.analyze(image_path)
            noise_score, _ = self.detector.noiseprint_expert.analyze(image_path)
            return {
                'ela': ela_score, 'visual': visual_score, 'layout': layout_score,
                'ocr': ocr_score, 'text_perp': text_score, 'font_gmm': font_score,
                'noiseprint': noise_score
            }
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_dataset_features(self, dataset_folder):
        features, labels = [], []
        genuine_folder = os.path.join(dataset_folder, "genuine")
        for img_name in tqdm(os.listdir(genuine_folder), desc="Genuine features"):
            img_path = os.path.join(genuine_folder, img_name)
            feats = self.extract_features_for_image(img_path)
            if feats:
                features.append(feats); labels.append(0)
        forged_folder = os.path.join(dataset_folder, "forged")
        for img_name in tqdm(os.listdir(forged_folder), desc="Forged features"):
            img_path = os.path.join(forged_folder, img_name)
            feats = self.extract_features_for_image(img_path)
            if feats:
                features.append(feats); labels.append(1)
        return features, labels

# ------------------------- 3. LIGHTGBM TRAINER -------------------------
class LightGBMTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = ['ela', 'visual', 'layout', 'ocr', 'text_perp', 'font_gmm', 'noiseprint']
    
    def train(self, features, labels, test_size=0.2):
        X = np.array([[f[name] for name in self.feature_names] for f in features])
        y = np.array(labels)
        print(f"\n📊 Dataset: {len(X)} samples, {len(self.feature_names)} features")
        print(f"   Genuine: {sum(y==0)}, Forged: {sum(y==1)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        self.model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                         class_weight='balanced', random_state=42, verbose=-1)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n📊 Model Performance:")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")
        
        importance = self.model.feature_importances_
        print("\n📈 Feature Importance:")
        for name, imp in sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp}")
        return self.model
    
    def save_model(self, model_path="lightgbm_model.txt"):
        if self.model:
            self.model.booster_.save_model(model_path)
            with open(model_path.replace('.txt', '_features.pkl'), 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"\n✅ Model saved to {model_path}")
        return model_path

# ------------------------- 4. MAIN PIPELINE -------------------------
def run_complete_pipeline(genuine_folder, output_dataset_folder="forged_dataset", model_output="lightgbm_model.txt"):
    print("="*60)
    print("COMPLETE FORGERY DETECTION PIPELINE")
    print("="*60)
    
    print("\n📁 Step 1: Generating forged dataset...")
    generator = ForgeryGenerator(genuine_folder, output_dataset_folder)
    dataset_path = generator.generate_dataset(num_forged_per_genuine=3)
    
    print("\n🤖 Step 2: Initializing EnhancedForgeryDetector...")
    detector = EnhancedForgeryDetector(regional_lang='en')
    
    print("\n🔍 Step 3: Extracting features with 7 experts...")
    extractor = FeatureExtractor(detector)
    features, labels = extractor.extract_dataset_features(dataset_path)
    
    if len(features) == 0:
        print("❌ No features extracted. Please check your images and detector.")
        return
    
    print("\n🎯 Step 4: Training LightGBM model...")
    trainer = LightGBMTrainer()
    trainer.train(features, labels)
    trainer.save_model(model_output)
    
    print("\n✅ Pipeline complete!")
    print(f"   - Dataset: {dataset_path}")
    print(f"   - Model: {model_output}")
    print("\n📌 Next steps:")
    print("   1. Update your EnhancedForgeryDetector to load this model")
    print("   2. Replace weighted fusion with model.predict()")

# ------------------------- COMMAND LINE -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--genuine_folder", type=str, default="data/genuine")
    parser.add_argument("--dataset_folder", type=str, default="forged_dataset")
    parser.add_argument("--model_output", type=str, default="lightgbm_model.txt")
    args = parser.parse_args()
    
    if args.train:
        if not os.path.exists(args.genuine_folder):
            print(f"❌ Genuine folder not found: {args.genuine_folder}")
        else:
            run_complete_pipeline(args.genuine_folder, args.dataset_folder, args.model_output)
    else:
        print("Usage: python training_pipeline.py --train --genuine_folder path/to/genuine")