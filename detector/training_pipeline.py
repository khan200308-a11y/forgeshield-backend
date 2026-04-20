import argparse
import json
import os
import pickle
import random
import shutil
import sys
import warnings

import cv2
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from unified_detector import DETECTOR_DIR, FEATURE_NAMES, EnhancedForgeryDetector

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


class ForgeryGenerator:
    def __init__(self, genuine_folder, output_folder, seed=42):
        self.genuine_folder = genuine_folder
        self.output_folder = output_folder
        self.random = random.Random(seed)
        ensure_dir(output_folder)
        ensure_dir(os.path.join(output_folder, "genuine"))
        ensure_dir(os.path.join(output_folder, "forged"))

    def _safe_write(self, path, image):
        if image is not None:
            cv2.imwrite(path, image)

    def copy_paste_forgery(self, src_img, dst_img):
        result = dst_img.copy()
        height, width = dst_img.shape[:2]
        src_height, src_width = src_img.shape[:2]
        patch_height = self.random.randint(30, max(31, min(150, max(height // 3, 31))))
        patch_width = self.random.randint(30, max(31, min(180, max(width // 3, 31))))
        patch_height = min(patch_height, src_height, height - 1)
        patch_width = min(patch_width, src_width, width - 1)

        if patch_height <= 0 or patch_width <= 0:
            return result

        x_src = self.random.randint(0, max(src_width - patch_width, 0))
        y_src = self.random.randint(0, max(src_height - patch_height, 0))
        patch = src_img[y_src : y_src + patch_height, x_src : x_src + patch_width]
        x_dst = self.random.randint(0, max(width - patch_width, 0))
        y_dst = self.random.randint(0, max(height - patch_height, 0))
        result[y_dst : y_dst + patch_height, x_dst : x_dst + patch_width] = patch
        return result

    def text_insertion_forgery(self, image):
        result = image.copy()
        height, width = image.shape[:2]
        rect_height = self.random.randint(18, max(19, min(56, max(height // 8, 19))))
        rect_width = self.random.randint(80, max(81, min(260, max(width // 2, 81))))
        x = self.random.randint(10, max(width - rect_width - 10, 10))
        y = self.random.randint(10, max(height - rect_height - 10, 10))
        cv2.rectangle(result, (x, y), (x + rect_width, y + rect_height), (255, 255, 255), -1)
        cv2.rectangle(result, (x, y), (x + rect_width, y + rect_height), (0, 0, 0), 1)
        replacement = self.random.choice(["UPDATED", "REVISED", "VALID", "PASS"])
        cv2.putText(result, replacement, (x + 6, y + rect_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return result

    def splicing_forgery(self, image_a, image_b):
        height = min(image_a.shape[0], image_b.shape[0])
        if height <= 1:
            return image_a.copy()

        image_a = cv2.resize(image_a, (int(image_a.shape[1] * height / image_a.shape[0]), height))
        image_b = cv2.resize(image_b, (int(image_b.shape[1] * height / image_b.shape[0]), height))
        split = self.random.randint(max(1, image_a.shape[1] // 4), max(2, (image_a.shape[1] * 3) // 4))
        left = image_a[:, :split]
        right = image_b[:, split:]
        return np.hstack([left, right]) if right.size else image_a.copy()

    def jpeg_recompression_genuine(self, image):
        quality = self.random.randint(35, 80)
        _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encoded, 1)

    def blur_and_scan_genuine(self, image):
        result = image.copy()
        result = cv2.GaussianBlur(result, (5, 5), 0)
        alpha = self.random.uniform(0.9, 1.08)
        beta = self.random.randint(-8, 8)
        return cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    def perspective_genuine(self, image):
        height, width = image.shape[:2]
        src = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        jitter = min(width, height) * 0.03
        dst = src + np.float32(
            [
                [self.random.uniform(-jitter, jitter), self.random.uniform(-jitter, jitter)],
                [self.random.uniform(-jitter, jitter), self.random.uniform(-jitter, jitter)],
                [self.random.uniform(-jitter, jitter), self.random.uniform(-jitter, jitter)],
                [self.random.uniform(-jitter, jitter), self.random.uniform(-jitter, jitter)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    def generate_dataset(self, num_forged_per_genuine=2, num_genuine_aug_per_genuine=1, max_genuine=None, overwrite=False):
        if overwrite and os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            ensure_dir(os.path.join(self.output_folder, "genuine"))
            ensure_dir(os.path.join(self.output_folder, "forged"))

        genuine_names = sorted([name for name in os.listdir(self.genuine_folder) if is_image_file(name)])
        if max_genuine:
            genuine_names = genuine_names[:max_genuine]
        if not genuine_names:
            raise ValueError(f"No genuine images found in {self.genuine_folder}")

        images = []
        for name in genuine_names:
            image = cv2.imread(os.path.join(self.genuine_folder, name))
            if image is not None:
                images.append((name, image))
                self._safe_write(os.path.join(self.output_folder, "genuine", name), image)

        genuine_augments = [self.jpeg_recompression_genuine, self.blur_and_scan_genuine, self.perspective_genuine]

        for index, (name, image) in enumerate(tqdm(images, desc="Generating synthetic dataset")):
            peers = [peer for peer_index, (_, peer) in enumerate(images) if peer_index != index]
            base = os.path.splitext(name)[0]

            for aug_index in range(num_genuine_aug_per_genuine):
                augmenter = self.random.choice(genuine_augments)
                output_name = f"{base}_genuine_aug_{aug_index}_{augmenter.__name__}.png"
                self._safe_write(os.path.join(self.output_folder, "genuine", output_name), augmenter(image))

            for variant_index in range(num_forged_per_genuine):
                forgery_type = self.random.choice(["copy_paste", "text_insertion", "splicing"])
                output_name = f"{base}_forged_{variant_index}_{forgery_type}.png"
                output_path = os.path.join(self.output_folder, "forged", output_name)

                if forgery_type == "copy_paste" and peers:
                    forged = self.copy_paste_forgery(self.random.choice(peers), image)
                elif forgery_type == "splicing" and peers:
                    forged = self.splicing_forgery(image, self.random.choice(peers))
                else:
                    forged = self.text_insertion_forgery(image)

                self._safe_write(output_path, forged)

        genuine_count = len([name for name in os.listdir(os.path.join(self.output_folder, "genuine")) if is_image_file(name)])
        forged_count = len([name for name in os.listdir(os.path.join(self.output_folder, "forged")) if is_image_file(name)])
        print(f"[training] Synthetic dataset ready with {genuine_count} genuine and {forged_count} forged samples.")
        return self.output_folder


def import_external_dataset(dataset_root, output_folder):
    if not dataset_root or not os.path.isdir(dataset_root):
        return {"genuine": 0, "forged": 0}

    stats = {"genuine": 0, "forged": 0}
    for label in ("genuine", "forged"):
        source = os.path.join(dataset_root, label)
        target = os.path.join(output_folder, label)
        if not os.path.isdir(source):
            continue
        for filename in os.listdir(source):
            if not is_image_file(filename):
                continue
            src_path = os.path.join(source, filename)
            target_name = f"external_{os.path.basename(dataset_root)}_{filename}"
            shutil.copy2(src_path, os.path.join(target, target_name))
            stats[label] += 1
    return stats


class FeatureExtractor:
    def __init__(self, detector):
        self.detector = detector

    def _safe_feature(self, fn, default=0.5):
        try:
            return fn()
        except Exception as exc:
            return default, {"status": "fallback", "error": str(exc)}

    def extract_features_for_image(self, image_path):
        ocr_data = self.detector._ocr_data(image_path)
        text = self.detector.get_full_text(image_path, ocr_data)
        document_type = self.detector._document_type(text)

        try:
            ela_score, _, _ = self.detector.ela_analysis(image_path)
        except Exception as exc:
            print(f"[training] ELA fallback for {image_path}: {exc}")
            ela_score = 0.5

        visual_score, _ = self._safe_feature(lambda: self.detector.visual_analysis(image_path))
        layout_score, _ = self._safe_feature(lambda: self.detector.layout_analysis(image_path, ocr_data))
        ocr_score, _ = self._safe_feature(lambda: self.detector.ocr_consistency_analysis(image_path, ocr_data))
        font_score, _ = self._safe_feature(lambda: self.detector.font_expert.analyze(image_path))
        noise_score, _ = self._safe_feature(lambda: self.detector.noiseprint_expert.analyze(image_path))
        copy_move_score, _ = self._safe_feature(lambda: self.detector.copy_move_analysis(image_path))
        semantic_score, _ = self._safe_feature(lambda: self.detector.semantic_analysis(image_path, ocr_data, document_type))
        offline_llm_score, _ = self._safe_feature(lambda: self.detector.offline_llm_expert.analyze(text, document_type), default=0.2)
        texture_score, _ = self._safe_feature(lambda: self.detector.texture_analysis(image_path))
        text_score, _ = self._safe_feature(lambda: self.detector.text_expert.analyze(text))

        return {
            "ela": float(ela_score),
            "visual": float(visual_score),
            "layout": float(layout_score),
            "ocr": float(ocr_score),
            "font_gmm": float(font_score),
            "noiseprint": float(noise_score),
            "copy_move": float(copy_move_score),
            "semantic": float(semantic_score),
            "offline_llm": float(offline_llm_score),
            "texture": float(texture_score),
            "text_perp": float(text_score),
        }

    def extract_dataset_features(self, dataset_folder):
        samples = []
        for label_name, label in (("genuine", 0), ("forged", 1)):
            folder = os.path.join(dataset_folder, label_name)
            for filename in tqdm(sorted(os.listdir(folder)), desc=f"Extracting {label_name}"):
                if not is_image_file(filename):
                    continue
                image_path = os.path.join(folder, filename)
                features = self.extract_features_for_image(image_path)
                samples.append({"path": image_path, "label": label, "features": features})
        return samples


class LightGBMTrainer:
    def __init__(self, num_threads=0, seed=42):
        self.model = None
        self.seed = seed
        self.num_threads = num_threads or os.cpu_count() or 4
        self.best_threshold = 0.62
        self.suspicious_threshold = 0.42
        self.metrics = {}
        self.selected_feature_names = list(FEATURE_NAMES)
        self.distribution_stats = {}

    def _matrix(self, samples, feature_names):
        x_data = np.array([[sample["features"][name] for name in feature_names] for sample in samples], dtype=np.float32)
        y_data = np.array([sample["label"] for sample in samples], dtype=np.int32)
        return x_data, y_data

    def _select_features(self, samples):
        variances = {}
        for name in FEATURE_NAMES:
            values = np.array([sample["features"][name] for sample in samples], dtype=np.float32)
            variances[name] = float(np.var(values))
        selected = [name for name in FEATURE_NAMES if variances[name] > 1e-6]
        if not selected:
            selected = list(FEATURE_NAMES)
        dropped = [name for name in FEATURE_NAMES if name not in selected]
        if dropped:
            print(f"[training] Dropping near-constant features: {', '.join(dropped)}")
        self.selected_feature_names = selected
        return variances

    def _build_model(self):
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=self.num_threads,
            verbose=-1,
        )

    def _compute_threshold(self, y_true, y_scores):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.25, 0.85, 61):
            predictions = (y_scores >= threshold).astype(int)
            score = f1_score(y_true, predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        suspicious_threshold = max(0.25, best_threshold - 0.15)
        return best_threshold, suspicious_threshold

    def train(self, samples):
        variances = self._select_features(samples)
        x_data, y_data = self._matrix(samples, self.selected_feature_names)
        feature_mean = np.mean(x_data, axis=0)
        feature_std = np.std(x_data, axis=0)
        feature_std = np.where(feature_std < 1e-4, 1e-4, feature_std)
        z_distance = np.mean(np.abs((x_data - feature_mean) / feature_std), axis=1)
        self.distribution_stats = {
            "feature_mean": {name: float(feature_mean[index]) for index, name in enumerate(self.selected_feature_names)},
            "feature_std": {name: float(feature_std[index]) for index, name in enumerate(self.selected_feature_names)},
            "distance_mean": float(np.mean(z_distance)),
            "distance_std": float(np.std(z_distance)),
            "distance_p95": float(np.percentile(z_distance, 95)),
        }
        print(f"[training] Samples: {len(x_data)} | Genuine: {int(np.sum(y_data == 0))} | Forged: {int(np.sum(y_data == 1))}")
        print(f"[training] Using features: {', '.join(self.selected_feature_names)}")

        n_splits = min(5, int(np.min(np.bincount(y_data))))
        n_splits = max(3, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof_proba = np.zeros(len(y_data), dtype=np.float32)

        for fold, (train_index, test_index) in enumerate(splitter.split(x_data, y_data), start=1):
            model = self._build_model()
            model.fit(x_data[train_index], y_data[train_index])
            oof_proba[test_index] = model.predict_proba(x_data[test_index])[:, 1]
            print(f"[training] Completed fold {fold}/{n_splits}")

        self.best_threshold, self.suspicious_threshold = self._compute_threshold(y_data, oof_proba)
        oof_pred = (oof_proba >= self.best_threshold).astype(int)

        self.metrics = {
            "accuracy": accuracy_score(y_data, oof_pred),
            "precision": precision_score(y_data, oof_pred, zero_division=0),
            "recall": recall_score(y_data, oof_pred, zero_division=0),
            "f1": f1_score(y_data, oof_pred, zero_division=0),
            "auc": roc_auc_score(y_data, oof_proba),
            "threshold": self.best_threshold,
            "suspicious_threshold": self.suspicious_threshold,
            "sample_count": int(len(samples)),
            "folds": int(n_splits),
        }

        print("[training] Cross-validated metrics:")
        print(f"  Accuracy : {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall   : {self.metrics['recall']:.4f}")
        print(f"  F1       : {self.metrics['f1']:.4f}")
        print(f"  AUC      : {self.metrics['auc']:.4f}")
        print(f"  Threshold: {self.best_threshold:.3f} | Suspicious threshold: {self.suspicious_threshold:.3f}")

        self.model = self._build_model()
        self.model.fit(x_data, y_data)
        importance = self.model.feature_importances_
        print("[training] Feature importance:")
        for name, score in sorted(zip(self.selected_feature_names, importance), key=lambda item: item[1], reverse=True):
            print(f"  {name}: {int(score)} (var={variances.get(name, 0.0):.6f})")

    def save(self, model_output):
        if self.model is None:
            raise ValueError("Model has not been trained.")

        self.model.booster_.save_model(model_output)
        metadata_path = model_output.replace(".txt", "_meta.json")
        features_path = model_output.replace(".txt", "_features.pkl")

        with open(features_path, "wb") as handle:
            pickle.dump(self.selected_feature_names, handle)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "feature_names": self.selected_feature_names,
                    "threshold": self.best_threshold,
                    "suspicious_threshold": self.suspicious_threshold,
                    "metrics": self.metrics,
                    "distribution_stats": self.distribution_stats,
                },
                handle,
                indent=2,
            )

        print(f"[training] Model saved to {model_output}")
        print(f"[training] Metadata saved to {metadata_path}")


def configure_runtime(args):
    os.environ["DETECTOR_USE_GPU"] = "1" if args.use_gpu else "0"
    os.environ["FORGESHIELD_TEXT_MODEL"] = args.text_model
    os.environ["FORGESHIELD_TEXT_MAX_LENGTH"] = str(args.text_max_length)
    os.environ["FORGESHIELD_ENABLE_LAYOUTLM"] = "1" if args.enable_layoutlm else "0"
    os.environ["FORGESHIELD_ENABLE_NOISEPRINT"] = "1" if args.enable_noiseprint else "0"
    os.environ["FORGESHIELD_ENABLE_TEXT_LM"] = "1" if args.enable_text_lm else "0"
    os.environ["FORGESHIELD_ENABLE_FONT_VIT"] = "1" if args.enable_font_vit else "0"
    os.environ["FORGESHIELD_ENABLE_OFFLINE_LLM"] = "1" if args.enable_offline_llm else "0"


def load_or_extract_features(cache_file, reuse_cache, extractor, dataset_folder):
    if reuse_cache and cache_file and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    samples = extractor.extract_dataset_features(dataset_folder)
    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(samples, handle, indent=2)
        print(f"[training] Cached features to {cache_file}")
    return samples


def run_complete_pipeline(args):
    set_seed(args.seed)
    configure_runtime(args)

    dataset_folder = os.path.abspath(args.dataset_folder)
    model_output = os.path.abspath(args.model_output)
    cache_file = os.path.abspath(args.cache_file) if args.cache_file else None
    ensure_dir(os.path.dirname(model_output))

    print("=" * 60)
    print("FORGESHIELD CPU-FRIENDLY TRAINING PIPELINE")
    print("=" * 60)
    print(f"[training] Genuine folder : {os.path.abspath(args.genuine_folder)}")
    print(f"[training] Dataset folder : {dataset_folder}")
    print(f"[training] Model output   : {model_output}")
    print(f"[training] Text model     : {args.text_model}")
    print(f"[training] LayoutLMv3     : {'enabled' if args.enable_layoutlm else 'disabled'}")
    print(f"[training] Noiseprint     : {'enabled' if args.enable_noiseprint else 'disabled'}")
    print(f"[training] Text LM        : {'enabled' if args.enable_text_lm else 'disabled'}")
    print(f"[training] Font ViT       : {'enabled' if args.enable_font_vit else 'disabled'}")
    print(f"[training] Offline LLM    : {'enabled' if args.enable_offline_llm else 'disabled'}")
    print(f"[training] GPU            : {'enabled' if args.use_gpu else 'disabled'}")

    generator = ForgeryGenerator(args.genuine_folder, dataset_folder, seed=args.seed)
    dataset_path = generator.generate_dataset(
        num_forged_per_genuine=args.num_forged_per_genuine,
        num_genuine_aug_per_genuine=args.num_genuine_aug_per_genuine,
        max_genuine=args.max_genuine,
        overwrite=args.overwrite_dataset,
    )

    imported = import_external_dataset(args.external_dataset_folder, dataset_path)
    if imported["genuine"] or imported["forged"]:
        print(
            f"[training] Imported external labeled data: {imported['genuine']} genuine, {imported['forged']} forged."
        )

    detector = EnhancedForgeryDetector(regional_lang=args.regional_lang, use_gpu=args.use_gpu)
    extractor = FeatureExtractor(detector)
    samples = load_or_extract_features(cache_file, args.reuse_cache, extractor, dataset_path)

    if not samples:
        raise ValueError("No training samples were extracted.")

    trainer = LightGBMTrainer(num_threads=args.num_threads, seed=args.seed)
    trainer.train(samples)
    trainer.save(model_output)
    print("[training] Pipeline complete.")


def build_parser():
    parser = argparse.ArgumentParser(description="Train ForgeShield fusion for CPU-friendly inference.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--genuine_folder", type=str, default=os.path.join("..", "data", "original"))
    parser.add_argument("--external_dataset_folder", type=str, default=os.path.join("..", "data", "forged_dataset"))
    parser.add_argument("--dataset_folder", type=str, default=os.path.join(DETECTOR_DIR, "generated_dataset"))
    parser.add_argument("--model_output", type=str, default=os.path.join(DETECTOR_DIR, "lightgbm_model.txt"))
    parser.add_argument("--cache_file", type=str, default=os.path.join(DETECTOR_DIR, "feature_cache.json"))
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--overwrite_dataset", action="store_true")
    parser.add_argument("--num_forged_per_genuine", type=int, default=3)
    parser.add_argument("--num_genuine_aug_per_genuine", type=int, default=1)
    parser.add_argument("--max_genuine", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--regional_lang", type=str, default="en")
    parser.add_argument("--text_model", type=str, default="distilgpt2")
    parser.add_argument("--text_max_length", type=int, default=192)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--enable_layoutlm", action="store_true")
    parser.add_argument("--enable_noiseprint", action="store_true")
    parser.add_argument("--enable_text_lm", action="store_true")
    parser.add_argument("--enable_font_vit", action="store_true")
    parser.add_argument("--enable_offline_llm", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.train:
        print("Usage: python training_pipeline.py --train --genuine_folder path/to/genuine")
        sys.exit(0)

    if not os.path.exists(args.genuine_folder):
        raise FileNotFoundError(f"Genuine folder not found: {args.genuine_folder}")

    args.max_genuine = args.max_genuine or None
    run_complete_pipeline(args)
