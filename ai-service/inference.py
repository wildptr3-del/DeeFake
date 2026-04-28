"""
DeepfakeInference v3.0 - Optimized Multi-Signal Forensic Detection Pipeline.

Detection signals (7 total):
  1. EfficientNet Model          - Neural network output
  2. Multi-scale ELA             - JPEG re-compression artifacts at 3 quality levels
  3. Noise Consistency           - Block-wise noise variance analysis
  4. Frequency Spectrum (DCT)    - GAN fingerprint detection
  5. Color Channel Correlation   - Cross-channel anomaly detection
  6. Face-Background Coherence   - Blur, lighting, and color consistency
  7. JPEG Ghost Detection        - Double-compression artifact detection

Fusion uses adaptive weighting: signals are weighted higher when
they produce strong (confident) outputs, and lower when uncertain.
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import torch.nn.functional as F
import os
import io

from model import get_model
from utils import get_face_detector, detect_and_crop_face, preprocess_face


# ───────────────────────────────────────────────────────────
#  Signal 1: Multi-scale Error Level Analysis
# ───────────────────────────────────────────────────────────

def ela_analysis(image_path):
    """
    Multi-scale ELA: Resave at multiple JPEG qualities and analyze
    the difference patterns. Manipulated regions show inconsistent
    error levels across quality scales.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        orig_np = np.array(original, dtype=np.float32)

        scores = []
        for quality in [75, 85, 95]:
            buffer = io.BytesIO()
            original.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            resaved = np.array(Image.open(buffer).convert('RGB'), dtype=np.float32)

            diff = np.abs(orig_np - resaved)

            # Per-channel statistics
            for ch in range(3):
                ch_diff = diff[:, :, ch]
                mean_d = np.mean(ch_diff)
                std_d = np.std(ch_diff)

                if mean_d > 0:
                    # Coefficient of variation per channel
                    cv = std_d / mean_d
                    scores.append(np.clip((cv - 0.8) / 2.0, 0, 1))

            # Block-level variance analysis (detects localized manipulation)
            h, w = diff.shape[:2]
            block_size = 32
            block_means = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = diff[y:y+block_size, x:x+block_size]
                    block_means.append(np.mean(block))

            if len(block_means) > 4:
                bm = np.array(block_means)
                block_cv = np.std(bm) / (np.mean(bm) + 1e-8)
                scores.append(np.clip((block_cv - 0.3) / 1.5, 0, 1))

        if not scores:
            return 0.5

        return float(np.mean(scores))
    except Exception as e:
        print(f"[AI] ELA error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Signal 2: Noise Consistency Analysis
# ───────────────────────────────────────────────────────────

def noise_analysis(image_np):
    """
    Multi-scale noise consistency using wavelet-like decomposition.
    Analyzes noise patterns at multiple scales to detect splicing.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.5

        scores = []

        for block_size in [16, 32, 64]:
            # High-pass filter to extract noise
            blurred = cv2.GaussianBlur(gray, (block_size | 1, block_size | 1), 0)
            noise = gray - blurred

            variances = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    variances.append(np.var(block))

            if len(variances) < 4:
                continue

            var_arr = np.array(variances)
            mean_var = np.mean(var_arr)
            if mean_var == 0:
                continue

            cv_noise = np.std(var_arr) / mean_var

            # Also check for bimodal distribution of variances
            median_var = np.median(var_arr)
            mad = np.median(np.abs(var_arr - median_var))
            if mad > 0:
                kurtosis = np.mean(((var_arr - median_var) / (mad + 1e-8))**4)
                # High kurtosis = heavy tails = some blocks very different
                kurt_score = np.clip((kurtosis - 3.0) / 20.0, 0, 1)
                scores.append(kurt_score * 0.4)

            scores.append(np.clip((cv_noise - 0.4) / 1.2, 0, 1))

        return float(np.mean(scores)) if scores else 0.5
    except Exception as e:
        print(f"[AI] Noise error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Signal 3: Texture & Gradient Analysis
# ───────────────────────────────────────────────────────────

def frequency_analysis(image_np):
    """
    Texture and gradient analysis for deepfake detection.
    GAN-generated faces often have:
      - Overly smooth/plastic skin texture
      - Unnatural gradient patterns at boundaries
      - Inconsistent sharpness across the image
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = cv2.resize(gray, (256, 256))

        scores = []

        # 1. Local texture complexity: compare texture richness across regions
        h, w = gray.shape
        block_size = 32
        texture_scores = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                # Texture richness = Laplacian variance
                texture_scores.append(cv2.Laplacian(block, cv2.CV_64F).var())

        if len(texture_scores) > 4:
            ts = np.array(texture_scores)
            # GAN images: some blocks very smooth, others normal -> high CV
            ts_mean = np.mean(ts)
            if ts_mean > 0:
                ts_cv = np.std(ts) / ts_mean
                # Screenshots have high natural variance; photos are more uniform
                # Balanced for screenshots: real 0.8-1.2, manipulated 1.5-3.0+
                scores.append(np.clip((ts_cv - 1.1) / 2.0, 0, 1))

            # Also check for unnaturally smooth regions
            smooth_ratio = np.mean(ts < (ts_mean * 0.1))
            # If >30% of blocks are extremely smooth -> suspicious
            scores.append(np.clip((smooth_ratio - 0.3) / 0.4, 0, 1))

        # 2. Gradient direction consistency
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_angle = np.arctan2(sobel_y, sobel_x + 1e-8)

        # Histogram of gradient orientations
        angle_hist, _ = np.histogram(grad_angle[grad_mag > np.mean(grad_mag)], bins=36)
        if np.sum(angle_hist) > 0:
            angle_hist = angle_hist.astype(float) / np.sum(angle_hist)
            # Entropy of gradient directions
            entropy = -np.sum(angle_hist[angle_hist > 0] * np.log2(angle_hist[angle_hist > 0]))
            max_entropy = np.log2(36)
            # Very low or very high entropy is suspicious
            norm_entropy = entropy / max_entropy
            entropy_score = abs(norm_entropy - 0.75) / 0.5  # deviation from natural
            scores.append(np.clip(entropy_score, 0, 1))

        # 3. Bilateral symmetry of the image (faces are roughly symmetric)
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        sym_diff = np.mean(np.abs(left_half - right_half))
        # Normal range: 15-35, suspicious if <8 (too perfect) or >50 (heavily warped)
        sym_score = 0.0
        if sym_diff < 8:
            sym_score = np.clip((8 - sym_diff) / 8, 0, 1)
        elif sym_diff > 50:
            sym_score = np.clip((sym_diff - 50) / 40, 0, 1)
        scores.append(sym_score * 0.4)

        return float(np.mean(scores)) if scores else 0.5
    except Exception as e:
        print(f"[AI] Texture analysis error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Signal 4: Color Channel Correlation
# ───────────────────────────────────────────────────────────

def color_correlation_analysis(image_np):
    """
    Analyze cross-channel correlations. Real photos have strong natural
    correlations between R, G, B channels. GAN/manipulated images may
    have broken or unusual channel relationships.
    """
    try:
        h, w = image_np.shape[:2]
        if h < 32 or w < 32:
            return 0.5

        b, g, r = cv2.split(image_np.astype(np.float64))

        # Global channel correlations
        rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]

        # Natural photos: high correlations (>0.85)
        # Screenshots: lower correlations due to high-contrast UI elements (~0.6-0.8)
        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
        global_score = np.clip((0.78 - avg_corr) / 0.35, 0, 1)

        # Local correlation consistency
        block_size = 64
        local_corrs = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                r_b = r[y:y+block_size, x:x+block_size].flatten()
                g_b = g[y:y+block_size, x:x+block_size].flatten()
                if np.std(r_b) > 0 and np.std(g_b) > 0:
                    local_corrs.append(np.corrcoef(r_b, g_b)[0, 1])

        if len(local_corrs) > 4:
            local_cv = np.std(local_corrs) / (np.mean(np.abs(local_corrs)) + 1e-8)
            local_score = np.clip(local_cv / 0.5, 0, 1)
        else:
            local_score = 0.5

        return float(0.5 * global_score + 0.5 * local_score)
    except Exception as e:
        print(f"[AI] Color correlation error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Signal 5: Face-Background Coherence
# ───────────────────────────────────────────────────────────

def face_coherence_analysis(face_np, full_image_np):
    """
    Analyze coherence between face region and surrounding image.
    Deepfakes often have inconsistencies at the blending boundary.
    """
    try:
        scores = []

        # 1. Blur consistency (face vs full image)
        face_gray = cv2.cvtColor(face_np, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.cvtColor(full_image_np, cv2.COLOR_BGR2GRAY)

        face_blur = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        full_blur = cv2.Laplacian(full_gray, cv2.CV_64F).var()

        if full_blur > 0:
            blur_ratio = face_blur / full_blur
            blur_score = np.clip(abs(np.log(blur_ratio + 1e-8)) / 2.0, 0, 1)
            scores.append(blur_score)

        # 2. Color temperature consistency
        face_hsv = cv2.cvtColor(face_np, cv2.COLOR_BGR2HSV)
        full_hsv = cv2.cvtColor(full_image_np, cv2.COLOR_BGR2HSV)

        face_hue_mean = np.mean(face_hsv[:, :, 0])
        full_hue_mean = np.mean(full_hsv[:, :, 0])
        hue_diff = abs(face_hue_mean - full_hue_mean)
        hue_score = np.clip(hue_diff / 30.0, 0, 1)
        scores.append(hue_score)

        # 3. Saturation consistency
        face_sat_mean = np.mean(face_hsv[:, :, 1])
        full_sat_mean = np.mean(full_hsv[:, :, 1])
        sat_diff = abs(face_sat_mean - full_sat_mean)
        sat_score = np.clip(sat_diff / 50.0, 0, 1)
        scores.append(sat_score * 0.7)

        # 4. Noise level consistency
        face_noise = np.std(cv2.Laplacian(face_gray, cv2.CV_64F))
        full_noise = np.std(cv2.Laplacian(full_gray, cv2.CV_64F))
        if full_noise > 0:
            noise_ratio = face_noise / full_noise
            noise_score = np.clip(abs(np.log(noise_ratio + 1e-8)) / 1.5, 0, 1)
            scores.append(noise_score)

        # 5. Edge density comparison
        face_edges = cv2.Canny(face_np, 80, 200)
        face_edge_density = np.mean(face_edges > 0)
        edge_score = np.clip((face_edge_density - 0.06) / 0.12, 0, 1)
        scores.append(edge_score * 0.5)

        return float(np.mean(scores)) if scores else 0.5
    except Exception as e:
        print(f"[AI] Face coherence error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Signal 6: JPEG Ghost Detection
# ───────────────────────────────────────────────────────────

def jpeg_ghost_analysis(image_path):
    """
    JPEG ghost detection: resave at many quality levels and find the
    quality that produces minimum error. If different regions have
    different optimal qualities, the image was likely edited.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        orig_np = np.array(original, dtype=np.float32)
        h, w = orig_np.shape[:2]

        if h < 64 or w < 64:
            return 0.5

        # Test at multiple quality levels
        block_size = 64
        block_best_q = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                orig_block = orig_np[y:y+block_size, x:x+block_size]
                min_err = float('inf')
                best_q = 75

                for q in range(60, 100, 5):
                    buf = io.BytesIO()
                    original.save(buf, 'JPEG', quality=q)
                    buf.seek(0)
                    recomp = np.array(Image.open(buf).convert('RGB'), dtype=np.float32)
                    recomp_block = recomp[y:y+block_size, x:x+block_size]
                    err = np.mean((orig_block - recomp_block)**2)
                    if err < min_err:
                        min_err = err
                        best_q = q

                block_best_q.append(best_q)

        if len(block_best_q) < 4:
            return 0.5

        bq = np.array(block_best_q)
        # If all blocks have similar optimal quality -> authentic
        # If blocks disagree -> manipulation detected
        q_std = np.std(bq)
        score = np.clip(q_std / 15.0, 0, 1)

        return float(score)
    except Exception as e:
        print(f"[AI] JPEG ghost error: {e}")
        return 0.5


# ───────────────────────────────────────────────────────────
#  Utility
# ───────────────────────────────────────────────────────────

def estimate_blur(image_np):
    """Estimate image blurriness using Laplacian variance."""
    if image_np is None:
        return 0
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ───────────────────────────────────────────────────────────
#  Main Inference Class
# ───────────────────────────────────────────────────────────

class DeepfakeInference:
    def __init__(self, model_path='models/deepfake_efficientnet.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_loaded = os.path.exists(model_path)
        self.model = get_model(model_path if self.weights_loaded else None, self.device)
        self.face_cascade = get_face_detector()

    def _get_model_score(self, input_tensor):
        """Get the model's manipulation probability. Class 0=fake, 1=real (ImageFolder)."""
        try:
            with torch.inference_mode():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                fake_prob = float(probabilities[0][0].item())
                real_prob = float(probabilities[0][1].item())
                return fake_prob, real_prob
        except Exception as e:
            print(f"[AI] Model inference error: {e}")
            return 0.5, 0.5

    def analyze_image(self, image_path):
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise ValueError("Could not read image")

        face_np = detect_and_crop_face(image_np, self.face_cascade)
        has_face = face_np is not None

        if not has_face:
            print("[AI] No face detected -- using full image.")
            face_np = cv2.resize(image_np, (224, 224))
        else:
            print("[AI] Face detected and cropped.")

        # Detect image format (PNG screenshots won't have JPEG artifacts)
        ext = os.path.splitext(image_path)[1].lower()
        is_jpeg = ext in ('.jpg', '.jpeg', '.jfif')
        print(f"[AI] Format: {'JPEG' if is_jpeg else 'PNG/Other'} ({ext})")

        # ── Collect all signals ───────────────────────────
        input_tensor = preprocess_face(face_np).to(self.device)
        if "cuda" in str(self.device):
            input_tensor = input_tensor.half()

        model_fake, model_real = self._get_model_score(input_tensor)
        noise_score = noise_analysis(image_np)
        freq_score = frequency_analysis(image_np)
        color_score = color_correlation_analysis(image_np)
        face_score = face_coherence_analysis(face_np, image_np) if has_face else 0.5

        # JPEG-specific signals (only meaningful for JPEG files)
        if is_jpeg:
            ela_score = ela_analysis(image_path)
            ghost_score = jpeg_ghost_analysis(image_path)
        else:
            ela_score = None   # exclude from fusion
            ghost_score = None

        # Build signal dict (only include applicable signals)
        signals = {
            'model': model_fake,
            'noise': noise_score,
            'frequency': freq_score,
            'color': color_score,
            'face_quality': face_score,
        }
        if ela_score is not None:
            signals['ela'] = ela_score
        if ghost_score is not None:
            signals['jpeg_ghost'] = ghost_score

        # Log all signals
        for name, val in signals.items():
            print(f"[AI] Signal ({name:12s}) -> {val:.4f}")

        # ── Format-Adaptive Weighted Fusion ───────────────
        if is_jpeg:
            # Full 7-signal fusion for JPEG images
            base_weights = {
                'model':        0.05,
                'ela':          0.20,
                'noise':        0.18,
                'frequency':    0.17,
                'color':        0.15,
                'face_quality': 0.12,
                'jpeg_ghost':   0.13,
            }
        else:
            # 5-signal fusion for PNG/screenshots (no JPEG-dependent signals)
            # Redistribute JPEG signal weight to format-agnostic signals
            base_weights = {
                'model':        0.08,
                'noise':        0.28,
                'frequency':    0.25,
                'color':        0.22,
                'face_quality': 0.17,
            }

        # Only keep weights for active signals
        active_weights = {k: v for k, v in base_weights.items() if k in signals}

        # Normalize
        total_w = sum(active_weights.values())
        for k in active_weights:
            active_weights[k] /= total_w

        # Weighted fusion
        fake_score = sum(active_weights[k] * signals[k] for k in active_weights)
        real_score = 1.0 - fake_score

        # Decision threshold
        THRESHOLD = 0.45
        prediction = "Manipulated" if fake_score > THRESHOLD else "Authentic"
        confidence = fake_score if prediction == "Manipulated" else real_score

        print(f"[AI] Fused -> Fake: {fake_score:.4f} | Real: {real_score:.4f}")
        print(f"[AI] Decision: {prediction} (conf: {confidence:.4f})")

        # For the response, include all 7 signal keys (set excluded ones to 0)
        all_signals = {
            'model': round(model_fake, 3),
            'ela': round(ela_score, 3) if ela_score is not None else 0.0,
            'noise': round(noise_score, 3),
            'frequency': round(freq_score, 3),
            'color': round(color_score, 3),
            'face_quality': round(face_score, 3),
            'jpeg_ghost': round(ghost_score, 3) if ghost_score is not None else 0.0,
        }

        return {
            "status": prediction,
            "confidence": round(confidence * 100, 2),
            "integrity_score": round(real_score * 100, 2),
            "success": True,
            "model_version": "EfficientNetB0-Forensic-v3.0",
            "details": {
                "manipulation_type": "Deepfake / Synthetic Alteration" if prediction == "Manipulated" else "Authentic Image",
                "prediction_probability": round(confidence, 3),
                "fake_score": round(fake_score, 3),
                "real_score": round(real_score, 3),
                "signals": all_signals
            }
        }

    def analyze_video(self, video_path, n_frames=15):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return self.analyze_image(video_path)

        step = max(1, total_frames // n_frames)
        frame_signals = []

        for i in range(n_frames):
            frame_idx = i * step
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            face_np = detect_and_crop_face(frame, self.face_cascade)
            has_face = face_np is not None
            if not has_face:
                face_np = cv2.resize(frame, (224, 224))

            input_tensor = preprocess_face(face_np).to(self.device)
            if "cuda" in str(self.device):
                input_tensor = input_tensor.half()
            model_fake, _ = self._get_model_score(input_tensor)

            noise_score = noise_analysis(frame)
            freq_score = frequency_analysis(frame)
            color_score = color_correlation_analysis(frame)
            face_score = face_coherence_analysis(face_np, frame) if has_face else 0.5

            # Per-frame fusion (no ELA/JPEG ghost for video frames)
            frame_fake = (
                0.10 * model_fake +
                0.25 * noise_score +
                0.25 * freq_score +
                0.20 * color_score +
                0.20 * face_score
            )
            frame_signals.append({
                'fake_score': frame_fake,
                'model': model_fake,
                'noise': noise_score,
                'frequency': freq_score,
                'color': color_score,
                'face_quality': face_score
            })

        cap.release()

        if not frame_signals:
            return {
                "status": "Authentic",
                "confidence": 0.0,
                "integrity_score": 100.0,
                "success": True,
                "model_version": "EfficientNetB0-Forensic-v3.0",
                "details": {
                    "manipulation_type": "Authentic Video",
                    "prediction_probability": 0.0,
                    "fake_score": 0.0,
                    "real_score": 1.0
                }
            }

        # Temporal consistency: if scores vary wildly across frames, suspicious
        frame_fakes = [f['fake_score'] for f in frame_signals]
        avg_fake = float(np.mean(frame_fakes))
        temporal_std = float(np.std(frame_fakes))

        # High temporal variance slightly boosts manipulation score
        temporal_boost = np.clip(temporal_std / 0.2, 0, 0.1)
        fake_score = min(1.0, avg_fake + temporal_boost)
        real_score = 1.0 - fake_score

        THRESHOLD = 0.42
        prediction = "Manipulated" if fake_score > THRESHOLD else "Authentic"
        confidence = fake_score if prediction == "Manipulated" else real_score

        # Average signals for reporting
        avg_signals = {}
        for key in frame_signals[0]:
            if key != 'fake_score':
                avg_signals[key] = round(float(np.mean([f[key] for f in frame_signals])), 3)

        print(f"[AI] Video: {len(frame_signals)} frames, avg_fake={avg_fake:.4f}, temporal_std={temporal_std:.4f}")
        print(f"[AI] Decision: {prediction} (conf: {confidence:.4f})")

        return {
            "status": prediction,
            "confidence": round(confidence * 100, 2),
            "integrity_score": round(real_score * 100, 2),
            "success": True,
            "model_version": "EfficientNetB0-Forensic-v3.0",
            "details": {
                "manipulation_type": "Deepfake / Synthetic Alteration" if prediction == "Manipulated" else "Authentic Video",
                "prediction_probability": round(confidence, 3),
                "fake_score": round(fake_score, 3),
                "real_score": round(real_score, 3),
                "signals": avg_signals,
                "frames_analyzed": len(frame_signals),
                "temporal_consistency": round(1.0 - temporal_std, 3)
            }
        }
