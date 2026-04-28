import os
import io
import uuid
import json
import traceback
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import networkx as nx

# Import your custom modules
from utils import compute_hashes, hash_distance, REFERENCE_PHASH, REFERENCE_AHASH
from vision_service import detect_web_vision
from social_filter import filter_social_urls, extract_unique_domains
from reach_score import calculate_reach_score
from database import init_db, save_score, get_score
from video_processor import analyze_video_frames
from inference import DeepfakeInference
from watermark import embed_lsb_watermark, extract_lsb_watermark

load_dotenv()

app = Flask(__name__)
# Render Free Tier has limited RAM, so we keep the limit reasonable
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  
CORS(app)

# Ensure DB and Uploads are ready
init_db()
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── GLOBAL DETECTOR (Loaded once at startup) ───────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deepfake_efficientnet.pth")
detector = None

def get_detector():
    """Lazy-load the detector to ensure it doesn't crash the start-up phase."""
    global detector
    if detector is None:
        print("[AI] Loading EfficientNet model into memory...")
        detector = DeepfakeInference(model_path=MODEL_PATH)
    return detector

# ─── Health Check (Wakes up the model) ────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    # Calling get_detector here ensures the model is "Hot" when the judge visits the site
    get_detector() 
    return jsonify({
        "status": "ok",
        "model_loaded": detector is not None,
        "service": "Deefake - AI Service"
    })

# ─── Optimized Deepfake Detection ──────────────────────────
@app.route("/api/detect", methods=["POST"], strict_slashes=False)
def detect_deepfake():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    uploaded = request.files["file"]
    file_id = request.form.get("file_id", str(uuid.uuid4()))
    
    # Save extension
    ext = os.path.splitext(uploaded.filename)[1].lower()
    temp_path = os.path.join(UPLOADS_DIR, f"temp_{file_id}{ext}")
    
    try:
        # 1. Save file once
        uploaded.save(temp_path)
        
        # 2. Get the pre-loaded detector
        ai_detector = get_detector()
        
        # 3. Choose analysis path
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # For videos, we only analyze a few frames to stay under 30s
            result = ai_detector.analyze_video(temp_path)
        else:
            # For images, analyze directly
            result = ai_detector.analyze_image(temp_path)
            
        return jsonify({
            "success": True,
            "is_deepfake": result.get("status") == "Manipulated",
            "confidence": float(result.get("confidence", 0)),
            "status": result.get("status"),
            "integrity_score": result.get("integrity_score", 100),
            "details": result.get("details", {})
        })

    except Exception as e:
        print(f"[AI] ERROR: {str(e)}")
        return jsonify({"success": False, "error": "Inference Timeout or Memory Limit"}), 500
    finally:
        # Cleanup immediately to save Render disk space
        if os.path.exists(temp_path):
            os.remove(temp_path)

# (Keep the rest of your routes: web-detect, watermark, etc. as they were)
# Note: Ensure you import numpy as np at the top for the watermark route.

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000)) 
    # Important: In production, pre-load the detector once
    get_detector()
    app.run(host="0.0.0.0", port=port)