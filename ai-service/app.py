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
# Keep limits tight for Render Free Tier RAM
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  
CORS(app)

# Ensure DB and Uploads are ready
init_db()
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── GLOBAL DETECTOR (Deferred Loading) ─────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deepfake_efficientnet.pth")
detector = None

def get_detector():
    """
    Lazy-load the detector. 
    This prevents the 'Port scan timeout' on Render.
    """
    global detector
    if detector is None:
        print("[AI] First request received. Loading EfficientNet model into memory...")
        detector = DeepfakeInference(model_path=MODEL_PATH)
        print("[AI] Model loaded successfully.")
    return detector

# ─── Health Check (The "Heater") ──────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    # Calling this starts the model loading process without blocking the port bind
    try:
        status = "warming_up"
        if detector:
            status = "ready"
        
        return jsonify({
            "status": "ok",
            "model_status": status,
            "service": "Deefake - AI Service"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── Optimized Deepfake Detection ──────────────────────────
@app.route("/api/detect", methods=["POST"], strict_slashes=False)
def detect_deepfake():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    uploaded = request.files["file"]
    file_id = request.form.get("file_id", str(uuid.uuid4()))
    
    ext = os.path.splitext(uploaded.filename)[1].lower()
    temp_path = os.path.join(UPLOADS_DIR, f"temp_{file_id}{ext}")
    
    try:
        uploaded.save(temp_path)
        
        # This will trigger the load if it's the first time
        ai_detector = get_detector()
        
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            result = ai_detector.analyze_video(temp_path)
        else:
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
        print(f"[AI] ERROR during detection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─── STARTUP CONFIGURATION ────────────────────────────────
# DO NOT load the detector here. Let the app bind the port first.
if __name__ == "__main__":
    # Render sets the PORT environment variable. We MUST use it.
    render_port = int(os.environ.get("PORT", 8000))
    print(f"[AI] Binding to port {render_port}...")
    
    # Setting debug to False for production speed
    app.run(host="0.0.0.0", port=render_port, debug=False)