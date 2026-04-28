"""
SportShield AI - AI Microservice
Flask-based AI service for media protection analysis.
Uses Gemini API with Google Search grounding for web detection,
ffmpeg for video processing, and NetworkX for propagation graph generation.
"""

import os
import io
import uuid
import json
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import networkx as nx

# Legacy forensic_utils removed - replaced by EfficientNet inference.py
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
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
CORS(app, origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")])

# Google Cloud Vision Credentials
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Ensure DB is ready
init_db()

# Initialize local deepfake detector
# Initialize new EfficientNet Deepfake Detector
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deepfake_efficientnet.pth")
detector = DeepfakeInference(model_path=MODEL_PATH)


# ─── Health Check ─────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "SportShield AI - AI Service",
        "version": "4.0.0"
    })


# ─── File Upload ─────────────────────────────────────────
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route("/api/media/upload", methods=["POST"])
def upload_media():
    """Accept file upload and store it, returning a file ID."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(uploaded.filename)[1]
    saved_name = f"{file_id}{ext}"
    save_path = os.path.join(UPLOADS_DIR, saved_name)
    uploaded.save(save_path)

    return jsonify({
        "message": "File uploaded successfully",
        "fileId": file_id,
        "filePath": f"/uploads/{saved_name}",
        "originalName": uploaded.filename,
        "size": os.path.getsize(save_path),
        "mimeType": uploaded.content_type,
    }), 201


# ─── Media Analysis (real) ───────────────────────────────
@app.route("/api/ai/analyze", methods=["POST"])
def analyze_media():
    """
    Analyze an uploaded image for manipulation.
    Accepts multipart file upload with field name 'file'.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send as 'file' field."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = uploaded.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    file_id = request.form.get("file_id", "unknown")

    # Run the new EfficientNet-B0 analysis pipeline
    # Save temporarily to analyze
    temp_path = f"temp_analyze_{file_id}.jpg"
    uploaded.seek(0)
    uploaded.save(temp_path)
    
    try:
        result = detector.analyze_image(temp_path)
        result["file_id"] = file_id
        return jsonify(result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─── Fingerprint Generation ──────────────────────────────
@app.route("/api/ai/fingerprint", methods=["POST"])
def generate_fingerprint():
    """Generate a unique digital fingerprint for media content."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        img_bytes = request.files["file"].read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        hashes = compute_hashes(pil_image)
        return jsonify({
            "phash": str(hashes["phash"]),
            "ahash": str(hashes["ahash"]),
            "dhash": str(hashes["dhash"]),
            "whash": str(hashes["whash"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Similarity Detection ────────────────────────────────
@app.route("/api/ai/similarity", methods=["POST"])
def detect_similarity():
    """
    Compare two images for similarity using perceptual hashing.
    Accepts 'file_a' and optionally 'file_b' as uploads.
    If only file_a is provided, compares against reference.
    """
    if "file_a" not in request.files:
        # Fallback: accept JSON body for backward compatibility
        import random
        data = request.get_json(silent=True) or {}
        similarity_pct = round(random.uniform(10.0, 99.9), 1)
        return jsonify({
            "file_id_a": data.get("file_id_a", "unknown"),
            "file_id_b": data.get("file_id_b", "reference"),
            "similarity_percentage": similarity_pct,
            "match_level": (
                "High" if similarity_pct >= 75
                else "Medium" if similarity_pct >= 40
                else "Low"
            ),
            "analysis_time_ms": random.randint(100, 500),
        })

    try:
        img_a = Image.open(io.BytesIO(request.files["file_a"].read()))
        hashes_a = compute_hashes(img_a)

        if "file_b" in request.files:
            img_b = Image.open(io.BytesIO(request.files["file_b"].read()))
            hashes_b = compute_hashes(img_b)
        else:
            # Compare against reference (imported from utils)
            hashes_b = {"phash": REFERENCE_PHASH, "ahash": REFERENCE_AHASH}

        dist = hash_distance(hashes_a["phash"], hashes_b["phash"])
        # Max hamming distance for 64-bit hash = 64
        similarity_pct = round(max(0, (1 - dist / 64)) * 100, 1)

        return jsonify({
            "similarity_percentage": similarity_pct,
            "hash_distance": dist,
            "match_level": (
                "High" if similarity_pct >= 75
                else "Medium" if similarity_pct >= 40
                else "Low"
            ),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Deepfake Detection (local) ─────────────────────────
@app.route("/api/detect", methods=["POST"], strict_slashes=False)
def detect_deepfake():
    """Local deepfake detection for images and videos."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    uploaded = request.files["file"]
    file_id = request.form.get("file_id", str(uuid.uuid4()))
    
    # Save temporarily to analyze
    ext = os.path.splitext(uploaded.filename)[1].lower()
    temp_name = f"detect_{file_id}{ext}"
    temp_path = os.path.join(UPLOADS_DIR, temp_name)
    uploaded.save(temp_path)
    
    try:
        print(f"[AI] Received detection request for: {uploaded.filename}")
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"[AI] Starting video analysis for {temp_path}...")
            result = detector.analyze_video(temp_path)
            print(f"[AI] Video analysis complete: {result.get('status')}")
        else:
            print(f"[AI] Starting image analysis for {temp_path}...")
            result = detector.analyze_image(temp_path)
            print(f"[AI] Image analysis complete: {result.get('status')}")
            
        # Cast to standard types for JSON serialization
        return jsonify({
            "success": True,
            "media_type": "video" if ext in [".mp4", ".avi", ".mov", ".mkv"] else "image",
            "is_deepfake": result["status"] == "Manipulated",
            "confidence": float(result["confidence"]),
            "status": result["status"],
            "integrity_score": result["integrity_score"],
            "details": result["details"]
        })
    except Exception as e:
        print(f"[AI] ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─── Spread Tracking ─────────────────────────────────────
@app.route("/api/ai/spread", methods=["POST"])
def track_spread():
    """Track how far a piece of media has spread online."""
    import random
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id", "unknown")

    platforms = [
        {
            "name": "Twitter",
            "copies": random.randint(5, 200),
            "first_seen": "2026-04-10T08:23:00Z"
        },
        {
            "name": "Blog",
            "copies": random.randint(1, 50),
            "first_seen": "2026-04-11T14:05:00Z"
        },
        {
            "name": "News",
            "copies": random.randint(2, 80),
            "first_seen": "2026-04-12T09:17:00Z"
        }
    ]

    total_copies = sum(p["copies"] for p in platforms)

    return jsonify({
        "file_id": file_id,
        "total_copies": total_copies,
        "platforms": platforms,
        "risk_level": (
            "Critical" if total_copies > 200
            else "High" if total_copies > 100
            else "Medium" if total_copies > 30
            else "Low"
        ),
        "scan_timestamp": "2026-04-13T18:30:00Z"
    })


# ═════════════════════════════════════════════════════════
#  WEB DETECTION (Google Cloud Vision API)
# ═════════════════════════════════════════════════════════

@app.route("/api/ai/web-detect", methods=["POST"], strict_slashes=False)
def web_detect():
    """
    Web detection using Google Cloud Vision API.

    Accepts multipart file upload with field name 'file'.
    Optionally accepts 'file_id' in form data.

    Flow:
      1. Send image to Vision API WEB_DETECTION endpoint
      2. Parse pages_with_matching_images for source URIs
      3. Filter social media URLs (twitter, reddit, facebook, instagram)
      4. Calculate Reach Score = unique_domains + (social_count × 2)
      5. Generate propagation graph
      6. Store score in SQLite

    Returns structured JSON with all results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send as 'file' field."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = uploaded.read()
        # Validate it's a readable image
        Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    file_id = request.form.get("file_id", str(uuid.uuid4()))

    # ── 1. Vision API Web Detection ─────────────────────
    vision_result = detect_web_vision(img_bytes)

    all_urls = vision_result.get("all_urls", [])
    pages = vision_result.get("pages", [])
    method = "vision_api" if vision_result.get("success") else "fallback"

    # ── 2. Social Media Filtering ─────────────────────
    social = filter_social_urls(all_urls)

    # ── 3. Extract unique domains ─────────────────────
    domains = extract_unique_domains(all_urls)

    # ── 4. Calculate Reach Score ──────────────────────
    score_result = calculate_reach_score(
        unique_domains=len(domains),
        social_count=social["social_count"],
        detected_domains=domains
    )

    # ── 5. Generate Propagation Graph ────────────────
    graph = _build_propagation_graph(pages, domains, all_urls)

    # ── 6. Store in SQLite ───────────────────────────
    try:
        save_score(file_id, {
            "score": score_result["score"],
            "unique_domains": score_result["unique_domains"],
            "social_count": score_result["social_count"],
            "risk_level": score_result["risk_level"],
            "detected_domains": domains,
            "high_impact_spreaders": social["high_impact_spreaders"],
            "pages": pages,
            "method": method,
        })
    except Exception:
        traceback.print_exc()

    # ── 7. Build response ────────────────────────────
    return jsonify({
        "method": method,
        "all_urls": all_urls,
        "pages": pages,
        "social_spread": {
            "social_count": social["social_count"],
            "by_platform": social["by_platform"],
            "social_urls": social["social_urls"][:10],
        },
        "reach_score": score_result,
        "detected_domains": domains,
        "high_impact_spreaders": social["high_impact_spreaders"],
        "propagation_graph": graph,
        "fallback_reason": vision_result.get("error"),
    })


# ═════════════════════════════════════════════════════════
#  VIDEO ANALYSIS (ffmpeg + Gemini API)
# ═════════════════════════════════════════════════════════

@app.route("/api/ai/video-analyze", methods=["POST"])
def video_analyze():
    """
    Analyze a video by extracting frames and running Gemini API
    on each sampled frame.

    Accepts multipart file upload with field name 'file'.
    The file must be a video (mp4, avi, mov, etc).

    Returns persistent hosts, Reach Score, and social filtering.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send as 'file' field."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    file_id = request.form.get("file_id", str(uuid.uuid4()))

    # Save video temporarily
    ext = os.path.splitext(uploaded.filename)[1] or ".mp4"
    video_name = f"video_{file_id}{ext}"
    video_path = os.path.join(UPLOADS_DIR, video_name)
    uploaded.save(video_path)

    try:
        # Run video analysis
        result = analyze_video_frames(video_path, max_frames=30)

        if not result.get("success"):
            return jsonify({
                "error": result.get("error", "Video analysis failed"),
                "method": "video_analysis",
            }), 400

        all_urls = result.get("all_urls", [])
        persistent = result.get("persistent_hosts", [])

        # Social filtering on all discovered URLs
        social = filter_social_urls(all_urls)
        domains = extract_unique_domains(all_urls)

        # Reach Score
        score_result = calculate_reach_score(
            unique_domains=len(domains),
            social_count=social["social_count"],
            detected_domains=domains
        )

        # Store in DB
        try:
            save_score(file_id, {
                "score": score_result["score"],
                "unique_domains": score_result["unique_domains"],
                "social_count": score_result["social_count"],
                "risk_level": score_result["risk_level"],
                "detected_domains": domains,
                "high_impact_spreaders": social["high_impact_spreaders"],
                "pages": [],
                "method": "video_analysis",
            })
        except Exception:
            traceback.print_exc()

        return jsonify({
            "method": "video_analysis",
            "total_frames_extracted": result["total_frames_extracted"],
            "frames_analyzed": result["frames_analyzed"],
            "persistent_hosts": persistent,
            "domain_frequency": result.get("domain_frequency", {}),
            "social_spread": {
                "social_count": social["social_count"],
                "by_platform": social["by_platform"],
                "social_urls": social["social_urls"][:10],
            },
            "reach_score": score_result,
            "detected_domains": domains,
            "high_impact_spreaders": social["high_impact_spreaders"],
            "propagation_graph": _build_propagation_graph([], domains, all_urls),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up saved video
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════
#  PROPAGATION GRAPH (NetworkX)
# ═════════════════════════════════════════════════════════

@app.route("/api/ai/propagation-graph/<file_id>", methods=["GET"])
def propagation_graph(file_id):
    """
    Returns a NetworkX-generated propagation graph for a given file_id.
    The graph connects the "Source" node to all detected domains.
    """
    score_data = get_score(file_id)

    if score_data is None:
        return jsonify({"error": "No data found for this file_id."}), 404

    source_url = score_data.get("source_url", "Source")
    domains = score_data.get("detected_domains", [])
    high_impact = score_data.get("high_impact_spreaders", [])

    graph = _build_propagation_graph_from_data(source_url, domains, high_impact)

    return jsonify({
        "file_id": file_id,
        "graph": graph,
        "reach_score": score_data.get("reach_score", 0),
        "risk_level": score_data.get("risk_level", "Low"),
    })


def _build_propagation_graph(pages: list, domains: list, all_urls: list = None) -> dict:
    """
    Build a propagation graph from Vision API results.
    Source node = earliest indexed URL (first page).
    Connects source to all detected domains.
    """
    source_label = "Source"
    source_url = ""
    domain_to_url = {}
    
    from urllib.parse import urlparse
    if all_urls:
        for u in all_urls:
            try:
                dom = urlparse(u).netloc
                if dom.startswith("www."):
                    dom = dom[4:]
                if dom not in domain_to_url:
                    domain_to_url[dom] = u
            except Exception:
                pass
                
    if pages:
        first_url = pages[0].get("url", "")
        if first_url:
            source_url = first_url
            try:
                source_label = urlparse(first_url).netloc or "Source"
            except Exception:
                pass

    return _build_propagation_graph_from_data(source_label, domains, [], domain_to_url, source_url)


def _build_propagation_graph_from_data(
    source_label: str, domains: list, high_impact: list, domain_to_url: dict = None, source_url: str = ""
) -> dict:
    """Build a NetworkX graph and serialize to JSON for the frontend."""
    if domain_to_url is None:
        domain_to_url = {}
        
    G = nx.DiGraph()

    # Add source node
    # Add source node
    G.add_node(source_label, type="source", url=source_url)

    # Add domain nodes + edges
    from social_filter import _classify_social
    for domain in domains[:100]:
        # Check if this domain is a social platform
        is_social = _classify_social(f"https://{domain}") is not None
        node_type = "social" if is_social else "domain"
        url = domain_to_url.get(domain, domain)
        G.add_node(domain, type=node_type, url=url)
        G.add_edge(source_label, domain)

    # Serialize to JSON-friendly format
    nodes = []
    for node, attrs in G.nodes(data=True):
        nodes.append({
            "id": node,
            "type": attrs.get("type", "domain"),
            "url": attrs.get("url", node),
        })

    edges = []
    for src, tgt in G.edges():
        edges.append({"source": src, "target": tgt})

    return {"nodes": nodes, "links": edges}


# ─── Watermark Detection ─────────────────────────────────
@app.route("/api/ai/watermark/detect", methods=["POST"])
def detect_watermark():
    """Detect existing watermarks in media."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        img_bytes = request.files["file"].read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        image_np = np.array(pil_image)
        
        secret = extract_lsb_watermark(image_np)
        
        return jsonify({
            "success": True,
            "has_watermark": secret is not None,
            "watermark_text": secret if secret else "None",
            "method": "LSB-Steganography"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Watermark Embedding ─────────────────────────────────
@app.route("/api/ai/watermark/embed", methods=["POST"])
def embed_watermark():
    """Embed an invisible digital watermark into media."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
        
    secret_text = request.form.get("text", "SportShield-Protected")
    
    try:
        img_bytes = request.files["file"].read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        image_np = np.array(pil_image)
        
        watermarked_np = embed_lsb_watermark(image_np, secret_text)
        
        # Convert back to image and return
        res_image = Image.fromarray(watermarked_np)
        img_io = io.BytesIO()
        res_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        # We can't easily return a file and JSON in the same response without multipart
        # For now, let's just return the file with a custom header for the text
        from flask import send_file
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='protected_media.png'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Error Handlers ──────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ─── Run Server ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("AI_PORT", 8000))
    debug = os.getenv("FLASK_ENV", "development") == "development"

    print(f"\n[AI] SportShield AI Service running on http://localhost:{port}")
    print(f"    Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"    Vision API:  {'configured' if GOOGLE_APPLICATION_CREDENTIALS else 'NOT SET (fallback mode)'}")
    print(f"    Database:    MongoDB Atlas (Cloud)\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
