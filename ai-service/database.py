"""
Deefake - MongoDB Atlas Database
Replaces SQLite for production-ready cloud deployment.
"""

import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = None
db = None

def init_db():
    """Initialize MongoDB connection."""
    global client, db
    try:
        if not MONGO_URI:
            print("[WARN] MONGO_URI not found in AI Service .env. Using local fallback mode?")
            return
            
        client = MongoClient(MONGO_URI)
        db = client.get_default_database()
        
        # Create index for fast lookups by file_id
        db.reach_scores.create_index("file_id")
        print("[AI DB] Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"[AI DB] MongoDB connection failed: {e}")

def save_score(file_id: str, score_data: dict) -> str:
    """Save a Reach Score record to MongoDB."""
    if db is None:
        init_db()
        
    detected_domains = score_data.get("detected_domains", [])
    high_impact = score_data.get("high_impact_spreaders", [])
    source_url = ""

    pages = score_data.get("pages", [])
    if pages:
        source_url = pages[0].get("url", "")

    record = {
        "file_id": file_id,
        "reach_score": score_data.get("score", 0),
        "unique_domains": score_data.get("unique_domains", 0),
        "social_count": score_data.get("social_count", 0),
        "risk_level": score_data.get("risk_level", "Low"),
        "source_url": source_url,
        "detected_domains": detected_domains,
        "high_impact_spreaders": high_impact,
        "method": score_data.get("method", "google_vision"),
        "created_at": datetime.utcnow()
    }

    result = db.reach_scores.insert_one(record)
    return str(result.inserted_id)

def get_score(file_id: str) -> dict | None:
    """Retrieve the most recent Reach Score from MongoDB."""
    if db is None:
        init_db()
        
    record = db.reach_scores.find_one(
        {"file_id": file_id},
        sort=[("created_at", -1)]
    )
    
    if record:
        record["_id"] = str(record["_id"])
        return record
    return None

def get_all_scores() -> list:
    """Retrieve recent Reach Scores from MongoDB."""
    if db is None:
        init_db()
        
    cursor = db.reach_scores.find().sort("created_at", -1).limit(100)
    results = []
    for record in cursor:
        record["_id"] = str(record["_id"])
        results.append(record)
    return results

# Auto-initialize
init_db()
