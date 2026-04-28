# 🛡️ Deefake – Synthetic Media Detection & Propagation Analytics

A production-grade AI platform for detecting deepfakes, synthetic media, and tracking their propagation across the web. The system combines a deep learning backbone (**EfficientNet-B0**) with a **7-signal forensic analysis pipeline** to provide high-accuracy detection even for images without a trained neural network.

## 🚀 Key Features

### 1. Advanced Forensic Detection (v3.0)
A multi-signal weighted fusion engine that analyzes media for manipulation signatures:
- **Neural Net (EfficientNet-B0)**: Deep learning classification.
- **Multi-scale ELA**: Detects JPEG re-compression artifacts.
- **Noise Consistency**: Identifies regional inconsistencies in high-frequency noise.
- **Texture & Gradient Analysis**: Detects "plastic skin" and unnatural gradient patterns.
- **Color Channel Correlation**: Identifies cross-channel anomalies in RGB relationships.
- **Face-Background Coherence**: Checks for lighting, blur, and color mismatches at face boundaries.
- **JPEG Ghost Detection**: Finds double-compression artifacts.

### 2. Propagation & Analytics
- **Web Detection**: Uses Google Cloud Vision to find pages hosting matching images.
- **Reach Score Algorithm**: Calculates impact based on unique domains and social media mentions.
- **Propagation Graph**: Interactive 2D visualization of how media has spread across the web.
- **Temporal Video Analysis**: Extracts frames via ffmpeg and performs consistency checks across time.

## 📂 Project Structure

```text
sportshield-ai/
├── backend/          # Node.js + Express + MongoDB (Persistence & Auth)
├── ai-service/       # Python Flask AI microservice (The Detection Engine)
│   ├── app.py              # Main Flask app & API endpoints
│   ├── inference.py        # 7-signal forensic pipeline v3.0
│   ├── model.py            # Optimized EfficientNet-B0 architecture
│   ├── train.py            # Dual-phase transfer learning pipeline
│   ├── dataset_loader.py   # Adaptive subset sizing & augmentation
│   ├── vision_service.py    # Google Cloud Vision wrapper
│   └── social_filter.py     # Social media URL filtering
├── frontend/         # React + Vite + TailwindCSS (Dashboard & Radar Charts)
└── .gitignore        # Optimized for clean GitHub pushes
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16+)
- Python (3.9+)
- MongoDB Atlas (Cloud) or Local MongoDB
- ffmpeg (required for video analysis)

### 1. Backend (Node.js)
```bash
cd backend
npm install
# Configure MONGODB_URI in .env
npm start
```

### 2. AI Service (Python)
```bash
cd ai-service
pip install -r requirements.txt
# Configure .env (optional: GOOGLE_APPLICATION_CREDENTIALS)
python app.py
```

### 3. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Training the Model
To fine-tune the deep learning model on your own dataset:
1. Organize data into `ai-service/dataset/train/` and `val/`.
2. Run the training script:
```bash
python train.py --epochs 20 --batch_size 32
```
The script uses **Adaptive Subset Sizing** to ensure fast training even on CPUs.

## 📄 License
MIT
