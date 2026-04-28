import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def get_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def detect_and_crop_face(image_np, face_cascade):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Optimization: Resize for faster face detection
    scale = 1.0
    max_dim = 640.0
    h, w = gray.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        gray_small = gray

    faces = face_cascade.detectMultiScale(gray_small, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Take the largest face found
    (x, y, w_f, h_f) = max(faces, key=lambda f: f[2] * f[3])
    
    # Scale back coordinates to original image size
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w_f = int(w_f / scale)
        h_f = int(h_f / scale)
    
    # Add 20% padding
    pad_w, pad_h = int(w_f * 0.2), int(h_f * 0.2)
    img_h, img_w = image_np.shape[:2]
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w_f + pad_w)
    y2 = min(img_h, y + h_f + pad_h)
    
    face_crop = image_np[y1:y2, x1:x2]
    return face_crop

def preprocess_face(face_np, img_size=224):
    """
    1. Resize to 224x224.
    2. Normalize using ImageNet mean and std.
    3. Convert to PyTorch tensor.
    """
    face_rgb = cv2.cvtColor(face_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_img).unsqueeze(0)

def compute_hashes(pil_img):
    """Simple average hash for similarity comparison."""
    img = pil_img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    avg = pixels.mean()
    diff = pixels > avg
    return {"phash": diff, "ahash": diff}

def hash_distance(h1, h2):
    """Hamming distance between two binary hashes."""
    return np.count_nonzero(h1 != h2)

# Default references for the similarity engine
REFERENCE_PHASH = np.zeros((8, 8), dtype=bool)
REFERENCE_AHASH = np.zeros((8, 8), dtype=bool)
