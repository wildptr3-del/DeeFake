/**
 * SportShield AI - API Service
 * All requests go through Vite proxy (no CORS issues)
 */

// ─── Upload media file ──────────────────────────────────
export async function uploadMedia(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/media/upload', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || 'Upload failed');
  }
  return res.json();
}

// ─── Analyze media ───────────────────────────────────────
export async function analyzeMedia(file, fileId) {
  const formData = new FormData();
  formData.append('file', file);
  if (fileId) formData.append('file_id', fileId);

  const res = await fetch('/api/ai/analyze', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Analysis failed');
  return res.json();
}

// ─── Similarity check ───────────────────────────────────
export async function checkSimilarity(fileIdA, fileIdB) {
  const res = await fetch('/api/ai/similarity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id_a: fileIdA, file_id_b: fileIdB || 'reference' }),
  });

  if (!res.ok) throw new Error('Similarity check failed');
  return res.json();
}

// ─── Spread tracking ────────────────────────────────────
export async function trackSpread(fileId) {
  const res = await fetch('/api/ai/spread', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId }),
  });

  if (!res.ok) throw new Error('Spread tracking failed');
  return res.json();
}

// ─── Web Detection (Gemini AI with Google Search) ───────
export async function webDetect(file, fileId) {
  const formData = new FormData();
  formData.append('file', file);
  if (fileId) formData.append('file_id', fileId);

  const res = await fetch('/api/ai/web-detect', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Web detection failed');
  return res.json();
}

// ─── Video Analysis (ffmpeg + Gemini API) ────────────────
export async function videoAnalyze(file, fileId) {
  const formData = new FormData();
  formData.append('file', file);
  if (fileId) formData.append('file_id', fileId);

  const res = await fetch('/api/ai/video-analyze', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Video analysis failed');
  return res.json();
}

// ─── Propagation Graph ──────────────────────────────────
export async function getPropagationGraph(fileId) {
  const res = await fetch(`/api/ai/propagation-graph/${fileId}`, {
    method: 'GET',
  });

  if (!res.ok) throw new Error('Propagation graph failed');
  return res.json();
}
// ─── Local Deepfake Detection ──────────────────────────
export async function detectDeepfake(file, fileId) {
  const formData = new FormData();
  formData.append('file', file);
  if (fileId) formData.append('file_id', fileId);

  const res = await fetch('/api/detect', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Deepfake detection failed');
  return res.json();
}

// ─── Digital Watermarking ──────────────────────────────
export async function embedWatermark(file, text) {
  const formData = new FormData();
  formData.append('file', file);
  if (text) formData.append('text', text);

  const res = await fetch('/api/ai/watermark/embed', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Watermark embedding failed');
  return res.blob(); // Returns the watermarked image blob
}

export async function detectWatermark(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/ai/watermark/detect', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Watermark detection failed');
  return res.json();
}
