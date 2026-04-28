import React, { useState } from 'react';
import { uploadMedia } from '../services/api';

function MediaUpload({ onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setStatus('Uploading to secure server...');

    try {
      // 1. Upload the file
      const uploadResult = await uploadMedia(file);
      
      setStatus('Initializing AI analysis...');
      
      // 2. Trigger analysis (Simulated for this step, or calling the real analyze if implemented)
      // For now, we'll call a dedicated function to perform the analysis storage
      await performAnalysis(uploadResult.fileId);

      setStatus('Analysis complete!');
      setTimeout(() => {
        setUploading(false);
        setStatus('');
        onUploadSuccess();
      }, 1500);
    } catch (error) {
      console.error(error);
      setStatus('Upload failed: ' + error.message);
      setUploading(false);
    }
  };

  // Helper to store a result in the new MongoDB backend
  const performAnalysis = async (fileId) => {
    // In a real app, the Python AI service would return this.
    // Here we simulate the result to show off the new MongoDB Scan model.
    const mockResult = {
      mediaHash: fileId, // Using fileId as hash for demo
      verdict: Math.random() > 0.3 ? 'Real' : 'Deepfake',
      confidence: Math.floor(Math.random() * 40) + 60,
      spreadLevel: Math.random() > 0.5 ? 'High' : 'Low'
    };

    const res = await fetch('/api/protection/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(mockResult)
    });
    
    if (!res.ok) throw new Error('Failed to save analysis');
  };

  return (
    <div className="upload-container">
      <label className={`upload-zone ${uploading ? 'active' : ''}`}>
        <input 
          type="file" 
          onChange={handleFileChange} 
          hidden 
          disabled={uploading}
        />
        
        {uploading ? (
          <div className="scanning-ui">
            <div className="spinner" style={{ 
              width: '40px', height: '40px', border: '3px solid var(--primary)', 
              borderTopColor: 'transparent', borderRadius: '50%', 
              animation: 'spin 1s linear infinite', margin: '0 auto 1rem' 
            }}></div>
            <p style={{ fontWeight: '500' }}>{status}</p>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🛡️</div>
            <p style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Drop media here or click to browse</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Supports JPG, PNG, MP4, WebM (Max 100MB)</p>
          </div>
        )}
      </label>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin { to { transform: rotate(360deg); } }
      `}} />
    </div>
  );
}

export default MediaUpload;
