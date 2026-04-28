import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { uploadMedia, detectDeepfake, webDetect, videoAnalyze, getPropagationGraph } from '../services/api';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  render() {
    if (this.state.hasError) return <div className="card" style={{ height: '420px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc' }}>
      <p style={{ color: '#64748b', fontWeight: '700' }}>Unable to load propagation radar. Please refresh.</p>
    </div>;
    return this.props.children;
  }
}

const RadarChart = ({ data, animate }) => {
  const size = 300;
  const center = size / 2;
  const radius = (size / 2) * 0.7;
  const angleStep = (Math.PI * 2) / data.length;

  const points = data.map((d, i) => {
    const r = radius * (animate ? d.value / 100 : 0.1);
    const x = center + r * Math.sin(i * angleStep);
    const y = center - r * Math.cos(i * angleStep);
    return `${x},${y}`;
  }).join(' ');

  const gridLevels = [0.25, 0.5, 0.75, 1];

  return (
    <div style={{ position: 'relative', width: '100%', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <svg width={size} height={size} style={{ overflow: 'visible' }}>
        {/* Grid Circles */}
        {gridLevels.map((lvl, idx) => (
          <circle key={idx} cx={center} cy={center} r={radius * lvl} fill="none" stroke="#e2e8f0" strokeDasharray="6, 4" strokeWidth="1" />
        ))}
        {/* Axis Lines */}
        {data.map((_, i) => {
          const x = center + radius * Math.sin(i * angleStep);
          const y = center - radius * Math.cos(i * angleStep);
          return <line key={i} x1={center} y1={center} x2={x} y2={y} stroke="#e2e8f0" strokeWidth="1.5" />;
        })}
        {/* Data Shape */}
        <polygon 
          points={points} 
          fill="url(#radarGradient)" 
          stroke="var(--primary)" 
          strokeWidth="4" 
          strokeLinejoin="round"
          style={{ transition: 'all 1.2s cubic-bezier(0.23, 1, 0.32, 1)', filter: 'drop-shadow(0 4px 12px var(--primary-glow))' }}
        />
        <defs>
          <linearGradient id="radarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.1" />
          </linearGradient>
        </defs>
        {/* Labels */}
        {data.map((d, i) => {
          const x = center + (radius + 25) * Math.sin(i * angleStep);
          const y = center - (radius + 20) * Math.cos(i * angleStep);
          return (
            <text 
              key={i} 
              x={x} 
              y={y} 
              textAnchor="middle" 
              style={{ fontSize: '0.65rem', fontWeight: '800', fill: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}
            >
              {d.label}
            </text>
          );
        })}
      </svg>
    </div>
  );
};

const NetworkGraph = ({ data, reachScore }) => {
  const fgRef = useRef();
  const containerRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 650, height: 420 });

  useEffect(() => {
    if (containerRef.current) {
      const { offsetWidth, offsetHeight } = containerRef.current;
      setDimensions({ width: offsetWidth, height: offsetHeight });
    }
  }, []);

  useEffect(() => {
    if (fgRef.current) {
      // Correctly center based on dynamic dimensions
      fgRef.current.d3Force('charge').strength(-350);
      fgRef.current.d3Force('center').x(dimensions.width / 2).y(dimensions.height / 2);
      fgRef.current.d3Force('link').distance(100);
      
      // Auto-fit after the simulation has a moment to start
      const timeout = setTimeout(() => {
        fgRef.current.zoomToFit(600, 100);
      }, 800);
      return () => clearTimeout(timeout);
    }
  }, [data, dimensions]);

  return (
    <div 
      ref={containerRef}
      className="force-graph-container" 
      style={{ 
        background: '#ffffff', 
        borderRadius: '2.5rem', 
        border: '1px solid #e2e8f0', 
        overflow: 'hidden', 
        height: '420px', 
        width: '100%', 
        boxShadow: 'inset 0 2px 10px 0 rgba(0, 0, 0, 0.03)',
        position: 'relative'
      }}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        height={dimensions.height}
        width={dimensions.width}
        nodeLabel={node => `${node.id} - ${node.type}`}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.01}
        linkDirectionalParticleWidth={2}
        linkColor={() => 'rgba(148, 163, 184, 0.3)'}
        linkWidth={1.5}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const isSource = node.type === 'source';
          const isCategory = !isSource && (node.id === 'Social Platforms' || node.id === 'Global Web');
          
          // Modern Premium Color Palette
          const colors = {
            source: { start: '#ff7043', end: '#f4511e', text: '#ffffff' },
            category: { start: '#ffb74d', end: '#fb8c00', text: '#ffffff' },
            domain: { border: '#ffd54f', text: '#424242' }
          };
          
          let theme = colors.domain;
          if (isSource) theme = colors.source;
          else if (isCategory) theme = colors.category;
          
          const label = node.id || 'Unknown';
          const subLabel = isSource ? `(Reach: ${reachScore || '0'})` : '';
          
          const fontSize = isSource ? 15/globalScale : isCategory ? 13/globalScale : 11/globalScale;
          const subFontSize = 10/globalScale;
          
          ctx.font = `${isSource || isCategory ? '800' : '600'} ${fontSize}px Inter, sans-serif`;
          const textWidth = ctx.measureText(label).width;
          const subTextWidth = subLabel ? ctx.measureText(subLabel).width : 0;
          
          const maxWidth = Math.max(textWidth, subTextWidth);
          const padding = fontSize * 1.2;
          const height = subLabel ? fontSize * 3 : fontSize * 1.8;
          const w = maxWidth + padding;
          const h = height;
          const r = h / 2;

          // Custom roundRect for maximum compatibility
          const drawRoundRect = (x, y, w, h, r) => {
            if (w < 2 * r) r = w / 2;
            if (h < 2 * r) r = h / 2;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.arcTo(x + w, y, x + w, y + h, r);
            ctx.arcTo(x + w, y + h, x, y + h, r);
            ctx.arcTo(x, y + h, x, y, r);
            ctx.arcTo(x, y, x + w, y, r);
            ctx.closePath();
          };

          // Glassmorphism / Shadow Effect
          ctx.shadowColor = 'rgba(0,0,0,0.12)';
          ctx.shadowBlur = 12 / globalScale;
          ctx.shadowOffsetY = 6 / globalScale;

          // Draw Node Shape
          if (isSource || isCategory) {
            try {
              const grd = ctx.createLinearGradient(node.x - w/2, node.y, node.x + w/2, node.y);
              grd.addColorStop(0, theme.start);
              grd.addColorStop(1, theme.end);
              ctx.fillStyle = grd;
            } catch (e) {
              ctx.fillStyle = theme.start; // Fallback
            }
            drawRoundRect(node.x - w/2, node.y - h/2, w, h, r);
            ctx.fill();
          } else {
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = theme.border;
            ctx.lineWidth = 2 / globalScale;
            drawRoundRect(node.x - w/2, node.y - h/2, w, h, r);
            ctx.fill();
            ctx.stroke();
          }

          // Reset for text
          ctx.shadowBlur = 0;
          ctx.shadowOffsetY = 0;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = theme.text;
          
          if (subLabel) {
            ctx.fillText(label, node.x, node.y - fontSize * 0.5);
            ctx.font = `700 ${subFontSize}px Inter, sans-serif`;
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.fillText(subLabel, node.x, node.y + fontSize * 0.9);
          } else {
            ctx.fillText(label, node.x, node.y);
          }

          node.__bckgDimensions = [w, h];
        }}
        onNodeClick={(node) => {
          const url = node.url || (node.id.includes('.') ? node.id : null);
          if (url) {
            const target = url.startsWith('http') ? url : `https://${url}`;
            window.open(target, '_blank', 'noopener,noreferrer');
          }
        }}
        onNodeHover={(node) => {
          const container = containerRef.current;
          if (container) {
            const canvas = container.querySelector('canvas');
            if (canvas) canvas.style.cursor = node ? 'pointer' : 'default';
          }
        }}
        // Modern Enhancements
        cooldownTicks={100}
        linkDirectionalParticles={4}
        linkDirectionalParticleSpeed={d => Math.random() * 0.01 + 0.005}
        linkDirectionalParticleWidth={3}
        linkDirectionalParticleColor={() => '#3b82f6'}
        linkColor={() => 'rgba(148, 163, 184, 0.15)'}
        linkWidth={1.5}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />
      
      {/* Background Mesh Decor */}
      <div style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100%', 
        pointerEvents: 'none',
        backgroundImage: `radial-gradient(#e2e8f0 1px, transparent 1px)`,
        backgroundSize: '30px 30px',
        opacity: 0.4,
        zIndex: 0
      }} />

      <div style={{ 
        position: 'absolute', 
        bottom: '20px', 
        right: '25px', 
        fontSize: '0.7rem', 
        color: '#64748b', 
        fontWeight: '800',
        letterSpacing: '1px',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#3b82f6' }} className="pulse-dot"></div>
        LIVE PROPAGATION RADAR
      </div>
    </div>
  );
};


const Logo = () => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '2rem', marginBottom: '3.5rem' }}>
    <img src="/logo.PNG" alt="Deefake Logo" style={{ height: '105px', width: 'auto' }} />
    <img src="/text logo.PNG" alt="Deefake Text Logo" style={{ height: '80px', width: 'auto', marginTop: '15px' }} />
  </div>
);

const HeatMap = ({ data }) => {
  if (!data || data.length === 0) return null;

  const sortedData = [...data].sort((a, b) => {
    const rankA = typeof a.rank === 'number' ? a.rank : 1000000;
    const rankB = typeof b.rank === 'number' ? b.rank : 1000000;
    return rankA - rankB;
  });

  const getHeatLevel = (rank) => {
    if (typeof rank !== 'number') return { level: 'Niche', color: '#f1f5f9', text: '#64748b', gradient: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)' };
    if (rank <= 100) return { level: 'CRITICAL', color: '#ef4444', text: '#fff', gradient: 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)', pulse: true };
    if (rank <= 10000) return { level: 'VIRAL', color: '#f87171', text: '#fff', gradient: 'linear-gradient(135deg, #ef4444 0%, #f87171 100%)' };
    if (rank <= 100000) return { level: 'HIGH', color: '#3b82f6', text: '#fff', gradient: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)' };
    if (rank <= 500000) return { level: 'MEDIUM', color: '#60a5fa', text: '#fff', gradient: 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)' };
    return { level: 'LOW', color: '#94a3b8', text: '#fff', gradient: 'linear-gradient(135deg, #64748b 0%, #94a3b8 100%)' };
  };

  return (
    <div style={{ marginTop: '2.5rem' }}>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
        gap: '1.25rem',
        background: 'rgba(255, 255, 255, 0.6)',
        padding: '2.5rem',
        borderRadius: '2.5rem',
        border: '1px solid rgba(226, 232, 240, 0.8)',
        backdropFilter: 'blur(16px)',
        boxShadow: 'inset 0 2px 40px rgba(0,0,0,0.02)'
      }}>
        {sortedData.map((item, idx) => {
          const heat = getHeatLevel(item.rank);
          
          return (
            <div key={idx} className={`fade-in ${heat.pulse ? 'pulse-subtle' : ''}`} style={{ 
              background: heat.gradient, 
              color: heat.text,
              padding: '1.25rem', 
              borderRadius: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'hidden',
              transition: 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
              boxShadow: heat.pulse ? '0 10px 20px rgba(239, 68, 68, 0.3)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              animationDelay: `${idx * 0.05}s`
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.05) translateY(-5px)';
              e.currentTarget.style.zIndex = '10';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1) translateY(0)';
              e.currentTarget.style.zIndex = '1';
            }}
            >
              {/* Decorative Circle */}
              <div style={{ 
                position: 'absolute', 
                top: '-15px', 
                right: '-15px', 
                width: '50px', 
                height: '50px', 
                background: 'rgba(255,255,255,0.15)', 
                borderRadius: '50%' 
              }}></div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <img 
                  src={`https://logo.clearbit.com/${item.domain}`} 
                  alt="" 
                  onError={(e) => e.target.style.display = 'none'}
                  style={{ width: '24px', height: '24px', borderRadius: '6px', background: 'white', padding: '2px' }}
                />
                <span style={{ fontSize: '0.7rem', fontWeight: '800', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
                  {item.domain}
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <span style={{ fontSize: '1.25rem', fontWeight: '900' }}>
                  #{typeof item.rank === 'number' ? item.rank.toLocaleString() : 'N/A'}
                </span>
                <span style={{ 
                  fontSize: '0.65rem', 
                  fontWeight: '700', 
                  background: 'rgba(255,255,255,0.2)', 
                  padding: '2px 8px', 
                  borderRadius: '10px', 
                  width: 'fit-content',
                  marginTop: '4px'
                }}>
                  {heat.level} REACH
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: '2rem', 
        justifyContent: 'center', 
        marginTop: '1.5rem',
        padding: '1rem',
        background: 'white',
        borderRadius: '1rem',
        border: '1px solid #f1f5f9'
      }}>
        {[
          { label: 'Critical', color: '#ef4444' },
          { label: 'Viral', color: '#f87171' },
          { label: 'High', color: '#3b82f6' },
          { label: 'Medium', color: '#60a5fa' },
          { label: 'Low', color: '#94a3b8' }
        ].map((l, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: l.color }}></div>
            <span style={{ fontSize: '0.7rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

function Dashboard() {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [animate, setAnimate] = useState(false);
  const [error, setError] = useState(null);
  const [impactData, setImpactData] = useState([]);

  const [currentFile, setCurrentFile] = useState(null);

  const [detectionResult, setDetectionResult] = useState({
    verdict: 'Ready',
    confidence: '0.0',
    predictions: [
      { label: 'Neural Scan', value: 0 },
      { label: 'Pixel Sync', value: 0 },
      { label: 'Metadata', value: 0 },
      { label: 'Frame Int.', value: 0 },
      { label: 'Auth Check', value: 0 }
    ]
  });

  const [spreadResult, setSpreadResult] = useState({
    verdict: 'Ready',
    confidence: '0.0',
    graphData: { nodes: [], links: [] }
  });

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setCurrentFile(file);

    setError(null);
    setAnalyzed(false);
    setAnimate(false);
    
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);

    setLoading(true);

    // Detect if the file is a video
    const isVideo = file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);

    try {
      // 1. Upload to Node.js backend
      const uploadRes = await uploadMedia(file);
      const fileId = uploadRes.fileId;

      // 2. Perform Real Deepfake Detection (via Proxy to Python)
      const deepfakeData = await detectDeepfake(file, fileId);
      
      // 3. Perform Spread Detection - use the right API for the media type
      let spreadData = { reach_score: { risk_level: 'Low', score: 0 }, propagation_graph: { nodes: [], links: [] }, all_urls: [] };
      try {
        if (isVideo) {
          // Videos go through the video-analyze endpoint
          const videoData = await videoAnalyze(file, fileId);
          spreadData = videoData;
        } else {
          // Images go through web-detect
          spreadData = await webDetect(file, fileId);
        }
      } catch (spreadErr) {
        console.error('Spread detection error (non-fatal):', spreadErr);
      }

      // 4. Map Detection Results
      const signals = deepfakeData.details?.signals || {};
      setDetectionResult({
        verdict: deepfakeData.is_deepfake ? 'Deepfake' : 'Real',
        confidence: Number(deepfakeData.confidence).toFixed(1),
        predictions: [
          { label: 'Neural Net', value: (signals.model || 0) * 100 },
          { label: 'ELA', value: (signals.ela || 0) * 100 },
          { label: 'Noise', value: (signals.noise || 0) * 100 },
          { label: 'Frequency', value: (signals.frequency || 0) * 100 },
          { label: 'Color', value: (signals.color || 0) * 100 },
          { label: 'Face', value: (signals.face_quality || 0) * 100 },
          { label: 'JPEG Ghost', value: (signals.jpeg_ghost || 0) * 100 }
        ]
      });

      // 5. Map Spread Results
      setSpreadResult({
        verdict: spreadData.reach_score?.risk_level || 'Low',
        confidence: (spreadData.reach_score?.score || 0).toFixed(1),
        graphData: spreadData.propagation_graph || { nodes: [], links: [] }
      });
      
      // 5.1 Perform Cloudflare Radar Impact Analysis
      if (spreadData.all_urls && spreadData.all_urls.length > 0) {
        try {
          const impactRes = await fetch('/api/spread/impact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls: spreadData.all_urls })
          });
          const impactJson = await impactRes.json();
          setImpactData(impactJson);
        } catch (err) {
          console.error('Cloudflare Impact Error:', err);
        }
      }

      // 6. Save final report to MongoDB (non-blocking)
      try {
        await fetch('/api/protection/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mediaHash: fileId,
            verdict: deepfakeData.is_deepfake ? 'Deepfake' : 'Real',
            confidence: deepfakeData.confidence,
            spreadLevel: spreadData.reach_score?.risk_level || 'Low'
          })
        });
      } catch (saveErr) {
        console.error('Report save error (non-fatal):', saveErr);
      }

      setAnalyzed(true);
      setTimeout(() => setAnimate(true), 100);
    } catch (err) {
      console.error('Analysis Pipeline Error:', err);
      setError('AI Analysis failed. Please ensure the Python service is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };



  const removeMedia = () => {
    setPreview(null);
    setAnalyzed(false);
    setAnimate(false);
    setError(null);
  };

  return (
    <div className="dashboard-container">
      
      <Logo />

      {error && (
        <div className="card fade-in" style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626', marginBottom: '2rem', padding: '1rem 2rem', borderRadius: '1rem' }}>
          <strong>⚠️ System Error:</strong> {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.8fr', gap: '2.5rem', marginBottom: '2.5rem' }}>
        
        <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', minHeight: '480px', padding: '2.5rem' }}>
          <h3 className="section-title">Input Analysis</h3>
          <div className="upload-box" style={{ flex: 1, position: 'relative', background: '#ffffff' }}>
            {loading && (
              <div className="loading-overlay" style={{ borderRadius: '2rem' }}>
                <div className="spinner-ring"></div>
                <p style={{ fontSize: '0.8rem', fontWeight: '800', color: 'var(--primary)', letterSpacing: '0.1em', marginTop: '1.5rem' }}>RUNNING REAL-TIME AI SCAN...</p>
              </div>
            )}
            
            {preview ? (
              <div className="preview-container" style={{ height: '100%', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', padding: '1rem' }}>
                <img src={preview} alt="Preview" className="image-preview" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: '1.5rem', boxShadow: '0 20px 40px rgba(0,0,0,0.1)' }} />
                <button className="btn-remove" onClick={removeMedia} style={{ position: 'absolute', top: '20px', right: '20px', width: '32px', height: '32px', borderRadius: '50%', background: 'rgba(15, 23, 42, 0.8)', color: 'white', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>×</button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '2.5rem' }}>
                <div style={{ background: '#f8fafc', padding: '2rem', borderRadius: '50%', boxShadow: '0 15px 30px rgba(0,0,0,0.04)', marginBottom: '2rem' }}>
                  <span className="upload-icon" style={{ fontSize: '3.5rem', margin: 0 }}>📂</span>
                </div>
                <p className="upload-text" style={{ fontSize: '1.1rem', fontWeight: '600', color: '#64748b', marginBottom: '1.5rem' }}>Select media for AI audit</p>
                <input type="file" id="media-upload" hidden onChange={handleUpload} />
                <button className="btn-choose" onClick={() => document.getElementById('media-upload').click()}>
                  Begin Deepfake Test
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="card fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '2rem', animationDelay: '0.1s' }}>
          <div className="result-section">
            <h3 className="section-title">Detection Verdict</h3>
            <div className="result-display" style={{ background: '#ffffff', padding: '1.75rem', borderRadius: '2rem', height: '100%', border: '1px solid #f1f5f9', boxShadow: '0 10px 30px rgba(0,0,0,0.02)' }}>
              <div className={`verdict-box ${detectionResult.verdict === 'Deepfake' ? 'fake' : 'real'} fade-in`} style={{ padding: '2rem', textAlign: 'center', fontSize: '2rem', fontWeight: '900', letterSpacing: '-0.02em' }}>
                {detectionResult.verdict}
              </div>
              <div className="confidence-display" style={{ marginTop: '2.5rem', textAlign: 'center' }}>
                <p className="confidence-label" style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: '800', letterSpacing: '0.15em', color: '#94a3b8' }}>AI Confidence</p>
                <p className="confidence-value" style={{ fontSize: '3.5rem', color: detectionResult.verdict === 'Deepfake' ? 'var(--danger)' : 'var(--primary)', marginTop: '0.5rem' }}>{detectionResult.confidence}%</p>
              </div>
            </div>
          </div>
          <div className="prediction-section">
            <h3 className="section-title">Deepfake Signals</h3>
            <RadarChart data={detectionResult.predictions} animate={animate} />
          </div>
        </div>
      </div>

        {/* BOTTOM ROW: Spread Part (Full Width) */}
      <div className="card fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '3rem', animationDelay: '0.2s', minHeight: '400px' }}>
        <div className="result-section">
          <h3 className="section-title">Spreadness Result</h3>
          <div className="result-display" style={{ height: '100%', justifyContent: 'center', background: '#ffffff', padding: '2.5rem', borderRadius: '2rem', border: '1px solid #f1f5f9', boxShadow: '0 10px 30px rgba(0,0,0,0.02)' }}>
            <div className={`verdict-box spread-medium fade-in`} style={{ padding: '2.5rem', textAlign: 'center', fontSize: '2rem', fontWeight: '900', letterSpacing: '-0.02em' }}>
              {spreadResult.verdict}
            </div>
            <div className="confidence-display" style={{ marginTop: '2.5rem', textAlign: 'center' }}>
              <p className="confidence-label" style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: '800', letterSpacing: '0.15em', color: '#94a3b8' }}>Spread Impact Score</p>
              <p className="confidence-value spread" style={{ fontSize: '3.5rem', marginTop: '0.5rem' }}>{spreadResult.confidence}</p>
            </div>
          </div>
        </div>
        <div className="prediction-section">
          <h3 className="section-title">Propagation Topology</h3>
          {analyzed ? (
            <ErrorBoundary>
              <NetworkGraph data={spreadResult.graphData} reachScore={spreadResult.confidence} />
            </ErrorBoundary>
          ) : (
            <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fcfdfe', borderRadius: '1.5rem', color: '#94a3b8', border: '1px dashed #cbd5e1' }}>
              Waiting for analysis...
            </div>
          )}
        </div>
      </div>

      {/* Cloudflare Viral Impact Heatmap */}
      {analyzed && impactData.length > 0 && (
        <div className="card fade-in" style={{ marginTop: '2.5rem', padding: '2rem', animationDelay: '0.3s' }}>
          <h3 className="section-title">Viral Spread Heatmap</h3>
          <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '-1rem' }}>
            Real-time intensity map of where the media is most active globally.
          </p>
          <HeatMap data={impactData} />
        </div>
      )}


    </div>
  );
}

export default Dashboard;
