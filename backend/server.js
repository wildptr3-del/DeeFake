/**
 * Deefake - Backend Server
 * Optimized for Render Production Deployment
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const path = require('path');
const proxy = require('express-http-proxy');

// Import config and routes
const connectDB = require('./config/database');
const mediaRoutes = require('./routes/media.routes');
const authRoutes = require('./routes/auth.routes');
const protectionRoutes = require('./routes/protection.routes');
const { processSpreadImpact } = require('./services/radarService');

// Connect to Database
connectDB();

const app = express();
const PORT = process.env.PORT || 10000;

// ─── Middleware ───────────────────────────────────────────
app.use(helmet({
  crossOriginResourcePolicy: false, // Essential for serving images to the frontend
}));

app.use(cors({
  origin: true, // Allows the frontend to communicate without CORS blocks
  credentials: true
}));

app.use(morgan('dev'));

// ─── AI Service Proxy (Python) ─────────────────────────────
// Hardcoded to ensure the backend always finds the AI microservice
const AI_SERVICE_URL = 'https://deefake-49zo.onrender.com';

const proxyOptions = {
  proxyReqPathResolver: (req) => {
    return req.baseUrl + (req.url === '/' ? '' : req.url);
  },
  timeout: 120000, 
  parseBody: false, // CRITICAL: Streams the file to Python without loading it into Node.js RAM
  limit: '200mb'
};

// These routes bypass the body-parser and go straight to Python
app.use('/api/ai', proxy(AI_SERVICE_URL, proxyOptions));
app.use('/api/detect', proxy(AI_SERVICE_URL, proxyOptions));

// ─── Body Parsers (For non-proxy routes) ──────────────────
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// ─── Rate Limiting (Demo-Safe) ───────────────────────────
const limiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute window
  max: 2000, // Very high limit to prevent 429 errors during the demo
  message: { error: 'Too many requests, please try again later.' }
});
app.use('/api/', limiter);

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ─── Internal Routes ──────────────────────────────────────
app.use('/api/auth', authRoutes);
app.use('/api/media', mediaRoutes);
app.use('/api/protection', protectionRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Deefake Backend',
    timestamp: new Date().toISOString()
  });
});

// Spread Impact Analysis (Cloudflare Radar)
app.post('/api/spread/impact', async (req, res) => {
  const { urls } = req.body;
  if (!urls || !Array.isArray(urls)) {
    return res.status(400).json({ error: 'Please provide an array of URLs' });
  }

  try {
    const impactData = await processSpreadImpact(urls);
    res.json(impactData);
  } catch (error) {
    res.status(500).json({ error: 'Failed to analyze spread impact' });
  }
});

// ─── Error Handling ───────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.use((err, req, res, next) => {
  console.error('Server Error:', err.stack);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error'
  });
});

// ─── Start Server ────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🛡️  Deefake Backend running on PORT: ${PORT}`);
  console.log(`   Environment: ${process.env.NODE_ENV || 'Production'}\n`);
});

module.exports = app;