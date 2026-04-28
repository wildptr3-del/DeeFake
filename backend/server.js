/**
 * Deefake - Backend Server
 * Express REST API entry point
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
const PORT = process.env.PORT || 5000;

// ─── Middleware ───────────────────────────────────────────
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'https://deefake-frontend.onrender.com',
  credentials: true
}));
app.use(morgan('dev'));

// AI Service Proxy (Python port 8000) - MUST be before body-parsers
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'https://deefake-49zo.onrender.com';
const proxyOptions = {
  proxyReqPathResolver: (req) => `${req.baseUrl}${req.url === '/' ? '' : req.url}`,
  timeout: 120000,
  limit: '200mb' // Match the AI service limit
};

app.use('/api/ai', proxy(AI_SERVICE_URL, proxyOptions));
app.use('/api/detect', proxy(AI_SERVICE_URL, proxyOptions));

app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: { error: 'Too many requests, please try again later.' }
});
app.use('/api/', limiter);

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ─── Routes ──────────────────────────────────────────────
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

// ─── 404 Handler ─────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// ─── Error Handler ───────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('Server Error:', err.stack);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error'
  });
});

// ─── Start Server ────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🛡️  Deefake Backend running on http://localhost:${PORT}`);
  console.log(`   Environment: ${process.env.NODE_ENV || 'Production'}\n`);
});

module.exports = app;
