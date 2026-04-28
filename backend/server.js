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
const PORT = process.env.PORT || 10000; // Render expects this to be dynamic

// ─── Middleware ───────────────────────────────────────────
app.use(helmet({
  crossOriginResourcePolicy: false, // Allows images to be served to the frontend
}));

app.use(cors({
  origin: true, // Emergency fix: Allow all origins for the hackathon demo
  credentials: true
}));

app.use(morgan('dev'));

// AI Service Proxy (Python) - MUST be before body-parsers
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'https://deefake-49zo.onrender.com';

const proxyOptions = {
  proxyReqPathResolver: (req) => {
    // Correctly resolves the path for the Python service
    const resolvedPath = req.baseUrl + (req.url === '/' ? '' : req.url);
    return resolvedPath;
  },
  timeout: 120000, // 2 minutes to handle cold starts
  proxyReqOptDecorator: function(proxyReqOpts, srcReq) {
    // Ensure large files can pass through the proxy
    return proxyReqOpts;
  }
};

// Routing for proxy
app.use('/api/ai', proxy(AI_SERVICE_URL, proxyOptions));
app.use('/api/detect', proxy(AI_SERVICE_URL, proxyOptions));

// Body parsers for internal routes
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// Rate limiting - INCREASED FOR DEMO
const limiter = rateLimit({
  windowMs: 1 * 60 * 1000, // Reduced to 1 minute window
  max: 2000, // Increased to 2000 to stop 429 errors
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
  console.log(`\n🛡️  Deefake Backend running on PORT: ${PORT}`);
  console.log(`   Environment: ${process.env.NODE_ENV || 'Production'}\n`);
});

module.exports = app;