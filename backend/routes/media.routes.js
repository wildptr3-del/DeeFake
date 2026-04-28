/**
 * Media Routes
 * Handles sports media upload, retrieval, and management
 */

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const Media = require('../models/Media');

const router = express.Router();

// ─── Ensure uploads directory exists ─────────────────────
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// ─── Multer configuration ────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOADS_DIR);
  },
  filename: (req, file, cb) => {
    const fileId = uuidv4();
    const ext = path.extname(file.originalname);
    cb(null, `${fileId}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100 MB max
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      'image/jpeg', 'image/png', 'image/gif', 'image/webp',
      'video/mp4', 'video/mpeg', 'video/quicktime',
      'audio/mpeg', 'audio/wav', 'audio/ogg',
    ];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`File type '${file.mimetype}' is not supported.`));
    }
  },
});

// ─── POST /api/media/upload ──────────────────────────────
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded. Use field name "file".' });
    }

    const fileId = path.parse(req.file.filename).name; // uuid from filename
    
    // Create Media record in MongoDB
    const mediaRecord = await Media.create({
      fileId: fileId,
      originalName: req.file.originalname,
      filename: req.file.filename,
      filePath: `/uploads/${req.file.filename}`,
      mimeType: req.file.mimetype,
      size: req.file.size
    });

    res.status(201).json({
      message: 'File uploaded successfully',
      fileId: mediaRecord.fileId,
      filePath: mediaRecord.filePath,
      originalName: mediaRecord.originalName,
      size: mediaRecord.size,
      mimeType: mediaRecord.mimeType,
    });
  } catch (error) {
    console.error('Upload Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ─── GET /api/media ──────────────────────────────────────
router.get('/', async (req, res) => {
  try {
    const files = await Media.find().sort({ uploadedAt: -1 });
    res.json({
      count: files.length,
      files
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─── GET /api/media/:id ──────────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const file = await Media.findOne({ fileId: req.params.id });
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }
    res.json(file);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─── DELETE /api/media/:id ───────────────────────────────
router.delete('/:id', async (req, res) => {
  try {
    const file = await Media.findOne({ fileId: req.params.id });
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Delete physical file
    const absolutePath = path.join(UPLOADS_DIR, file.filename);
    if (fs.existsSync(absolutePath)) {
      fs.unlinkSync(absolutePath);
    }

    // Delete from MongoDB
    await Media.deleteOne({ fileId: req.params.id });
    
    res.json({ message: 'File deleted successfully', fileId: req.params.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─── Multer error handler ────────────────────────────────
router.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'File too large. Maximum size is 100 MB.' });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err) {
    return res.status(400).json({ error: err.message });
  }
  next();
});

module.exports = router;
