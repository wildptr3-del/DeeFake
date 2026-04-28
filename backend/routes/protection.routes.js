/**
 * Protection Routes
 * Handles AI-powered media protection operations
 */

const express = require('express');
const router = express.Router();
const Scan = require('../models/Scan');

/**
 * @route   POST /api/protection/analyze
 * @desc    Analyze media for deepfakes and spread level
 * @access  Public
 */
router.post('/analyze', async (req, res) => {
  try {
    const { mediaHash, verdict, confidence, spreadLevel } = req.body;

    if (!mediaHash) {
      return res.status(400).json({ error: 'mediaHash is required for analysis.' });
    }

    // Use findOneAndUpdate with upsert: true to ensure uniqueness by mediaHash
    // If the same media is scanned again, it updates the record instead of creating a duplicate.
    const scanResult = await Scan.findOneAndUpdate(
      { mediaHash },
      { 
        verdict, 
        confidence, 
        spreadLevel, 
        timestamp: new Date() 
      },
      { 
        new: true,      // Return the updated document
        upsert: true,   // Create if it doesn't exist
        runValidators: true 
      }
    );

    // Return a single clean object for the frontend to immediately display the final result
    res.status(200).json({
      mediaHash: scanResult.mediaHash,
      verdict: scanResult.verdict,
      confidence: scanResult.confidence,
      spreadLevel: scanResult.spreadLevel,
      timestamp: scanResult.timestamp
    });
  } catch (error) {
    console.error('Analyze Error:', error);
    res.status(500).json({ error: 'Internal server error during media analysis.' });
  }
});

/**
 * @route   GET /api/protection/reports
 * @desc    Get all protection scan reports
 * @access  Public
 */
router.get('/reports', async (req, res) => {
  try {
    const reports = await Scan.find().sort({ timestamp: -1 });
    res.json({
      count: reports.length,
      reports
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
