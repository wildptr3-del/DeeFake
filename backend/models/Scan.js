const mongoose = require('mongoose');

const ScanSchema = new mongoose.Schema({
  mediaHash: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  verdict: {
    type: String,
    required: true,
    enum: ['Deepfake', 'Real']
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  spreadLevel: {
    type: String,
    required: true,
    enum: ['Low', 'Medium', 'High', 'Critical']
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Scan', ScanSchema);
