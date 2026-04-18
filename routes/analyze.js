const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const { pdfToImages } = require('../utils/pdfToImage');
const { analyzeDocument } = require('../utils/claudeAnalyzer');

// Allowed MIME types and their expected extensions
const ALLOWED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/jpg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
};

/**
 * Determine the Claude-compatible media type from a MIME type.
 */
function toClaudeMediaType(mimeType) {
  if (mimeType === 'image/jpeg' || mimeType === 'image/jpg') return 'image/jpeg';
  if (mimeType === 'image/png') return 'image/png';
  return 'image/png';
}

/**
 * Optimize an image buffer with sharp before sending to Claude.
 * Resizes to max 2000px on longest side and converts to PNG.
 */
async function optimizeImage(inputBuffer) {
  return sharp(inputBuffer)
    .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
    .png({ compressionLevel: 6 })
    .toBuffer();
}

/**
 * Determine the overall verdict from an array of per-page results.
 */
function aggregateVerdict(results) {
  const verdicts = results.map((r) => r.verdict);
  if (verdicts.includes('FORGED')) return 'FORGED';
  if (verdicts.includes('SUSPICIOUS')) return 'SUSPICIOUS';
  return 'AUTHENTIC';
}

/**
 * Determine the overall risk from an array of per-page results.
 */
function aggregateRisk(results) {
  const risks = results.map((r) => r.risk_level);
  if (risks.includes('HIGH')) return 'HIGH';
  if (risks.includes('MEDIUM')) return 'MEDIUM';
  return 'LOW';
}

// POST /api/analyze
router.post('/', async (req, res) => {
  const start = Date.now();
  const file = req.file;

  // ── Validate file presence ───────────────────────────────────────────────────
  if (!file) {
    return res.status(400).json({
      success: false,
      error: 'No file uploaded. Send a file in the "document" field.',
      code: 'NO_FILE',
    });
  }

  const originalName = file.originalname || 'unknown';
  const mimeType = file.mimetype;
  const ext = path.extname(originalName).toLowerCase();

  // ── Validate MIME type ───────────────────────────────────────────────────────
  if (!ALLOWED_TYPES[mimeType]) {
    safeUnlink(file.path);
    return res.status(400).json({
      success: false,
      error: `Unsupported file type: ${mimeType}. Allowed: PDF, JPG, PNG.`,
      code: 'INVALID_FILE_TYPE',
    });
  }

  // ── Validate extension matches MIME ─────────────────────────────────────────
  const allowedExts = ALLOWED_TYPES[mimeType];
  if (!allowedExts.includes(ext)) {
    safeUnlink(file.path);
    return res.status(400).json({
      success: false,
      error: `File extension "${ext}" does not match detected MIME type "${mimeType}".`,
      code: 'MIME_EXTENSION_MISMATCH',
    });
  }

  // ── Validate file size (10 MB) ───────────────────────────────────────────────
  if (file.size > 10 * 1024 * 1024) {
    safeUnlink(file.path);
    return res.status(400).json({
      success: false,
      error: 'File exceeds the 10 MB size limit.',
      code: 'FILE_TOO_LARGE',
    });
  }

  let base64Pages = [];
  let totalPages = 1;

  try {
    // ── Convert to base64 image(s) ─────────────────────────────────────────────
    if (mimeType === 'application/pdf') {
      base64Pages = await pdfToImages(file.path);
      totalPages = base64Pages.length;
    } else {
      // Image file — optimize with sharp then base64
      const rawBuffer = fs.readFileSync(file.path);
      const optimized = await optimizeImage(rawBuffer);
      base64Pages = [optimized.toString('base64')];
      totalPages = 1;
    }
  } catch (convErr) {
    safeUnlink(file.path);
    console.error(`[analyze] Conversion error: ${convErr.message}`);
    return res.status(422).json({
      success: false,
      error: `Could not process file: ${convErr.message}`,
      code: 'CONVERSION_FAILED',
    });
  }

  // ── Analyze each page with Claude ────────────────────────────────────────────
  const results = [];
  const claudeMediaType = mimeType === 'application/pdf' ? 'image/png' : toClaudeMediaType(mimeType);

  for (let i = 0; i < base64Pages.length; i++) {
    console.log(`[analyze] Analyzing page ${i + 1} of ${totalPages} for "${originalName}"`);
    try {
      let pageBase64 = base64Pages[i];

      // For PDF-converted pages already in base64, re-optimize through sharp if needed
      if (mimeType === 'application/pdf') {
        const buf = Buffer.from(pageBase64, 'base64');
        const optimized = await optimizeImage(buf);
        pageBase64 = optimized.toString('base64');
      }

      const language = req.body && req.body.language ? req.body.language : 'English';
      const pageResult = await analyzeDocument(pageBase64, claudeMediaType, language);
      results.push({ page: i + 1, ...pageResult });
    } catch (analysisErr) {
      console.error(`[analyze] Page ${i + 1} analysis failed: ${analysisErr.message}`);
      results.push({
        page: i + 1,
        verdict: 'SUSPICIOUS',
        overall_confidence: 0,
        summary: `Page analysis failed: ${analysisErr.message}`,
        risk_level: 'MEDIUM',
        document_type_detected: 'unknown',
        flags: [],
        recommendations: ['Manual review required.'],
        _error: analysisErr.message,
      });
    }
  }

  // ── Cleanup temp file ────────────────────────────────────────────────────────
  safeUnlink(file.path);

  const processingTime = Date.now() - start;

  return res.status(200).json({
    success: true,
    filename: originalName,
    total_pages: totalPages,
    processing_time_ms: processingTime,
    results,
    overall_verdict: aggregateVerdict(results),
    overall_risk: aggregateRisk(results),
  });
});

function safeUnlink(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (e) {
    console.warn(`[analyze] Could not delete temp file ${filePath}: ${e.message}`);
  }
}

module.exports = router;
