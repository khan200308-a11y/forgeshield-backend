const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const { pdfToImages } = require('../utils/pdfToImage');
const { analyzeDocument } = require('../utils/claudeAnalyzer');
const { callPythonDetector } = require('../utils/pythonDetector');

const ALLOWED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/jpg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
};

function toClaudeMediaType(mimeType) {
  if (mimeType === 'image/jpeg' || mimeType === 'image/jpg') return 'image/jpeg';
  return 'image/png';
}

async function optimizeImage(inputBuffer) {
  return sharp(inputBuffer)
    .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
    .png({ compressionLevel: 6 })
    .toBuffer();
}

function aggregateVerdict(results) {
  const verdicts = results.map((r) => r.verdict);
  if (verdicts.includes('FORGED')) return 'FORGED';
  if (verdicts.includes('SUSPICIOUS')) return 'SUSPICIOUS';
  return 'AUTHENTIC';
}

function aggregateRisk(results) {
  const risks = results.map((r) => r.risk_level);
  if (risks.includes('HIGH')) return 'HIGH';
  if (risks.includes('MEDIUM')) return 'MEDIUM';
  return 'LOW';
}

/**
 * Merge Claude and Python detector verdicts.
 * Python returns FORGED/GENUINE; Claude returns FORGED/SUSPICIOUS/AUTHENTIC.
 * We take the more severe of the two.
 */
function mergeVerdicts(claudeVerdict, pythonVerdict, pythonConfidence) {
  const severity = { FORGED: 3, SUSPICIOUS: 2, AUTHENTIC: 1, GENUINE: 1 };
  const pythonMapped = pythonVerdict === 'FORGED' ? 'FORGED' : 'AUTHENTIC';

  if ((severity[claudeVerdict] ?? 1) >= (severity[pythonMapped] ?? 1)) {
    return claudeVerdict;
  }
  // Python says FORGED with high confidence — escalate
  if (pythonConfidence > 0.6) return 'FORGED';
  return 'SUSPICIOUS';
}

function mergeRiskLevel(claudeRisk, pythonConfidence) {
  if (pythonConfidence >= 0.75) return 'HIGH';
  if (pythonConfidence >= 0.45) {
    const riskOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
    return (riskOrder[claudeRisk] ?? 1) >= 2 ? claudeRisk : 'MEDIUM';
  }
  return claudeRisk;
}

// POST /api/analyze
router.post('/', async (req, res) => {
  const start = Date.now();
  const file = req.file;

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

  if (!ALLOWED_TYPES[mimeType]) {
    safeUnlink(file.path);
    return res.status(400).json({
      success: false,
      error: `Unsupported file type: ${mimeType}. Allowed: PDF, JPG, PNG.`,
      code: 'INVALID_FILE_TYPE',
    });
  }

  const allowedExts = ALLOWED_TYPES[mimeType];
  if (!allowedExts.includes(ext)) {
    safeUnlink(file.path);
    return res.status(400).json({
      success: false,
      error: `File extension "${ext}" does not match detected MIME type "${mimeType}".`,
      code: 'MIME_EXTENSION_MISMATCH',
    });
  }

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
    if (mimeType === 'application/pdf') {
      base64Pages = await pdfToImages(file.path);
      totalPages = base64Pages.length;
    } else {
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

  const results = [];
  const claudeMediaType = mimeType === 'application/pdf' ? 'image/png' : toClaudeMediaType(mimeType);
  const language = req.body?.language || 'English';

  for (let i = 0; i < base64Pages.length; i++) {
    console.log(`[analyze] Page ${i + 1}/${totalPages} — "${originalName}"`);

    let pageBase64 = base64Pages[i];
    if (mimeType === 'application/pdf') {
      const buf = Buffer.from(pageBase64, 'base64');
      const optimized = await optimizeImage(buf);
      pageBase64 = optimized.toString('base64');
    }

    // ── Claude Vision analysis ────────────────────────────────────────────────
    let claudeResult;
    try {
      claudeResult = await analyzeDocument(pageBase64, claudeMediaType, language);
    } catch (analysisErr) {
      console.error(`[analyze] Claude failed on page ${i + 1}: ${analysisErr.message}`);
      claudeResult = {
        verdict: 'SUSPICIOUS',
        overall_confidence: 0,
        summary: `Claude analysis failed: ${analysisErr.message}`,
        risk_level: 'MEDIUM',
        document_type_detected: 'unknown',
        flags: [],
        recommendations: ['Manual review required.'],
        _claude_error: analysisErr.message,
      };
    }

    // ── Python ML detector analysis ───────────────────────────────────────────
    let pythonResult = null;
    const pyResponse = await callPythonDetector(pageBase64);
    if (pyResponse.success) {
      pythonResult = pyResponse.data;
      console.log(
        `[analyze] Python detector page ${i + 1}: ${pythonResult.verdict} (${(pythonResult.confidence * 100).toFixed(1)}%)`
      );
    } else {
      console.warn(`[analyze] Python detector unavailable for page ${i + 1}: ${pyResponse.error}`);
    }

    // ── Merge results ──────────────────────────────────────────────────────────
    let mergedVerdict = claudeResult.verdict;
    let mergedRisk = claudeResult.risk_level;

    if (pythonResult) {
      mergedVerdict = mergeVerdicts(claudeResult.verdict, pythonResult.verdict, pythonResult.confidence);
      mergedRisk = mergeRiskLevel(claudeResult.risk_level, pythonResult.confidence);
    }

    results.push({
      page: i + 1,
      // Merged top-level verdict
      verdict: mergedVerdict,
      risk_level: mergedRisk,
      // Claude fields
      overall_confidence: claudeResult.overall_confidence,
      summary: claudeResult.summary,
      document_type_detected: claudeResult.document_type_detected,
      flags: claudeResult.flags,
      recommendations: claudeResult.recommendations,
      // Claude raw verdict (for transparency)
      claude_analysis: {
        verdict: claudeResult.verdict,
        confidence: claudeResult.overall_confidence,
        risk_level: claudeResult.risk_level,
      },
      // Python detector results (null if service was unavailable)
      python_analysis: pythonResult
        ? {
            verdict: pythonResult.verdict,
            confidence: pythonResult.confidence,
            scores: pythonResult.scores,
            explanations: pythonResult.explanations,
            visualization_base64: pythonResult.visualization_base64 || null,
          }
        : null,
    });
  }

  safeUnlink(file.path);

  return res.status(200).json({
    success: true,
    filename: originalName,
    total_pages: totalPages,
    processing_time_ms: Date.now() - start,
    results,
    overall_verdict: aggregateVerdict(results),
    overall_risk: aggregateRisk(results),
  });
});

function safeUnlink(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch (e) {
    console.warn(`[analyze] Could not delete temp file ${filePath}: ${e.message}`);
  }
}

module.exports = router;
