const express = require('express');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const { pdfToImages } = require('../utils/pdfToImage');
const { analyzeDocument } = require('../utils/claudeAnalyzer');
const { callPythonDetector } = require('../utils/pythonDetector');

const router = express.Router();

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

function normalizeConfidence(value) {
  if (value == null || Number.isNaN(Number(value))) return 0;
  const numeric = Number(value);
  return numeric <= 1 ? Math.round(numeric * 100) : Math.round(numeric);
}

function mapPythonVerdict(verdict) {
  if (verdict === 'FORGED') return 'FORGED';
  if (verdict === 'SUSPICIOUS') return 'SUSPICIOUS';
  return 'AUTHENTIC';
}

function deriveRiskFromConfidence(confidence) {
  if (confidence >= 0.75) return 'HIGH';
  if (confidence >= 0.45) return 'MEDIUM';
  return 'LOW';
}

function aggregateVerdict(results) {
  const verdicts = results.map((result) => result.verdict);
  if (verdicts.includes('FORGED')) return 'FORGED';
  if (verdicts.includes('SUSPICIOUS')) return 'SUSPICIOUS';
  return 'AUTHENTIC';
}

function aggregateRisk(results) {
  const risks = results.map((result) => result.risk_level);
  if (risks.includes('HIGH')) return 'HIGH';
  if (risks.includes('MEDIUM')) return 'MEDIUM';
  return 'LOW';
}

function summarizeAnalysisSource(results) {
  const sources = results.map((result) => result.analysis_source);
  if (sources.includes('fusion')) return 'fusion';
  if (sources.includes('python_only')) return 'python_only';
  if (sources.includes('claude_only')) return 'claude_only';
  return 'unknown';
}

function buildAnalysisSource(claudeResult, pythonResult) {
  if (claudeResult && pythonResult) return 'fusion';
  if (pythonResult) return 'python_only';
  if (claudeResult) return 'claude_only';
  return 'unavailable';
}

function confidenceSourceLabel(source) {
  if (source === 'python_only') return 'Python detector confidence';
  if (source === 'claude_only') return 'Claude confidence';
  if (source === 'fusion') return 'Fusion confidence';
  return 'Confidence';
}

function analysisSourceLabel(source) {
  if (source === 'python_only') return 'Python detector only';
  if (source === 'claude_only') return 'Claude only';
  if (source === 'fusion') return 'Claude + Python fusion';
  return 'Unknown source';
}

function mergeVerdicts(claudeVerdict, pythonVerdict, pythonConfidence) {
  const severity = { FORGED: 3, SUSPICIOUS: 2, AUTHENTIC: 1 };
  const pythonMapped = mapPythonVerdict(pythonVerdict);

  if ((severity[pythonMapped] ?? 1) > (severity[claudeVerdict] ?? 1)) {
    if (pythonMapped === 'FORGED' && pythonConfidence > 0.6) return 'FORGED';
    return pythonMapped;
  }

  return claudeVerdict;
}

function mergeRiskLevel(claudeRisk, pythonRisk, pythonConfidence) {
  if (pythonRisk === 'HIGH' || pythonConfidence >= 0.75) return 'HIGH';
  if (pythonRisk === 'MEDIUM' || pythonConfidence >= 0.45) {
    return claudeRisk === 'HIGH' ? 'HIGH' : 'MEDIUM';
  }
  return claudeRisk;
}

function mergeConfidence(claudeConfidence, pythonConfidence) {
  const claude = normalizeConfidence(claudeConfidence);
  const python = normalizeConfidence(pythonConfidence);

  if (!claude && !python) return 0;
  if (!claude) return python;
  if (!python) return claude;

  return Math.round((claude * 0.55) + (python * 0.45));
}

function mergeFlags(claudeFlags = [], pythonFlags = []) {
  if (!claudeFlags.length) return pythonFlags;
  if (!pythonFlags.length) return claudeFlags;
  return [...claudeFlags, ...pythonFlags].slice(0, 8);
}

function mergeRecommendations(claudeRecommendations = [], pythonRecommendations = []) {
  return [...new Set([...claudeRecommendations, ...pythonRecommendations])];
}

function buildPythonTopLevel(pythonResult) {
  const confidence = Number(pythonResult.confidence || 0);
  return {
    verdict: mapPythonVerdict(pythonResult.verdict),
    overall_confidence: normalizeConfidence(confidence),
    summary:
      pythonResult.summary ||
      `Python detector classified this document as ${mapPythonVerdict(pythonResult.verdict)} with ${normalizeConfidence(confidence)}% confidence.`,
    risk_level: pythonResult.risk_level || deriveRiskFromConfidence(confidence),
    document_type_detected: pythonResult.document_type_detected || 'unknown',
    flags: pythonResult.flags || [],
    recommendations:
      pythonResult.recommendations && pythonResult.recommendations.length
        ? pythonResult.recommendations
        : ['Manual review recommended.'],
  };
}

function safeUnlink(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch (error) {
    console.warn(`[analyze] Could not delete temp file ${filePath}: ${error.message}`);
  }
}

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

  if (!ALLOWED_TYPES[mimeType].includes(ext)) {
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
    }
  } catch (conversionError) {
    safeUnlink(file.path);
    console.error(`[analyze] Conversion error: ${conversionError.message}`);
    return res.status(422).json({
      success: false,
      error: `Could not process file: ${conversionError.message}`,
      code: 'CONVERSION_FAILED',
    });
  }

  const results = [];
  const claudeMediaType = mimeType === 'application/pdf' ? 'image/png' : toClaudeMediaType(mimeType);
  const language = req.body?.language || 'English';

  for (let index = 0; index < base64Pages.length; index += 1) {
    console.log(`[analyze] Page ${index + 1}/${totalPages} - "${originalName}"`);

    let pageBase64 = base64Pages[index];
    if (mimeType === 'application/pdf') {
      const optimized = await optimizeImage(Buffer.from(pageBase64, 'base64'));
      pageBase64 = optimized.toString('base64');
    }

    const [claudeOutcome, pythonOutcome] = await Promise.allSettled([
      analyzeDocument(pageBase64, claudeMediaType, language),
      callPythonDetector(pageBase64),
    ]);

    let claudeResult = null;
    let claudeError = null;
    if (claudeOutcome.status === 'fulfilled') {
      claudeResult = claudeOutcome.value;
    } else {
      claudeError = claudeOutcome.reason?.message || 'Claude analysis failed';
      console.error(`[analyze] Claude failed on page ${index + 1}: ${claudeError}`);
    }

    let pythonResult = null;
    let pythonError = null;
    if (pythonOutcome.status === 'fulfilled' && pythonOutcome.value.success) {
      pythonResult = pythonOutcome.value.data;
      console.log(
        `[analyze] Python detector page ${index + 1}: ${pythonResult.verdict} (${(Number(pythonResult.confidence || 0) * 100).toFixed(1)}%)`
      );
    } else if (pythonOutcome.status === 'fulfilled') {
      pythonError = pythonOutcome.value.error;
      console.warn(`[analyze] Python detector unavailable for page ${index + 1}: ${pythonError}`);
    } else {
      pythonError = pythonOutcome.reason?.message || 'Python detector failed';
      console.warn(`[analyze] Python detector failed on page ${index + 1}: ${pythonError}`);
    }

    if (!claudeResult && !pythonResult) {
      safeUnlink(file.path);
      return res.status(503).json({
        success: false,
        error: 'Both Claude and the Python detector are unavailable for this document.',
        code: 'ANALYSIS_UNAVAILABLE',
        details: {
          claude: claudeError,
          python: pythonError,
        },
      });
    }

    const pythonTopLevel = pythonResult ? buildPythonTopLevel(pythonResult) : null;
    const analysisSource = buildAnalysisSource(claudeResult, pythonResult);

    let finalResult;
    if (!claudeResult && pythonTopLevel) {
      finalResult = {
        ...pythonTopLevel,
        verdict: pythonTopLevel.verdict,
        risk_level: pythonTopLevel.risk_level,
      };
    } else if (claudeResult && pythonTopLevel) {
      finalResult = {
        verdict: mergeVerdicts(claudeResult.verdict, pythonResult.verdict, Number(pythonResult.confidence || 0)),
        risk_level: mergeRiskLevel(
          claudeResult.risk_level,
          pythonTopLevel.risk_level,
          Number(pythonResult.confidence || 0)
        ),
        overall_confidence: mergeConfidence(claudeResult.overall_confidence, Number(pythonResult.confidence || 0)),
        summary: claudeResult.summary || pythonTopLevel.summary,
        document_type_detected:
          claudeResult.document_type_detected && claudeResult.document_type_detected !== 'unknown'
            ? claudeResult.document_type_detected
            : pythonTopLevel.document_type_detected,
        flags: mergeFlags(claudeResult.flags, pythonTopLevel.flags),
        recommendations: mergeRecommendations(claudeResult.recommendations, pythonTopLevel.recommendations),
      };
    } else {
      finalResult = {
        verdict: claudeResult.verdict,
        risk_level: claudeResult.risk_level,
        overall_confidence: normalizeConfidence(claudeResult.overall_confidence),
        summary: claudeResult.summary,
        document_type_detected: claudeResult.document_type_detected,
        flags: claudeResult.flags,
        recommendations: claudeResult.recommendations,
      };
    }

    results.push({
      page: index + 1,
      analysis_source: analysisSource,
      analysis_source_label: analysisSourceLabel(analysisSource),
      confidence_source_label: confidenceSourceLabel(analysisSource),
      verdict: finalResult.verdict,
      risk_level: finalResult.risk_level,
      overall_confidence: finalResult.overall_confidence,
      summary: finalResult.summary,
      document_type_detected: finalResult.document_type_detected,
      flags: finalResult.flags,
      recommendations: finalResult.recommendations,
      claude_analysis: {
        available: Boolean(claudeResult),
        verdict: claudeResult?.verdict ?? null,
        confidence: normalizeConfidence(claudeResult?.overall_confidence),
        risk_level: claudeResult?.risk_level ?? null,
        error: claudeError,
      },
      python_analysis: pythonResult
        ? {
            verdict: pythonResult.verdict,
            confidence: Number(pythonResult.confidence || 0),
            risk_level: pythonResult.risk_level || deriveRiskFromConfidence(Number(pythonResult.confidence || 0)),
            summary: pythonResult.summary || pythonTopLevel.summary,
            flags: pythonResult.flags || [],
            recommendations: pythonResult.recommendations || [],
            document_type_detected: pythonResult.document_type_detected || 'unknown',
            ocr_backend: pythonResult.ocr_backend || null,
            scores: pythonResult.scores,
            explanations: pythonResult.explanations,
            component_status: pythonResult.component_status || {},
            model_reliability: pythonResult.model_reliability || null,
            suspicious_regions: pythonResult.suspicious_regions || [],
            region_crops: pythonResult.region_crops || [],
            offline_llm_summary: pythonResult.offline_llm_summary || null,
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
    overall_analysis_source: summarizeAnalysisSource(results),
    overall_analysis_source_label: analysisSourceLabel(summarizeAnalysisSource(results)),
    overall_verdict: aggregateVerdict(results),
    overall_risk: aggregateRisk(results),
  });
});

module.exports = router;
