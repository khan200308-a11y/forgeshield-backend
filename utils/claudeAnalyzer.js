const { GoogleGenerativeAI } = require('@google/generative-ai');

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const MODEL = 'gemini-2.0-flash';

const SYSTEM_PROMPT = `You are a forensic document examiner AI with expertise in identifying forged, manipulated, or tampered documents. Your role is to analyze document images with the precision of a trained document fraud investigator.

Analyze the provided document image across these six dimensions:

1. TEXT TAMPERING: Inconsistent character spacing, misaligned text baselines, copy-paste artifacts, text color/opacity variations, or content that is contextually inconsistent with surrounding text.

2. FONT INCONSISTENCIES: Variations in font family, weight, size, rendering quality, or anti-aliasing that differ from the dominant font style of the document.

3. LAYOUT ANOMALIES: Misaligned elements, uneven margins, inconsistent line spacing, elements that break the document's visual grid, or sections that appear composited from different sources.

4. IMAGE EDITING ARTIFACTS: JPEG compression blocking, cloning artifacts, healing brush traces, unnatural edges around pasted elements, inconsistent noise patterns, or shadow/lighting mismatches.

5. SEAL AND SIGNATURE ANALYSIS: Digital insertion of stamps or seals (check for sharp edges, missing ink spread), signature inconsistencies, or missing expected security features.

6. METADATA AND CONTEXTUAL ANALYSIS: Dates that are inconsistent, serial numbers that don't follow expected patterns, official letterheads that appear modified, or formatting that deviates from standard document conventions.

You MUST respond ONLY with a valid JSON object. No explanation before or after. No markdown code fences. Raw JSON only.

Use this exact schema:
{
  "verdict": "FORGED" | "SUSPICIOUS" | "AUTHENTIC",
  "overall_confidence": <integer 0-100>,
  "summary": "<2-3 sentence assessment>",
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "document_type_detected": "<certificate|marksheet|ID card|letter|admit card|bonafide|unknown>",
  "flags": [
    {
      "id": "flag_<n>",
      "category": "<TEXT_TAMPERING|FONT_INCONSISTENCY|LAYOUT_ANOMALY|IMAGE_ARTIFACT|SEAL_SIGNATURE|METADATA>",
      "description": "<specific observation>",
      "severity": "<HIGH|MEDIUM|LOW>",
      "confidence": <integer 0-100>,
      "region": {
        "description": "<human readable location e.g. bottom-left signature block>",
        "x_percent": <0-100>,
        "y_percent": <0-100>,
        "width_percent": <0-100>,
        "height_percent": <0-100>
      }
    }
  ],
  "recommendations": ["<action item>"]
}

If the document appears genuine, return empty flags array, verdict AUTHENTIC, and overall_confidence above 80.`;

const RETRY_PROMPT = `The previous response was not valid JSON. Return ONLY a raw JSON object with no markdown, no code fences, no explanation. Use this schema exactly:
{
  "verdict": "AUTHENTIC",
  "overall_confidence": 50,
  "summary": "Unable to fully analyze.",
  "risk_level": "LOW",
  "document_type_detected": "unknown",
  "flags": [],
  "recommendations": []
}
Correct the JSON and return only the raw object.`;

/**
 * Strip markdown code fences if the model accidentally wraps the JSON.
 */
function stripMarkdownFences(text) {
  return text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();
}

/**
 * Parse the model's response text into a validated JSON object.
 */
function parseResponse(text) {
  const cleaned = stripMarkdownFences(text);
  return JSON.parse(cleaned);
}

/**
 * Analyze a single document page image with Gemini Vision.
 *
 * @param {string} base64Image - Base64-encoded PNG/JPEG image string (no data URI prefix).
 * @param {string} [mediaType='image/png'] - MIME type of the image.
 * @param {string} [language='English'] - Language hint for regional documents.
 * @returns {Promise<Object>} Parsed forensic analysis result.
 */
async function analyzeDocument(base64Image, mediaType = 'image/png', language = 'English') {
  const languageInstruction = language && language.toLowerCase() !== 'english'
    ? `\n\nIMPORTANT: This document may contain text in ${language}. Apply the same forensic analysis criteria to non-Latin scripts. Look for font inconsistencies and character spacing anomalies specific to ${language} typography.`
    : '';

  const model = genAI.getGenerativeModel({
    model: MODEL,
    systemInstruction: SYSTEM_PROMPT + languageInstruction,
    generationConfig: { maxOutputTokens: 2048 },
  });

  const imagePart = { inlineData: { mimeType: mediaType, data: base64Image } };
  const textPart = { text: 'Analyze this document for signs of forgery or tampering. Return raw JSON only.' };

  const start = Date.now();

  // ── First attempt ────────────────────────────────────────────────────────────
  let responseText;
  try {
    const result = await model.generateContent([imagePart, textPart]);
    responseText = result.response.text();
    console.log(`[analyzer] Gemini responded in ${Date.now() - start}ms`);
  } catch (apiErr) {
    throw new Error(`Gemini API call failed: ${apiErr.message}`);
  }

  // ── Parse first attempt ───────────────────────────────────────────────────────
  try {
    return parseResponse(responseText);
  } catch (parseErr) {
    console.warn(`[analyzer] JSON parse failed on first attempt: ${parseErr.message}`);
    console.warn(`[analyzer] Raw response snippet: ${responseText.slice(0, 200)}`);
  }

  // ── Retry with correction prompt ─────────────────────────────────────────────
  console.log('[analyzer] Retrying with JSON correction prompt...');
  try {
    const chat = model.startChat({
      history: [
        { role: 'user', parts: [imagePart, textPart] },
        { role: 'model', parts: [{ text: responseText }] },
      ],
    });
    const retryResult = await chat.sendMessage(RETRY_PROMPT);
    const retryText = retryResult.response.text();
    const parsed = parseResponse(retryText);
    console.log('[analyzer] Retry parse succeeded');
    return parsed;
  } catch (retryErr) {
    console.error(`[analyzer] Retry also failed: ${retryErr.message}`);
    return {
      verdict: 'SUSPICIOUS',
      overall_confidence: 0,
      summary: 'Analysis could not be completed due to a response parsing error. Manual review is recommended.',
      risk_level: 'MEDIUM',
      document_type_detected: 'unknown',
      flags: [],
      recommendations: ['Manual review required — automated analysis failed.'],
      _parse_error: true,
    };
  }
}

module.exports = { analyzeDocument };
