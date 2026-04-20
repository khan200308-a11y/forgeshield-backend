const Anthropic = require('@anthropic-ai/sdk');

const MODEL = 'claude-opus-4-7';
const PLACEHOLDER_API_KEY = 'your_anthropic_api_key_here';

function isClaudeConfigured() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  return Boolean(apiKey && apiKey !== PLACEHOLDER_API_KEY);
}

function getClient() {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!isClaudeConfigured()) {
    throw new Error(
      'ANTHROPIC_API_KEY is missing. Create backend/.env from .env.example and set a real Anthropic API key.'
    );
  }

  return new Anthropic({ apiKey });
}

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

function stripFences(text) {
  return text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
}

function parseResponse(text) {
  return JSON.parse(stripFences(text));
}

/**
 * Analyze a single document page image with Claude Vision.
 *
 * @param {string} base64Image - Base64-encoded PNG/JPEG (no data URI prefix).
 * @param {string} mediaType   - 'image/png' or 'image/jpeg'
 * @param {string} language    - Language hint for regional documents.
 */
async function analyzeDocument(base64Image, mediaType = 'image/png', language = 'English') {
  const langNote = language && language.toLowerCase() !== 'english'
    ? `\n\nIMPORTANT: This document may contain text in ${language}. Apply the same forensic analysis criteria to non-Latin scripts. Look for font inconsistencies and character spacing anomalies specific to ${language} typography.`
    : '';

  const start = Date.now();

  // ── First attempt ────────────────────────────────────────────────────────────
  let responseText;
  try {
    const client = getClient();
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: [
        {
          type: 'text',
          text: SYSTEM_PROMPT + langNote,
          cache_control: { type: 'ephemeral' }, // prompt caching for system prompt
        },
      ],
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: { type: 'base64', media_type: mediaType, data: base64Image },
            },
            {
              type: 'text',
              text: 'Analyze this document for signs of forgery or tampering. Return raw JSON only.',
            },
          ],
        },
      ],
    });
    responseText = response.content[0].text;
    console.log(`[analyzer] Claude responded in ${Date.now() - start}ms`);
  } catch (apiErr) {
    throw new Error(`Claude API call failed: ${apiErr.message}`);
  }

  // ── Parse first attempt ───────────────────────────────────────────────────────
  try {
    return parseResponse(responseText);
  } catch (parseErr) {
    console.warn(`[analyzer] JSON parse failed on first attempt: ${parseErr.message}`);
    console.warn(`[analyzer] Raw snippet: ${responseText.slice(0, 200)}`);
  }

  // ── Retry with correction prompt ─────────────────────────────────────────────
  console.log('[analyzer] Retrying with JSON correction prompt...');
  try {
    const client = getClient();
    const retry = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: [
        {
          type: 'text',
          text: SYSTEM_PROMPT + langNote,
          cache_control: { type: 'ephemeral' },
        },
      ],
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Image } },
            { type: 'text', text: 'Analyze this document for signs of forgery or tampering. Return raw JSON only.' },
          ],
        },
        { role: 'assistant', content: responseText },
        {
          role: 'user',
          content: 'Your previous response was not valid JSON. Return ONLY a raw JSON object — no markdown, no code fences, no explanation.',
        },
      ],
    });
    const retryText = retry.content[0].text;
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

module.exports = { analyzeDocument, isClaudeConfigured };
