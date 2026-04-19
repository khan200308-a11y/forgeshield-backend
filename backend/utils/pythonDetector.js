const axios = require('axios');

const PYTHON_API_URL = process.env.PYTHON_DETECTOR_URL || 'http://localhost:8000';
const PYTHON_TIMEOUT_MS = parseInt(process.env.PYTHON_DETECTOR_TIMEOUT_MS || '120000', 10);

/**
 * Call the Python ForgeShield detector for a single base64-encoded PNG image.
 * Returns { success, data } on success or { success: false, error } on failure.
 *
 * @param {string} base64Image - base64 PNG string (no data URI prefix)
 */
async function callPythonDetector(base64Image) {
  try {
    const response = await axios.post(
      `${PYTHON_API_URL}/detect`,
      { image_base64: base64Image },
      { timeout: PYTHON_TIMEOUT_MS }
    );
    return { success: true, data: response.data };
  } catch (err) {
    const detail = err.response?.data?.detail ?? err.message;
    console.warn(`[pythonDetector] Call failed: ${detail}`);
    return { success: false, error: detail };
  }
}

/**
 * Check if the Python detector service is reachable and ready.
 */
async function checkPythonDetectorHealth() {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/health`, { timeout: 5000 });
    return response.data;
  } catch {
    return { status: 'unreachable', detector_ready: false };
  }
}

module.exports = { callPythonDetector, checkPythonDetectorHealth };
