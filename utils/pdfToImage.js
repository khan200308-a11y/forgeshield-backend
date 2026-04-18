const path = require('path');
const fs = require('fs');
const { fromPath } = require('pdf2pic');

/**
 * Convert a PDF file to an array of base64 PNG strings (one per page).
 * Primary method: pdf2pic (requires GraphicsMagick + Ghostscript).
 * Fallback method: pdfjs-dist (pure JS, no system deps).
 *
 * @param {string} pdfPath - Absolute path to the uploaded PDF file.
 * @returns {Promise<string[]>} Array of base64-encoded PNG strings.
 */
async function pdfToImages(pdfPath) {
  const uploadsDir = path.join(__dirname, '..', 'uploads');

  try {
    console.log(`[pdfToImage] Attempting pdf2pic conversion for: ${path.basename(pdfPath)}`);
    const images = await convertWithPdf2pic(pdfPath, uploadsDir);
    console.log(`[pdfToImage] pdf2pic succeeded — ${images.length} page(s) converted`);
    return images;
  } catch (err) {
    console.warn(`[pdfToImage] pdf2pic failed (${err.message}) — falling back to pdfjs-dist`);
    try {
      const images = await convertWithPdfjs(pdfPath);
      console.log(`[pdfToImage] pdfjs-dist succeeded — ${images.length} page(s) converted`);
      return images;
    } catch (fallbackErr) {
      console.error(`[pdfToImage] pdfjs-dist also failed: ${fallbackErr.message}`);
      throw new Error(`PDF conversion failed: ${fallbackErr.message}`);
    }
  }
}

/**
 * Primary converter using pdf2pic (needs GraphicsMagick + Ghostscript).
 */
async function convertWithPdf2pic(pdfPath, outputDir) {
  // First determine page count via pdfjs so we know how many pages to request
  const pageCount = await getPdfPageCount(pdfPath);

  const options = {
    density: 300,
    saveFilename: `page_${Date.now()}`,
    savePath: outputDir,
    format: 'png',
    width: 2480,
    height: 3508,
  };

  const convert = fromPath(pdfPath, options);
  const base64Images = [];

  for (let i = 1; i <= pageCount; i++) {
    const result = await convert(i, { responseType: 'base64' });
    if (result && result.base64) {
      base64Images.push(result.base64);
    } else {
      // Fallback: save to file, read it, then delete
      const fileResult = await convert(i, { responseType: 'image' });
      if (fileResult && fileResult.path) {
        const data = fs.readFileSync(fileResult.path);
        base64Images.push(data.toString('base64'));
        fs.unlinkSync(fileResult.path);
      }
    }
  }

  if (base64Images.length === 0) {
    throw new Error('pdf2pic produced no output images');
  }

  return base64Images;
}

/**
 * Fallback converter using pdfjs-dist (pure JavaScript, no system deps).
 * Renders each page to a canvas buffer and returns base64 PNGs.
 */
async function convertWithPdfjs(pdfPath) {
  // pdfjs-dist v4+ uses ES modules internally; we use the legacy/compat build
  const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');

  // Suppress the worker warning in Node environment
  pdfjsLib.GlobalWorkerOptions.workerSrc = false;

  const pdfData = new Uint8Array(fs.readFileSync(pdfPath));
  const loadingTask = pdfjsLib.getDocument({ data: pdfData, useWorkerFetch: false, isEvalSupported: false, useSystemFonts: true });
  const pdf = await loadingTask.promise;

  const base64Images = [];
  const scale = 2.0; // ~150 DPI equivalent for canvas rendering

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale });

    // Use canvas from the 'canvas' package if available, otherwise use a minimal shim
    let canvas;
    let context;
    try {
      const { createCanvas } = require('canvas');
      canvas = createCanvas(viewport.width, viewport.height);
      context = canvas.getContext('2d');
    } catch {
      // canvas package not installed — create a minimal object that pdfjs can use
      throw new Error('canvas package not available; install it with: npm install canvas');
    }

    const renderContext = {
      canvasContext: context,
      viewport,
    };

    await page.render(renderContext).promise;

    const pngBuffer = canvas.toBuffer('image/png');
    base64Images.push(pngBuffer.toString('base64'));
  }

  return base64Images;
}

/**
 * Get page count from a PDF using pdfjs-dist.
 */
async function getPdfPageCount(pdfPath) {
  try {
    const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');
    pdfjsLib.GlobalWorkerOptions.workerSrc = false;
    const pdfData = new Uint8Array(fs.readFileSync(pdfPath));
    const loadingTask = pdfjsLib.getDocument({ data: pdfData, useWorkerFetch: false, isEvalSupported: false, useSystemFonts: true });
    const pdf = await loadingTask.promise;
    return pdf.numPages;
  } catch {
    // If we can't determine page count, assume 1 and let pdf2pic handle it
    console.warn('[pdfToImage] Could not determine page count — defaulting to 1');
    return 1;
  }
}

module.exports = { pdfToImages };
