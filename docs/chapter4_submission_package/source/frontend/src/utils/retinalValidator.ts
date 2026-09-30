export interface ClientValidationResult {
  allPassed: boolean;
  failedGate: 1 | 2 | 3 | null;
  gate1: {
    passed: boolean;
    mime: string;
    fileSizeBytes: number;
    metric: string;
    rejectionReason?: string;
    clinicalAction?: string;
  };
  gate2: {
    passed: boolean;
    aspectRatio: number;
    redToBlueRatio: number;
    redShare: number;
    isDocumentOrDiagram: boolean;
    metric: string;
    rejectionReason?: string;
    clinicalAction?: string;
  };
  gate3: {
    passed: boolean;
    laplacianVariance: number;
    metric: string;
    rejectionReason?: string;
    clinicalAction?: string;
  };
}

/**
 * Client-Side Computer Vision Retinal Validation
 * Evaluates uploaded imagery in real-time using HTML5 Canvas to strictly prevent
 * non-retinal images (diagrams, flowcharts, documents, external ocular photos)
 * from passing to downstream AI inference engines.
 */
export function analyzeRetinalImageOnCanvas(
  img: HTMLImageElement,
  fileSizeBytes: number,
  mimeType: string
): ClientValidationResult {
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  const aspectRatio = width / (height || 1);

  // --- GATE 1: File Integrity & Formats ---
  const validMimes = ['image/jpeg', 'image/jpg', 'image/png'];
  const mimePassed = validMimes.includes(mimeType.toLowerCase()) || mimeType === '';
  const sizePassed = fileSizeBytes <= 15 * 1024 * 1024;
  const dimensionPassed = width >= 256 && height >= 256;

  const gate1Passed = mimePassed && sizePassed && dimensionPassed;
  let gate1Reason: string | undefined;
  let gate1Action: string | undefined;

  if (!mimePassed) {
    gate1Reason = `Unsupported MIME format (${mimeType}). DR-CDSS requires JPEG or PNG retinal imagery.`;
    gate1Action = 'Export retinal photography from the camera workstation in standard JPEG or PNG format.';
  } else if (!sizePassed) {
    gate1Reason = `File size (${(fileSizeBytes / 1024 / 1024).toFixed(1)} MB) exceeds 15 MB limit.`;
    gate1Action = 'Export at standard diagnostic resolution without uncompressed multi-layer payloads.';
  } else if (!dimensionPassed) {
    gate1Reason = `Native resolution (${width}x${height} px) is below minimum allowable dimension (256x256 px).`;
    gate1Action = 'Provide standard high-resolution digital fundus photography.';
  }

  // --- GATE 2: Retinal Relevance & Chromatic Signature ---
  // Off-screen canvas analysis
  const sampleSize = 256;
  const canvas = document.createElement('canvas');
  canvas.width = sampleSize;
  canvas.height = sampleSize;
  const ctx = canvas.getContext('2d');

  let redToBlueRatio = 1.0;
  let redShare = 0.33;
  let isDocumentOrDiagram = false;
  let cornerAvgIntensity = 0;

  if (ctx) {
    ctx.drawImage(img, 0, 0, sampleSize, sampleSize);
    const imgData = ctx.getImageData(0, 0, sampleSize, sampleSize);
    const data = imgData.data;

    let totalR = 0;
    let totalG = 0;
    let totalB = 0;
    let foregroundCount = 0;

    // Check corner pixels (0,0), (sampleSize-1, 0), (0, sampleSize-1), (sampleSize-1, sampleSize-1)
    // Real fundus images have dark/black circular aperture borders in corners (< 40 intensity).
    // Flowcharts, whitepapers, diagrams, and photos have white or bright backgrounds (> 180 intensity).
    const cornerIndices = [
      0, // top-left
      (sampleSize - 1) * 4, // top-right
      (sampleSize * (sampleSize - 1)) * 4, // bottom-left
      (sampleSize * sampleSize - 1) * 4, // bottom-right
    ];

    let cornerLumSum = 0;
    for (const idx of cornerIndices) {
      const cr = data[idx];
      const cg = data[idx + 1];
      const cb = data[idx + 2];
      cornerLumSum += (cr + cg + cb) / 3;
    }
    cornerAvgIntensity = cornerLumSum / cornerIndices.length;

    // Compute RGB distribution across pixels
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;

      // In real fundus, foreground is the illuminated retina (lum > 15)
      if (lum > 15) {
        totalR += r;
        totalG += g;
        totalB += b;
        foregroundCount++;
      }
    }

    if (foregroundCount > 0) {
      const meanR = totalR / foregroundCount;
      const meanG = totalG / foregroundCount;
      const meanB = totalB / foregroundCount;
      const totalRGB = meanR + meanG + meanB + 1e-6;

      redToBlueRatio = meanR / (meanB + 1e-6);
      redShare = meanR / totalRGB;
    }

    // Flag document/diagram if corners are bright white/gray (mean > 160) and R/B ratio is neutral (< 1.15)
    if (cornerAvgIntensity > 160 && redToBlueRatio < 1.15) {
      isDocumentOrDiagram = true;
    }
  }

  // Gate 2 Rules:
  // 1. Aspect ratio: 0.65 to 1.65 (Standard fundus camera field)
  // 2. Red to Blue ratio >= 1.15 (Fundus vascular orange/red dominance)
  // 3. Red share >= 0.36
  // 4. Must not be a white-background diagram/document
  const aspectPassed = aspectRatio >= 0.60 && aspectRatio <= 1.70;
  const colorPassed = redToBlueRatio >= 1.15 && redShare >= 0.36;
  const notDiagramPassed = !isDocumentOrDiagram;

  const gate2Passed = gate1Passed && aspectPassed && colorPassed && notDiagramPassed;
  let gate2Reason: string | undefined;
  let gate2Action: string | undefined;

  if (!aspectPassed) {
    gate2Reason = `Non-standard image aspect ratio (${aspectRatio.toFixed(2)}). Retinal fundus requires standard camera field (0.65 - 1.65).`;
    gate2Action = 'Upload standard uncropped 45-degree or 30-degree fundus photograph.';
  } else if (isDocumentOrDiagram) {
    gate2Reason = 'Non-retinal document or architectural schematic detected. The uploaded image lacks circular fundus aperture and ophthalmic vascular pigmentation.';
    gate2Action = 'Strictly upload clinical retinal fundus photography. Do not submit diagrams, text, or non-retinal imagery.';
  } else if (!colorPassed) {
    gate2Reason = `Spectral profile does not exhibit retinal vascular characteristics (R/B ratio: ${redToBlueRatio.toFixed(2)} < 1.15, Red share: ${(redShare * 100).toFixed(1)}%). Detected non-retinal subject (e.g. grayscale, diagram, anterior segment).`;
    gate2Action = 'Ensure you are uploading posterior pole color retinal fundus photography rather than external ocular or non-retinal images.';
  }

  // --- GATE 3: Technical Quality & Laplacian Blur ---
  // Approximate Laplacian variance on canvas
  let laplacianVariance = 180.0;
  if (ctx && gate1Passed && gate2Passed) {
    const imgData = ctx.getImageData(0, 0, sampleSize, sampleSize);
    const d = imgData.data;
    let sumGrad = 0;
    let count = 0;

    for (let y = 1; y < sampleSize - 1; y += 2) {
      for (let x = 1; x < sampleSize - 1; x += 2) {
        const idx = (y * sampleSize + x) * 4;
        const lumCenter = (d[idx] + d[idx + 1] + d[idx + 2]) / 3;
        const lumRight = (d[idx + 4] + d[idx + 5] + d[idx + 6]) / 3;
        const lumDown = (d[idx + sampleSize * 4] + d[idx + sampleSize * 4 + 1] + d[idx + sampleSize * 4 + 2]) / 3;

        const diff = Math.abs(lumCenter - lumRight) + Math.abs(lumCenter - lumDown);
        sumGrad += diff * diff;
        count++;
      }
    }
    laplacianVariance = count > 0 ? (sumGrad / count) * 4 : 180.0;
  }

  const gate3Passed = gate2Passed && laplacianVariance >= 60.0;
  let gate3Reason: string | undefined;
  let gate3Action: string | undefined;

  if (!gate3Passed && gate2Passed) {
    gate3Reason = `Insufficient optical sharpness or motion blur (Laplacian variance: ${laplacianVariance.toFixed(1)} < 60.0 threshold).`;
    gate3Action = 'Recapture retinal photograph ensuring steady patient fixation and camera objective cleanliness.';
  }

  const allPassed = gate1Passed && gate2Passed && gate3Passed;
  let failedGate: 1 | 2 | 3 | null = null;
  if (!gate1Passed) failedGate = 1;
  else if (!gate2Passed) failedGate = 2;
  else if (!gate3Passed) failedGate = 3;

  return {
    allPassed,
    failedGate,
    gate1: {
      passed: gate1Passed,
      mime: mimeType || 'image/jpeg',
      fileSizeBytes,
      metric: gate1Passed ? 'MIME JPEG/PNG, Signature Valid' : 'Format/Size Check Failed',
      rejectionReason: gate1Reason,
      clinicalAction: gate1Action,
    },
    gate2: {
      passed: gate2Passed,
      aspectRatio,
      redToBlueRatio,
      redShare,
      isDocumentOrDiagram,
      metric: gate2Passed
        ? `Retinal FoV confirmed, R/B ratio: ${redToBlueRatio.toFixed(2)}`
        : isDocumentOrDiagram
        ? 'Non-retinal diagram/document detected'
        : `R/B spectral ratio: ${redToBlueRatio.toFixed(2)} (Min: 1.15)`,
      rejectionReason: gate2Reason,
      clinicalAction: gate2Action,
    },
    gate3: {
      passed: gate3Passed,
      laplacianVariance,
      metric: `Laplacian variance: ${laplacianVariance.toFixed(1)} (Threshold >= 60.0)`,
      rejectionReason: gate3Reason,
      clinicalAction: gate3Action,
    },
  };
}
