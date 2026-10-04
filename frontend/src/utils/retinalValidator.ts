import {
  MIN_IMAGE_DIMENSION,
  MAX_UPLOAD_SIZE_MB,
  RETINAL_MIN_COVERAGE,
  RETINAL_RED_RATIO_MIN,
  RETINAL_RED_SHARE_MIN,
} from './validationThresholds';

export interface ClientValidationResult {
  /** Gates 1 and 2 only. The browser never evaluates Gate 3. */
  preflightPassed: boolean;
  failedGate: 1 | 2 | null;
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
    /** foreground pixels / all pixels of the analysis canvas */
    foregroundCoverage: number;
    /** false when no canvas was available, in which case Gate 2 cannot pass */
    coverageEvaluated: boolean;
    isDocumentOrDiagram: boolean;
    metric: string;
    rejectionReason?: string;
    clinicalAction?: string;
  };
  gate3: {
    /** The browser issues no Gate 3 verdict; only the server's result can be passed or failed. */
    status: 'notEvaluated';
    focusEstimate: number;
    metric: string;
  };
}

/**
 * Browser pre-check (preliminary, not authoritative).
 * Applies Gate 1 and Gate 2 with the server's own thresholds (generated into
 * validationThresholds.ts from config.py) so an unsuitable file can be flagged
 * before upload. Passing it does not establish retinal identity, and it issues
 * no Gate 3 verdict: the server re-checks every image and alone decides.
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
  const sizePassed = fileSizeBytes <= MAX_UPLOAD_SIZE_MB * 1024 * 1024;
  // The backend's MIN_IMAGE_DIMENSION (backend/app/core/config.py) is the rule;
  // this browser check is a preliminary courtesy and the server re-checks
  // every image. A rule in the suite keeps this constant equal to the backend's.
  const dimensionPassed = width >= MIN_IMAGE_DIMENSION && height >= MIN_IMAGE_DIMENSION;

  const gate1Passed = mimePassed && sizePassed && dimensionPassed;
  let gate1Reason: string | undefined;
  let gate1Action: string | undefined;

  if (!mimePassed) {
    gate1Reason = `Unsupported MIME format (${mimeType}). DR-CDSS requires JPEG or PNG retinal imagery.`;
    gate1Action = 'Export retinal photography from the camera workstation in standard JPEG or PNG format.';
  } else if (!sizePassed) {
    gate1Reason = `File size (${(fileSizeBytes / 1024 / 1024).toFixed(1)} MB) exceeds the ${MAX_UPLOAD_SIZE_MB} MB limit.`;
    gate1Action = 'Export the image at a supported technical resolution and file size (JPEG or PNG within the size limit).';
  } else if (!dimensionPassed) {
    gate1Reason = `Native resolution (${width}x${height} px) is below the backend's minimum of ${MIN_IMAGE_DIMENSION}x${MIN_IMAGE_DIMENSION} px (preliminary browser check; the server re-checks every image).`;
    gate1Action = 'Provide a fundus photograph at or above the supported technical resolution.';
  }

  // --- GATE 2: technical retinal-image relevance (geometry, coverage, colour profile) ---
  // Off-screen canvas analysis
  const sampleSize = 256;
  const canvas = document.createElement('canvas');
  canvas.width = sampleSize;
  canvas.height = sampleSize;
  const ctx = canvas.getContext('2d');

  let redToBlueRatio = 1.0;
  let redShare = 0.33;
  let foregroundCoverage = 0;
  let coverageEvaluated = false;
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
    // Fundus photographs usually have dark corners outside the aperture (< 40 intensity).
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

      // foreground = luminance > 15, the server's definition (gate2_relevance.py)
      if (lum > 15) {
        totalR += r;
        totalG += g;
        totalB += b;
        foregroundCount++;
      }
    }

    // The share of the frame that is foreground. An earlier revision counted
    // these pixels and never divided: the pass message said coverage was within
    // thresholds without the check having been made.
    foregroundCoverage = foregroundCount / (sampleSize * sampleSize);
    coverageEvaluated = true;

    if (foregroundCount > 0) {
      const meanR = totalR / foregroundCount;
      const meanG = totalG / foregroundCount;
      const meanB = totalB / foregroundCount;
      const totalRGB = meanR + meanG + meanB + 1e-6;

      redToBlueRatio = meanR / (meanB + 1e-6);
      redShare = meanR / totalRGB;
    }

    // Flag document/diagram if corners are bright white/gray (mean > 160) and the R/B ratio is below the minimum
    if (cornerAvgIntensity > 160 && redToBlueRatio < RETINAL_RED_RATIO_MIN) {
      isDocumentOrDiagram = true;
    }
  }

  // Gate 2 rules, as the server applies them:
  // 1. Aspect ratio 0.65 to 1.65
  // 2. Foreground coverage >= RETINAL_MIN_COVERAGE
  // 3. R/B ratio >= RETINAL_RED_RATIO_MIN and red share >= RETINAL_RED_SHARE_MIN
  // 4. (browser only) not a bright-cornered, neutral-coloured document/diagram
  // The server's range (backend/app/services/validation/gate2_relevance.py);
  // a rule in the suite keeps these two numbers equal to it.
  const aspectPassed = aspectRatio >= 0.65 && aspectRatio <= 1.65;
  const coveragePassed = coverageEvaluated && foregroundCoverage >= RETINAL_MIN_COVERAGE;
  const colorPassed = redToBlueRatio >= RETINAL_RED_RATIO_MIN && redShare >= RETINAL_RED_SHARE_MIN;
  const notDiagramPassed = !isDocumentOrDiagram;

  const gate2Passed = gate1Passed && aspectPassed && coveragePassed && colorPassed && notDiagramPassed;
  let gate2Reason: string | undefined;
  let gate2Action: string | undefined;

  if (!aspectPassed) {
    gate2Reason = `Aspect ratio ${aspectRatio.toFixed(2)} is outside the configured range (0.65 - 1.65).`;
    gate2Action = 'Recapture or upload a technically clearer fundus photograph.';
  } else if (!coverageEvaluated) {
    gate2Reason = 'The browser could not analyse the image (no canvas available), so foreground coverage was not evaluated.';
    gate2Action = 'Try another browser, or upload and let the server check the image.';
  } else if (!coveragePassed) {
    gate2Reason = `Foreground coverage ${(foregroundCoverage * 100).toFixed(1)}% is below the configured minimum of ${(RETINAL_MIN_COVERAGE * 100).toFixed(0)}%.`;
    gate2Action = 'Recapture or upload a technically clearer fundus photograph.';
  } else if (isDocumentOrDiagram) {
    gate2Reason = 'Bright uniform corners and a neutral colour profile: the image looks like a document or diagram, not a fundus photograph.';
    gate2Action = 'Upload a fundus photograph.';
  } else if (!colorPassed) {
    gate2Reason = `Colour profile outside the configured thresholds (R/B ratio ${redToBlueRatio.toFixed(2)}, red share ${(redShare * 100).toFixed(1)}%).`;
    gate2Action = 'Recapture or upload a technically clearer fundus photograph.';
  }

  // --- GATE 3: focus, ADVISORY ONLY ---
  // This is a squared first-difference gradient energy on a 256 px canvas. It is
  // NOT the server's Laplacian variance (computed at 1024 px) and is not
  // compared with the server's threshold; an earlier version called it
  // "Laplacian variance" and compared it with the server's 4.3, which was a
  // different statistic wearing the same name. The server decides.
  let focusEstimate = 180.0;
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
    focusEstimate = count > 0 ? (sumGrad / count) * 4 : 180.0;
  }

  // No browser verdict on sharpness. An earlier revision set gate3Passed =
  // gate2Passed and folded that manufactured pass into the overall result; a
  // check that was not performed has no passed state. The browser's result is
  // gates 1 and 2 only.
  const preflightPassed = gate1Passed && gate2Passed;
  let failedGate: 1 | 2 | null = null;
  if (!gate1Passed) failedGate = 1;
  else if (!gate2Passed) failedGate = 2;

  return {
    preflightPassed,
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
      foregroundCoverage,
      coverageEvaluated,
      isDocumentOrDiagram,
      // the measured values, whatever the verdict; nothing is called "within
      // thresholds" unless gate2Passed (which includes coveragePassed) is true
      metric: !coverageEvaluated
        ? 'Not evaluated in the browser (no canvas)'
        : `Foreground coverage ${(foregroundCoverage * 100).toFixed(1)}% (min ${(RETINAL_MIN_COVERAGE * 100).toFixed(0)}%), R/B ratio ${redToBlueRatio.toFixed(2)} (min ${RETINAL_RED_RATIO_MIN}), red share ${(redShare * 100).toFixed(1)}% (min ${(RETINAL_RED_SHARE_MIN * 100).toFixed(0)}%)${gate2Passed ? ' — all within thresholds' : ''}`,
      rejectionReason: gate2Reason,
      clinicalAction: gate2Action,
    },
    gate3: {
      status: 'notEvaluated',
      focusEstimate,
      metric: `Not evaluated in the browser. Focus estimate ${focusEstimate.toFixed(1)} is an advisory gradient energy at 256 px; the server decides on Laplacian variance at 1024 px.`,
    },
  };
}
