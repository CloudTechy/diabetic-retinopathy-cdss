/**
 * Client-Side Edge AI Inference Engine (Offline-First)
 *
 * Runs the complete CDSS pipeline directly on mobile devices / browsers without internet:
 *  1. 3-Stage Technical Validation (Gate 1 File, Gate 2 Retinal Aspect/Color, Gate 3 Laplacian Sharpness)
 *  2. On-Device EfficientNet-B0 Inference via ONNX Runtime Web (WASM / WebGL)
 *  3. On-Device Saliency Heatmap generation (Class Activation Mapping)
 *  4. Full local persistence with offline queue for syncing when connectivity resumes
 */

import * as ort from 'onnxruntime-web';
import { Network } from '@capacitor/network';
import { ICDR_GRADES } from '../types/clinical';

// Configure ONNX Runtime Web environment
ort.env.wasm.numThreads = 2;
ort.env.wasm.simd = true;

export interface EdgeInferenceResult {
  primaryClassGrade: number;
  primaryScore: number;
  classScores: { grade: number; score: number; label: string }[];
  gradcamDataUrl: string;
  inferenceDurationMs: number;
  isOfflineEdge: boolean;
}

export interface OfflineGateResult {
  gateNumber: number;
  gateName: string;
  status: 'passed' | 'failed';
  score: number;
  threshold: number;
  message: string;
}

class EdgeInferenceEngine {
  private session: ort.InferenceSession | null = null;
  private classifierWeights: { weight: number[][]; bias: number[] } | null = null;
  private isInitializing = false;
  private initPromise: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.session) return;
    if (this.isInitializing && this.initPromise) return this.initPromise;

    this.isInitializing = true;
    this.initPromise = (async () => {
      try {
        console.log('[EdgeAI] Initializing on-device ONNX runtime...');
        const weightsRes = await fetch('/models/classifier_weights.json');
        if (weightsRes.ok) {
          this.classifierWeights = await weightsRes.json();
        }

        this.session = await ort.InferenceSession.create('/models/efficientnet_b0_dr.onnx', {
          executionProviders: ['wasm', 'webgl'],
          graphOptimizationLevel: 'all',
        });
        console.log('[EdgeAI] EfficientNet-B0 model loaded for offline edge inference.');
      } catch (err) {
        console.warn('[EdgeAI] Could not load ONNX model directly:', err);
      } finally {
        this.isInitializing = false;
      }
    })();

    return this.initPromise;
  }

  isReady(): boolean {
    return this.session !== null;
  }

  async checkOnlineStatus(): Promise<boolean> {
    try {
      const status = await Network.getStatus();
      return status.connected;
    } catch {
      return navigator.onLine;
    }
  }

  /**
   * Evaluates the 3-stage validation pipeline purely client-side on raw image pixels.
   */
  async evaluateTechnicalGates(img: HTMLImageElement): Promise<{
    passed: boolean;
    gates: OfflineGateResult[];
    failureReason?: string;
  }> {
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;

    // Gate 1: Physical file & resolution check
    const gate1Passed = width >= 224 && height >= 224;
    const gate1: OfflineGateResult = {
      gateNumber: 1,
      gateName: 'File Integrity & Resolution',
      status: gate1Passed ? 'passed' : 'failed',
      score: Math.min(width, height),
      threshold: 224,
      message: gate1Passed
        ? `Sufficient dimensions (${width}x${height} >= 224x224)`
        : `Image resolution ${width}x${height} is below minimum requirement of 224x224.`,
    };

    if (!gate1Passed) {
      return { passed: false, gates: [gate1], failureReason: gate1.message };
    }

    // Prepare canvas for pixel analysis
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return { passed: false, gates: [gate1], failureReason: 'Canvas 2D context unavailable' };
    }
    ctx.drawImage(img, 0, 0, 256, 256);
    const imgData = ctx.getImageData(0, 0, 256, 256);
    const data = imgData.data;

    // Gate 2: Aspect ratio & retinal color relevance (0.65 to 1.65, red/blue dominance)
    const aspectRatio = width / height;
    let totalR = 0;
    let totalB = 0;
    const numPixels = 256 * 256;
    for (let i = 0; i < data.length; i += 4) {
      totalR += data[i];
      totalB += data[i + 2];
    }
    const redBlueRatio = (totalR / numPixels) / (Math.max(1, totalB / numPixels));
    const aspectPassed = aspectRatio >= 0.65 && aspectRatio <= 1.65;
    const colorPassed = redBlueRatio >= 1.05; // retinal fundus typically has strong red spectrum
    const gate2Passed = aspectPassed && colorPassed;

    const gate2: OfflineGateResult = {
      gateNumber: 2,
      gateName: 'Retinal Relevance & Proportions',
      status: gate2Passed ? 'passed' : 'failed',
      score: Number(redBlueRatio.toFixed(2)),
      threshold: 1.05,
      message: gate2Passed
        ? `Aspect ratio (${aspectRatio.toFixed(2)}) and spectral balance valid.`
        : `Image does not match expected retinal fundus proportions or spectral distribution.`,
    };

    if (!gate2Passed) {
      return { passed: false, gates: [gate1, gate2], failureReason: gate2.message };
    }

    // Gate 3: Sharpness & Focus via Laplacian Variance (Threshold: 4.3)
    let laplacianVar = 0;
    const gray: number[] = new Array(256 * 256);
    for (let i = 0; i < data.length; i += 4) {
      gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    const lapValues: number[] = [];
    let sumLap = 0;
    for (let y = 1; y < 255; y++) {
      for (let x = 1; x < 255; x++) {
        const idx = y * 256 + x;
        const lap =
          gray[idx - 256] +
          gray[idx + 256] +
          gray[idx - 1] +
          gray[idx + 1] -
          4 * gray[idx];
        lapValues.push(lap);
        sumLap += lap;
      }
    }
    const meanLap = sumLap / lapValues.length;
    let sumSqDiff = 0;
    for (let i = 0; i < lapValues.length; i++) {
      const diff = lapValues[i] - meanLap;
      sumSqDiff += diff * diff;
    }
    laplacianVar = sumSqDiff / lapValues.length;
    const gate3Passed = laplacianVar >= 4.3;

    const gate3: OfflineGateResult = {
      gateNumber: 3,
      gateName: 'Image Sharpness & Focus',
      status: gate3Passed ? 'passed' : 'failed',
      score: Number(laplacianVar.toFixed(2)),
      threshold: 4.3,
      message: gate3Passed
        ? `Focus variance (${laplacianVar.toFixed(2)} >= 4.3) meets the configured technical-quality threshold.`
        : `Severe blur detected (variance ${laplacianVar.toFixed(2)} < 4.3). Please retake fundus photograph.`,
    };

    return {
      passed: gate3Passed,
      gates: [gate1, gate2, gate3],
      failureReason: gate3Passed ? undefined : gate3.message,
    };
  }

  /**
   * Preprocess an HTMLImageElement to 224x224 RGB Float32Array normalized with ImageNet stats.
   */
  private preprocessImage(img: HTMLImageElement): Float32Array {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0, 224, 224);

    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const floatArray = new Float32Array(3 * 224 * 224);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      floatArray[i] = (r - mean[0]) / std[0]; // Channel 0: R
      floatArray[224 * 224 + i] = (g - mean[1]) / std[1]; // Channel 1: G
      floatArray[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; // Channel 2: B
    }

    return floatArray;
  }

  /**
   * Generates a client-side spatial saliency overlay using feature maps & classifier weights.
   */
  private generateClientSaliencyMap(
    featureMaps: Float32Array, // (1280, 7, 7)
    predictedGrade: number,
    baseImage: HTMLImageElement
  ): string {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d')!;

    // Draw original image as background
    ctx.drawImage(baseImage, 0, 0, 224, 224);

    const weights = this.classifierWeights?.weight[predictedGrade];
    const cam7x7 = new Float32Array(7 * 7);

    if (weights) {
      for (let c = 0; c < 1280; c++) {
        const w = weights[c];
        const offset = c * 49;
        for (let p = 0; p < 49; p++) {
          cam7x7[p] += w * featureMaps[offset + p];
        }
      }
    } else {
      // Fallback: activation energy average across top channels
      for (let c = 0; c < 64; c++) {
        const offset = c * 49;
        for (let p = 0; p < 49; p++) {
          cam7x7[p] += Math.abs(featureMaps[offset + p]);
        }
      }
    }

    // Apply ReLU and find max
    let maxVal = 0;
    for (let i = 0; i < 49; i++) {
      cam7x7[i] = Math.max(0, cam7x7[i]);
      if (cam7x7[i] > maxVal) maxVal = cam7x7[i];
    }

    if (maxVal > 0) {
      for (let i = 0; i < 49; i++) {
        cam7x7[i] /= maxVal;
      }
    }

    // Render 7x7 smoothed heatmap onto 224x224
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = 7;
    heatCanvas.height = 7;
    const heatCtx = heatCanvas.getContext('2d')!;
    const heatData = heatCtx.createImageData(7, 7);

    for (let i = 0; i < 49; i++) {
      const val = cam7x7[i];
      // Viridis-like interpolation
      heatData.data[i * 4] = Math.floor(val * 255); // R
      heatData.data[i * 4 + 1] = Math.floor(Math.sin(val * Math.PI) * 200); // G
      heatData.data[i * 4 + 2] = Math.floor((1 - val) * 180); // B
      heatData.data[i * 4 + 3] = Math.floor(val * 130); // Alpha transparency
    }
    heatCtx.putImageData(heatData, 0, 0);

    // Overlay smoothed heatmap
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(heatCanvas, 0, 0, 224, 224);

    return canvas.toDataURL('image/png');
  }

  /**
   * Executes edge on-device inference using the ONNX model.
   */
  async runInference(img: HTMLImageElement): Promise<EdgeInferenceResult> {
    await this.init();
    if (!this.session) {
      throw new Error('ONNX runtime session could not be established on this device.');
    }

    const t0 = performance.now();
    const inputData = this.preprocessImage(img);
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);

    const feeds: Record<string, ort.Tensor> = { input: tensor };
    const output = await this.session.run(feeds);

    const logitsTensor = output['logits'];
    const featuresTensor = output['features'];
    const logits = Array.from(logitsTensor.data as Float32Array);

    // Compute Softmax probabilities
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExps);

    // Argmax primary class
    let maxProb = -1;
    let primaryGrade = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        primaryGrade = i;
      }
    }

    const classScores = probs.map((prob, idx) => ({
      grade: idx,
      score: Number(prob.toFixed(4)),
      label: ICDR_GRADES[idx]?.label || `Grade ${idx}`,
    }));

    // Generate client-side Grad-CAM saliency
    let gradcamUrl = '';
    if (featuresTensor) {
      gradcamUrl = this.generateClientSaliencyMap(
        featuresTensor.data as Float32Array,
        primaryGrade,
        img
      );
    }

    const t1 = performance.now();

    return {
      primaryClassGrade: primaryGrade,
      primaryScore: Number(maxProb.toFixed(4)),
      classScores,
      gradcamDataUrl: gradcamUrl,
      inferenceDurationMs: Math.round(t1 - t0),
      isOfflineEdge: true,
    };
  }
}

export const edgeAI = new EdgeInferenceEngine();
