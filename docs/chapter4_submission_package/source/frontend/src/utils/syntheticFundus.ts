import { ColormapType, interpolateColormap } from './colormaps';

/**
 * Creates a synthetic retinal fundus photograph canvas for clinical demonstration
 * @param grade ICDR grade (0 = Normal, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = PDR)
 * @param laterality 'OD' (Right Eye) or 'OS' (Left Eye)
 * @param width Canvas width
 * @param height Canvas height
 */
export function generateSyntheticFundus(
  grade: number = 2,
  laterality: 'OD' | 'OS' = 'OD',
  width: number = 800,
  height: number = 800
): { fundusDataUrl: string; rawCanvas: HTMLCanvasElement } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return { fundusDataUrl: '', rawCanvas: canvas };

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.44;

  // 1. Black outer background (aperture boundary)
  ctx.fillStyle = '#08080c';
  ctx.fillRect(0, 0, width, height);

  // 2. Circular Fundus Mask
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.clip();

  // Background Retinal Hue: deep orange-red choroid gradient
  const bgGrad = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius);
  bgGrad.addColorStop(0, '#c34113');
  bgGrad.addColorStop(0.5, '#992a09');
  bgGrad.addColorStop(0.85, '#6c1505');
  bgGrad.addColorStop(1, '#2c0602');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, width, height);

  // Subtle pigment texture
  for (let i = 0; i < 400; i++) {
    const angle = Math.random() * Math.PI * 2;
    const dist = Math.random() * radius;
    const px = cx + Math.cos(angle) * dist;
    const py = cy + Math.sin(angle) * dist;
    ctx.fillStyle = `rgba(${Math.random() > 0.5 ? '180, 50, 10' : '80, 10, 5'}, ${0.05 + Math.random() * 0.08})`;
    ctx.beginPath();
    ctx.arc(px, py, 2 + Math.random() * 6, 0, Math.PI * 2);
    ctx.fill();
  }

  // 3. Optic Disc: Anatomical placement (Nasal side)
  // In OD (Right Eye), nasal is to the left (clinician's left).
  // In OS (Left Eye), nasal is to the right.
  const discX = laterality === 'OD' ? cx - radius * 0.48 : cx + radius * 0.48;
  const discY = cy - radius * 0.05;
  const discRadius = radius * 0.16;

  // Optic disc halo & cup
  const discGrad = ctx.createRadialGradient(discX, discY, discRadius * 0.2, discX, discY, discRadius);
  discGrad.addColorStop(0, '#fff3cc');
  discGrad.addColorStop(0.6, '#ffcc88');
  discGrad.addColorStop(0.9, '#e68844');
  discGrad.addColorStop(1, '#a33311');
  ctx.fillStyle = discGrad;
  ctx.beginPath();
  ctx.arc(discX, discY, discRadius, 0, Math.PI * 2);
  ctx.fill();

  // Optic Cup center
  ctx.fillStyle = '#fffaea';
  ctx.beginPath();
  ctx.arc(discX + (laterality === 'OD' ? 3 : -3), discY, discRadius * 0.45, 0, Math.PI * 2);
  ctx.fill();

  // 4. Macula / Fovea: Temporal side, slightly inferior to disc
  const maculaX = laterality === 'OD' ? cx + radius * 0.28 : cx - radius * 0.28;
  const maculaY = cy + radius * 0.04;
  const maculaGrad = ctx.createRadialGradient(maculaX, maculaY, 2, maculaX, maculaY, radius * 0.22);
  maculaGrad.addColorStop(0, '#360904');
  maculaGrad.addColorStop(0.4, '#581507');
  maculaGrad.addColorStop(1, 'rgba(108, 21, 5, 0)');
  ctx.fillStyle = maculaGrad;
  ctx.beginPath();
  ctx.arc(maculaX, maculaY, radius * 0.22, 0, Math.PI * 2);
  ctx.fill();

  // Foveal reflex point
  ctx.fillStyle = 'rgba(255, 230, 200, 0.4)';
  ctx.beginPath();
  ctx.arc(maculaX, maculaY, 1.5, 0, Math.PI * 2);
  ctx.fill();

  // 5. Retinal Vascular Tree (Superior & Inferior Temporal/Nasal Arcades)
  const drawVesselArcade = (startX: number, startY: number, angleOffset: number, length: number, widthBase: number, isArtery: boolean) => {
    ctx.lineWidth = widthBase;
    ctx.strokeStyle = isArtery ? 'rgba(195, 45, 25, 0.85)' : 'rgba(98, 12, 12, 0.9)';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(startX, startY);

    let curX = startX;
    let curY = startY;
    const steps = 18;
    const stepLen = length / steps;

    for (let s = 1; s <= steps; s++) {
      const curve = Math.sin((s / steps) * Math.PI) * (isArtery ? 18 : 24);
      const angle = angleOffset + (curve * Math.PI) / 180;
      curX += Math.cos(angle) * stepLen;
      curY += Math.sin(angle) * stepLen;
      ctx.lineTo(curX, curY);

      // Micro branches
      if (s % 4 === 0 && s < steps - 2) {
        ctx.save();
        ctx.lineWidth = Math.max(1, ctx.lineWidth * 0.45);
        ctx.beginPath();
        ctx.moveTo(curX, curY);
        const branchAngle = angle + (Math.random() > 0.5 ? 0.6 : -0.6);
        ctx.lineTo(curX + Math.cos(branchAngle) * (stepLen * 2.5), curY + Math.sin(branchAngle) * (stepLen * 2.5));
        ctx.stroke();
        ctx.restore();
      }
    }
    ctx.stroke();
  };

  // Arcades radiating from disc
  const dir = laterality === 'OD' ? 1 : -1;
  // Superior temporal
  drawVesselArcade(discX, discY, -0.4 * dir, radius * 0.85, 4.5, false);
  drawVesselArcade(discX, discY, -0.32 * dir, radius * 0.78, 3.2, true);
  // Inferior temporal
  drawVesselArcade(discX, discY, 0.4 * dir, radius * 0.88, 4.8, false);
  drawVesselArcade(discX, discY, 0.34 * dir, radius * 0.80, 3.4, true);
  // Nasal vessels
  drawVesselArcade(discX, discY, Math.PI - 0.3 * dir, radius * 0.45, 3.0, false);
  drawVesselArcade(discX, discY, Math.PI + 0.3 * dir, radius * 0.45, 3.0, true);

  // 6. Pathology Lesions based on ICDR Grade
  if (grade >= 1) {
    // Microaneurysms (Grade 1+)
    const lesionCount = grade === 1 ? 8 : grade === 2 ? 35 : grade === 3 ? 90 : 120;
    for (let i = 0; i < lesionCount; i++) {
      // Clustered mainly in parafoveal and temporal zones
      const lx = maculaX + (Math.random() - 0.45) * radius * 0.7;
      const ly = maculaY + (Math.random() - 0.45) * radius * 0.7;
      // Microaneurysm: tiny sharp dark red dot
      ctx.fillStyle = 'rgba(75, 4, 4, 0.95)';
      ctx.beginPath();
      ctx.arc(lx, ly, 1.2 + Math.random() * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (grade >= 2) {
    // Hard Exudates (bright yellowish lipid deposits)
    for (let i = 0; i < (grade === 2 ? 15 : 45); i++) {
      const ex = maculaX + (Math.random() - 0.3) * radius * 0.45;
      const ey = maculaY + (Math.random() - 0.5) * radius * 0.45;
      ctx.fillStyle = '#ffea70';
      ctx.beginPath();
      ctx.arc(ex, ey, 2 + Math.random() * 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    // Dot and Blot hemorrhages
    for (let i = 0; i < (grade === 2 ? 12 : 50); i++) {
      const hx = cx + (Math.random() - 0.5) * radius * 0.9;
      const hy = cy + (Math.random() - 0.5) * radius * 0.9;
      ctx.fillStyle = '#5c0505';
      ctx.beginPath();
      ctx.arc(hx, hy, 3 + Math.random() * 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (grade >= 3) {
    // Cotton Wool Spots (fuzzy whitish soft exudates - nerve fiber infarctions)
    for (let i = 0; i < 6; i++) {
      const cwx = cx + (Math.random() - 0.4) * radius * 0.6;
      const cwy = cy + (Math.random() - 0.4) * radius * 0.6;
      const cwGrad = ctx.createRadialGradient(cwx, cwy, 1, cwx, cwy, 14);
      cwGrad.addColorStop(0, 'rgba(255, 255, 230, 0.75)');
      cwGrad.addColorStop(1, 'rgba(255, 255, 230, 0)');
      ctx.fillStyle = cwGrad;
      ctx.beginPath();
      ctx.arc(cwx, cwy, 14, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (grade >= 4) {
    // Proliferative Neovascularization (frond-like irregular vessels)
    ctx.strokeStyle = 'rgba(220, 30, 30, 0.9)';
    ctx.lineWidth = 1.8;
    for (let i = 0; i < 4; i++) {
      let nx = discX + (Math.random() - 0.5) * 40;
      let ny = discY + (Math.random() - 0.5) * 40;
      ctx.beginPath();
      ctx.moveTo(nx, ny);
      for (let j = 0; j < 10; j++) {
        nx += (Math.random() - 0.5) * 12;
        ny += (Math.random() - 0.5) * 12;
        ctx.lineTo(nx, ny);
      }
      ctx.stroke();
    }
  }

  // Vignette boundary shading around circular aperture
  const vigGrad = ctx.createRadialGradient(cx, cy, radius * 0.85, cx, cy, radius);
  vigGrad.addColorStop(0, 'rgba(0,0,0,0)');
  vigGrad.addColorStop(1, 'rgba(0,0,0,0.85)');
  ctx.fillStyle = vigGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();

  return {
    fundusDataUrl: canvas.toDataURL('image/jpeg', 0.92),
    rawCanvas: canvas,
  };
}

/**
 * Creates Grad-CAM saliency activation heatmap overlaid with specific colormap
 * @param grade ICDR grade to place salient hotspots
 * @param laterality 'OD' or 'OS'
 * @param colormap Viridis, Inferno, Magma, Cividis
 * @param threshold Activation cutoff threshold [0.0 - 0.9]
 * @param width Canvas width
 * @param height Canvas height
 */
export function generateSyntheticGradCam(
  grade: number = 2,
  laterality: 'OD' | 'OS' = 'OD',
  colormap: ColormapType = 'viridis',
  threshold: number = 0.25,
  width: number = 800,
  height: number = 800
): { gradcamDataUrl: string; rawCanvas: HTMLCanvasElement } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return { gradcamDataUrl: '', rawCanvas: canvas };

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.44;

  // Activation matrix representation
  const gridW = 32;
  const gridH = 32;
  const activations = new Float32Array(gridW * gridH);

  // Define hot zones based on grade and laterality
  const maculaGridX = (laterality === 'OD' ? 0.64 : 0.36) * gridW;
  const maculaGridY = 0.52 * gridH;
  const arcadeGridY = 0.38 * gridH;

  const centers: { x: number; y: number; sigma: number; weight: number }[] = [];

  if (grade === 0) {
    // Diffuse baseline background attribution
    centers.push({ x: maculaGridX, y: maculaGridY, sigma: 4.5, weight: 0.35 });
  } else if (grade === 1) {
    centers.push({ x: maculaGridX - 2, y: maculaGridY + 2, sigma: 2.2, weight: 0.85 });
    centers.push({ x: maculaGridX + 3, y: maculaGridY - 1, sigma: 1.8, weight: 0.70 });
  } else if (grade === 2) {
    // Inferotemporal / macula clusters
    centers.push({ x: maculaGridX, y: maculaGridY + 3, sigma: 3.2, weight: 0.96 });
    centers.push({ x: maculaGridX - 4, y: arcadeGridY + 5, sigma: 2.8, weight: 0.88 });
    centers.push({ x: maculaGridX + 4, y: maculaGridY - 3, sigma: 2.5, weight: 0.75 });
  } else if (grade === 3) {
    // Multi-quadrant activations
    centers.push({ x: maculaGridX, y: maculaGridY + 4, sigma: 3.8, weight: 0.98 });
    centers.push({ x: maculaGridX - 6, y: 0.3 * gridH, sigma: 3.0, weight: 0.92 });
    centers.push({ x: maculaGridX + 5, y: 0.7 * gridH, sigma: 3.4, weight: 0.89 });
    centers.push({ x: 0.35 * gridW, y: 0.65 * gridH, sigma: 2.6, weight: 0.84 });
  } else {
    // Grade 4: Disc and widespread neovascular zones
    const discGridX = (laterality === 'OD' ? 0.26 : 0.74) * gridW;
    centers.push({ x: discGridX, y: 0.48 * gridH, sigma: 3.5, weight: 1.0 });
    centers.push({ x: maculaGridX, y: maculaGridY, sigma: 4.2, weight: 0.95 });
    centers.push({ x: 0.5 * gridW, y: 0.28 * gridH, sigma: 3.2, weight: 0.91 });
  }

  // Compute Gaussian activation field
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      let val = 0.05; // baseline
      for (const c of centers) {
        const dx = gx - c.x;
        const dy = gy - c.y;
        const d2 = dx * dx + dy * dy;
        val += c.weight * Math.exp(-d2 / (2 * c.sigma * c.sigma));
      }
      activations[gy * gridW + gx] = Math.min(1.0, val);
    }
  }

  // Render to canvas with colormap and threshold
  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      // Check if inside circular fundus
      const dx = px - cx;
      const dy = py - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);

      const pIdx = (py * width + px) * 4;

      if (dist > radius) {
        data[pIdx + 3] = 0; // transparent outside fundus
        continue;
      }

      // Bilinear interpolation on activation grid
      const gx = (px / width) * (gridW - 1);
      const gy = (py / height) * (gridH - 1);
      const gxi = Math.floor(gx);
      const gyi = Math.floor(gy);
      const fracX = gx - gxi;
      const fracY = gy - gyi;

      const a00 = activations[gyi * gridW + gxi];
      const a10 = activations[gyi * gridW + Math.min(gridW - 1, gxi + 1)];
      const a01 = activations[Math.min(gridH - 1, gyi + 1) * gridW + gxi];
      const a11 = activations[Math.min(gridH - 1, gyi + 1) * gridW + Math.min(gridW - 1, gxi + 1)];

      const val = (1 - fracX) * (1 - fracY) * a00 +
                  fracX * (1 - fracY) * a10 +
                  (1 - fracX) * fracY * a01 +
                  fracX * fracY * a11;

      if (val < threshold) {
        // Below threshold: isolate focal lesions by setting transparent
        data[pIdx + 3] = 0;
      } else {
        const [r, g, b] = interpolateColormap(val, colormap);
        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        // Alpha curve scales smoothly from 0 to 255 based on intensity above threshold
        const alphaFrac = (val - threshold) / (1.0 - threshold);
        data[pIdx + 3] = Math.min(255, Math.floor(alphaFrac * 255 * 0.95));
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);

  return {
    gradcamDataUrl: canvas.toDataURL('image/png'),
    rawCanvas: canvas,
  };
}
