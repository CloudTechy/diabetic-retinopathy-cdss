export type ColormapType = 'viridis' | 'inferno' | 'magma' | 'cividis';

export interface ColormapDefinition {
  id: ColormapType;
  name: string;
  description: string;
  cssGradient: string;
  stops: [number, number, number][]; // RGB normalized [0-255]
}

export const COLORMAPS: Record<ColormapType, ColormapDefinition> = {
  viridis: {
    id: 'viridis',
    name: 'Viridis (Perceptually Uniform & Safe)',
    description: 'Optimal contrast against retinal fundus orange/red. Deuteranopia & Protanopia safe.',
    cssGradient: 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)',
    stops: [
      [68, 1, 84],
      [72, 35, 107],
      [65, 68, 135],
      [53, 95, 141],
      [42, 120, 142],
      [33, 145, 140],
      [34, 168, 132],
      [68, 191, 112],
      [122, 209, 81],
      [189, 223, 38],
      [253, 231, 37],
    ],
  },
  inferno: {
    id: 'inferno',
    name: 'Inferno (high contrast)',
    description: 'Black to vivid violet, red and yellow. Excellent for microaneurysm focal hot-spots.',
    cssGradient: 'linear-gradient(to right, #000004, #320a5e, #781c6d, #bb3754, #ed6925, #fcb014, #fcffa4)',
    stops: [
      [0, 0, 4],
      [40, 11, 84],
      [101, 21, 110],
      [159, 42, 99],
      [212, 72, 66],
      [245, 125, 21],
      [250, 187, 36],
      [252, 255, 164],
    ],
  },
  magma: {
    id: 'magma',
    name: 'Magma (High-Contrast Attribution)',
    description: 'Monotonically increasing dark purple to peach to white.',
    cssGradient: 'linear-gradient(to right, #000004, #2c115f, #721f81, #b73779, #f1605d, #feb078, #fcfdbf)',
    stops: [
      [0, 0, 4],
      [51, 16, 103],
      [114, 31, 129],
      [183, 55, 121],
      [237, 105, 93],
      [254, 176, 120],
      [252, 253, 191],
    ],
  },
  cividis: {
    id: 'cividis',
    name: 'Cividis (Formally CVD Optimized)',
    description: 'Dark blue to neutral grey to light yellow. Maximum perceptual uniformity under severe CVD.',
    cssGradient: 'linear-gradient(to right, #00204d, #414d6b, #7c7b78, #b9ac70, #ffea46)',
    stops: [
      [0, 32, 77],
      [36, 59, 100],
      [65, 77, 107],
      [97, 98, 111],
      [132, 122, 114],
      [170, 150, 115],
      [211, 184, 110],
      [255, 234, 70],
    ],
  },
};

/**
 * Interpolates RGB color from normalized value [0.0 - 1.0] for a given colormap
 */
export function interpolateColormap(
  value: number,
  colormap: ColormapType = 'viridis'
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));
  const stops = COLORMAPS[colormap].stops;
  const numSegments = stops.length - 1;
  const scaled = clamped * numSegments;
  const idx = Math.floor(scaled);
  const frac = scaled - idx;

  if (idx >= numSegments) {
    return stops[numSegments];
  }

  const [r1, g1, b1] = stops[idx];
  const [r2, g2, b2] = stops[idx + 1];

  const r = Math.round(r1 + (r2 - r1) * frac);
  const g = Math.round(g1 + (g2 - g1) * frac);
  const b = Math.round(b1 + (b2 - b1) * frac);

  return [r, g, b];
}
