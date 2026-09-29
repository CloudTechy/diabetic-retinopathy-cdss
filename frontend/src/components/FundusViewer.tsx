import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Columns,
  Eye,
  Sliders,
  Layers,
  Sparkles,
  Info
} from 'lucide-react';
import { ColormapType, COLORMAPS } from '../utils/colormaps';
import { EyeLaterality } from '../types/clinical';
import { generateSyntheticGradCam } from '../utils/syntheticFundus';

interface FundusViewerProps {
  imageUrl: string;
  gradcamUrl?: string;
  laterality: EyeLaterality;
  nativeResolution?: string;
  grade?: number;
  readOnly?: boolean;
}

export const FundusViewer: React.FC<FundusViewerProps> = ({
  imageUrl,
  gradcamUrl: initialGradcamUrl,
  laterality,
  nativeResolution = '2240x1488 px',
  grade = 2,
}) => {
  // Transform State (Affine pan/zoom)
  const [scale, setScale] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Blending & Attribution Controls
  const [opacity, setOpacity] = useState<number>(45);
  const [isBlinking, setIsBlinking] = useState<boolean>(false);
  const [colormap, setColormap] = useState<ColormapType>('viridis');
  const [threshold, setThreshold] = useState<number>(0.25);
  const [sideBySide, setSideBySide] = useState<boolean>(false);
  const [showControlsDrawer, setShowControlsDrawer] = useState<boolean>(false);

  // Dynamic Grad-CAM generation if colormap or threshold changes
  const [gradcamSrc, setGradcamSrc] = useState<string>(initialGradcamUrl || '');

  // Regenerate Grad-CAM when colormap or threshold changes
  useEffect(() => {
    const res = generateSyntheticGradCam(grade, laterality, colormap, threshold);
    setGradcamSrc(res.gradcamDataUrl);
  }, [grade, laterality, colormap, threshold]);

  const containerRef = useRef<HTMLDivElement>(null);

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    setScale((prev) => Math.min(8.0, +(prev * 1.25).toFixed(2)));
  }, []);

  const handleZoomOut = useCallback(() => {
    setScale((prev) => Math.max(0.75, +(prev / 1.25).toFixed(2)));
  }, []);

  const handleReset = useCallback(() => {
    setScale(1.0);
    setPan({ x: 0, y: 0 });
  }, []);

  // Keyboard navigation contract
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if focus is in an input or textarea
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        handleZoomIn();
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        handleZoomOut();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReset();
      } else if (e.code === 'Space' && !e.repeat) {
        // Spacebar hold blinks Grad-CAM
        e.preventDefault();
        setIsBlinking(true);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setPan((p) => ({ ...p, y: p.y + 30 }));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setPan((p) => ({ ...p, y: p.y - 30 }));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setPan((p) => ({ ...p, x: p.x + 30 }));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setPan((p) => ({ ...p, x: p.x - 30 }));
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsBlinking(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleZoomIn, handleZoomOut, handleReset]);

  // Pan via Mouse Drag
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // primary click only
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Wheel zoom centered at viewport
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => {
      const next = +(prev * zoomFactor).toFixed(2);
      return Math.max(0.75, Math.min(8.0, next));
    });
  };

  const activeOpacity = isBlinking ? 0 : opacity / 100;

  return (
    <section
      aria-label="Retinal Fundus and Grad-CAM Attribution Viewer"
      className="relative flex flex-col w-full h-[620px] bg-clinical-surface-dark rounded-xl overflow-hidden border border-slate-800 shadow-xl select-none"
      role="region"
    >
      {/* Viewer Control Toolbar */}
      <header className="flex flex-wrap items-center justify-between px-3 sm:px-4 py-2 bg-slate-900/95 backdrop-blur border-b border-slate-800 text-slate-200 z-20 gap-2">
        {/* Left: Anatomical & Resolution Metadata */}
        <div className="flex items-center space-x-2">
          <span
            className={`text-xs font-mono uppercase px-2 py-0.5 rounded font-bold ${
              laterality === 'OD' ? 'bg-teal-900/80 text-teal-300 border border-teal-700' : 'bg-blue-900/80 text-blue-300 border border-blue-700'
            }`}
          >
            {laterality === 'OD' ? 'OD (Right Eye)' : 'OS (Left Eye)'}
          </span>
          <span className="text-xs text-slate-400 font-mono hidden md:inline">
            Native: {nativeResolution}
          </span>
        </div>

        {/* Center: Zoom & Ergonomic Actions */}
        <div className="flex items-center space-x-1" role="toolbar" aria-label="Viewer zoom and display controls">
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-teal-400 transition"
            title="Zoom In (+)"
            aria-label="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-teal-400 transition"
            title="Zoom Out (-)"
            aria-label="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono text-teal-400 px-1 font-semibold min-w-[42px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={handleReset}
            className="px-2 py-1 text-xs font-mono rounded text-slate-300 hover:text-white hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-teal-400 transition"
            title="Reset Zoom & Pan (R)"
            aria-label="Reset zoom and pan"
          >
            <RotateCcw className="w-3.5 h-3.5 inline mr-1" />
            Reset
          </button>

          <div className="h-4 w-px bg-slate-700 mx-1" aria-hidden="true" />

          {/* Side-by-Side Mode Toggle */}
          <button
            type="button"
            onClick={() => setSideBySide((prev) => !prev)}
            className={`px-2 py-1 text-xs rounded transition flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-teal-400 ${
              sideBySide ? 'bg-teal-700 text-white font-bold' : 'text-slate-300 hover:bg-slate-800'
            }`}
            aria-pressed={sideBySide}
            title="Toggle Split Dual Synchronized View"
          >
            <Columns className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Side-by-Side</span>
          </button>

          {/* Quick Blink Button (Flicker inspection) */}
          <button
            type="button"
            onMouseDown={() => setIsBlinking(true)}
            onMouseUp={() => setIsBlinking(false)}
            onTouchStart={() => setIsBlinking(true)}
            onTouchEnd={() => setIsBlinking(false)}
            className={`px-2 py-1 text-xs rounded transition flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-teal-400 ${
              isBlinking ? 'bg-amber-600 text-white font-bold' : 'text-slate-300 hover:bg-slate-800'
            }`}
            title="Press and hold to blink overlay off (Spacebar)"
            aria-label="Hold to blink overlay"
          >
            <Eye className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Blink (Space)</span>
          </button>

          <button
            type="button"
            onClick={() => setShowControlsDrawer((prev) => !prev)}
            className={`p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-teal-400 transition ${
              showControlsDrawer ? 'bg-slate-800 text-teal-400' : ''
            }`}
            title="Toggle Advanced Blending Filters"
            aria-label="Toggle Advanced Filters"
          >
            <Sliders className="w-4 h-4" />
          </button>
        </div>

        {/* Right: Grad-CAM Continuous Opacity Slider */}
        <div className="flex items-center space-x-2.5">
          <label htmlFor="gradcam-opacity-slider" className="text-xs font-medium text-slate-300 flex items-center gap-1">
            <Layers className="w-3 h-3 text-teal-400" />
            <span className="hidden sm:inline">Grad-CAM:</span>
          </label>
          <input
            id="gradcam-opacity-slider"
            type="range"
            min="0"
            max="100"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-24 sm:w-28 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400 focus-visible:ring-2 focus-visible:ring-teal-400"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={opacity}
            aria-label="Grad-CAM Heatmap Opacity Percentage"
          />
          <span className="text-xs font-mono w-9 text-right text-teal-400 font-bold" aria-live="polite">
            {isBlinking ? '0%' : `${opacity}%`}
          </span>
        </div>
      </header>

      {/* Advanced Blending & Filter Settings Drawer */}
      {showControlsDrawer && (
        <div className="bg-slate-900 border-b border-slate-800 p-3 px-4 text-xs text-slate-300 flex flex-wrap items-center justify-between gap-4 z-20">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label htmlFor="colormap-select" className="font-semibold text-slate-300">
                Colormap:
              </label>
              <select
                id="colormap-select"
                value={colormap}
                onChange={(e) => setColormap(e.target.value as ColormapType)}
                className="bg-slate-800 border border-slate-700 text-slate-100 text-xs rounded px-2.5 py-1 focus-visible:ring-2 focus-visible:ring-teal-400"
              >
                {Object.values(COLORMAPS).map((cmap) => (
                  <option key={cmap.id} value={cmap.id}>
                    {cmap.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <label htmlFor="threshold-slider" className="font-semibold text-slate-300">
                Contour Threshold:
              </label>
              <input
                id="threshold-slider"
                type="range"
                min="0"
                max="80"
                value={threshold * 100}
                onChange={(e) => setThreshold(Number(e.target.value) / 100)}
                className="w-20 h-1.5 bg-slate-700 rounded appearance-none cursor-pointer accent-amber-400"
              />
              <span className="font-mono text-amber-400 font-bold w-10">
                {(threshold).toFixed(2)}
              </span>
            </div>
          </div>

          <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-teal-400 flex-shrink-0" />
            <span>Colormaps are perceptually uniform and CVD accessible (Viridis/Inferno).</span>
          </div>
        </div>
      )}

      {/* Main Viewport Container */}
      <div
        ref={containerRef}
        className={`relative flex-1 w-full h-full overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing ${
          sideBySide ? 'grid grid-cols-2 divide-x divide-slate-800' : ''
        }`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        tabIndex={0}
        role="application"
        aria-label="Interactive fundus canvas. Use mouse drag or arrow keys to pan, plus and minus keys to zoom."
      >
        {/* Pane 1 (Left / Primary View) */}
        <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
          <div
            className="absolute transition-transform duration-75 origin-center pointer-events-none"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            }}
          >
            {/* Base Layer: High-Res Fundus Photography */}
            <img
              src={imageUrl}
              alt="Raw retinal fundus photography"
              className="max-w-none w-[560px] h-[560px] object-contain rounded-full shadow-2xl"
              draggable={false}
            />

            {/* Attribution Layer: Grad-CAM Saliency Heatmap (Overlay) */}
            {gradcamSrc && !sideBySide && (
              <img
                src={gradcamSrc}
                alt="Grad-CAM Saliency Attribution Heatmap"
                className="absolute inset-0 max-w-none w-[560px] h-[560px] object-contain rounded-full pointer-events-none transition-opacity duration-150"
                style={{ opacity: activeOpacity }}
                draggable={false}
              />
            )}
          </div>

          {/* Subtitle tag */}
          {sideBySide && (
            <div className="absolute top-3 left-3 bg-slate-900/80 px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 border border-slate-700">
              Raw Fundus Photography
            </div>
          )}
        </div>

        {/* Pane 2 (Side-by-Side Right Synchronized Pane) */}
        {sideBySide && (
          <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
            <div
              className="absolute transition-transform duration-75 origin-center pointer-events-none"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
              }}
            >
              {/* Base Fundus */}
              <img
                src={imageUrl}
                alt="Raw fundus photography reference"
                className="max-w-none w-[560px] h-[560px] object-contain rounded-full shadow-2xl"
                draggable={false}
              />
              {/* Overlaid Heatmap */}
              {gradcamSrc && (
                <img
                  src={gradcamSrc}
                  alt="Grad-CAM attribution overlay"
                  className="absolute inset-0 max-w-none w-[560px] h-[560px] object-contain rounded-full pointer-events-none transition-opacity duration-150"
                  style={{ opacity: activeOpacity }}
                  draggable={false}
                />
              )}
            </div>

            <div className="absolute top-3 left-3 bg-slate-900/80 px-2.5 py-1 rounded text-[11px] font-mono text-teal-300 border border-teal-700 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-teal-400" />
              Raw Fundus + Grad-CAM ({Math.round(activeOpacity * 100)}%)
            </div>
          </div>
        )}
      </div>

      {/* Bottom Colormap & Attribution Scale Legend */}
      <footer className="flex flex-wrap items-center justify-between px-3 sm:px-4 py-2 bg-slate-900/95 border-t border-slate-800 text-xs text-slate-400 z-20 gap-2">
        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-medium text-slate-300">Colormap:</span>
          <span className="font-semibold text-slate-200 capitalize">{colormap}</span>
          <span className="text-[11px] text-slate-500 hidden sm:inline">•</span>
          <span className="text-[11px] text-slate-400 hidden sm:inline">
            {COLORMAPS[colormap].description.split('.')[0]}
          </span>
        </div>

        {/* Visual Attribution Legend Swatch */}
        <div className="flex items-center space-x-2" aria-hidden="true">
          <span className="text-[10px] text-slate-400 font-mono">0.0 (Baseline)</span>
          <div
            className="w-28 sm:w-36 h-2 rounded border border-slate-700 shadow-inner"
            style={{ background: COLORMAPS[colormap].cssGradient }}
          />
          <span className="text-[10px] text-teal-400 font-mono font-bold">1.0 (Peak Saliency)</span>
        </div>
      </footer>
    </section>
  );
};
